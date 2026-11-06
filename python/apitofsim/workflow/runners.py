from collections import deque
from dataclasses import dataclass, field
from functools import cache
from operator import index
from typing import Any, Callable, Optional, Tuple

import numpy
from pint import get_application_registry

from apitofsim import apitofsimraw as raw
from apitofsim.api import (
    ApiTofError,
    ApiTofOverflowError,
    CollisionTimeOptions,
    Histogram,
    MassSpecBatch,
    MassSpecFinalResult,
    MassSpecInputFragmentationPathway,
    MassSpecIntermediateCounter,
    MassSpecLogItem,
    MassSpecResult,
    MassSpecSubstanceSingleInput,
    MassSpecSubstanceTreeInput,
    MassSpectrometer,
    MSSubstanceTreeNode,
    MSSubstanceTreePathway,
    SampleMode,
    VibEnergyIndex,
    VibEnergyMode,
    scaled_density,
    scaled_rate_const,
)
from apitofsim.config import collision_time_options, simulation_option

from .base import SimulationMode
from .db import (
    ExperimentDatabase,
    SuperClusterDatabase,
)
from .db_utils import get_or_insert, get_through_join_else, insert_via_arrow

ureg = get_application_registry()
Q_ = ureg.Quantity


def counter_fragmented_total(counters):
    return int(counters.n_fragmented_total.sum())


@dataclass(frozen=True)
class PreparedGroup:
    """One parent; pathway/product order is the native event-index order."""

    cluster_id: int
    pathway_ids: tuple[int, ...]
    product_ids: tuple[tuple[int, int], ...]
    substance: Any
    cluster_label: str
    tree_cluster_ids: dict[int, int] | None = None


@dataclass(frozen=True)
class PreparedSimulation:
    """Inputs bound to a started run and the runner that prepared them."""

    run_id: int
    mass_spec: MassSpectrometer
    groups: tuple[PreparedGroup, ...]
    dos_histogram_id: int
    rate_histogram_id: int
    _runner: object = field(repr=False, compare=False)
    collision_time: CollisionTimeOptions = field(default_factory=CollisionTimeOptions)
    realizations: int = 100


@dataclass(frozen=True)
class RecordedMassSpecResult:
    """Root completion; provisional until any outer recording scope commits."""

    run_id: int
    cluster_id: int | None
    pathway_ids: tuple[int, ...]
    result_id: int
    result: MassSpecResult


@dataclass(frozen=True)
class RecordedFailure:
    """Immutable failure identity in the recording stream's result database."""

    failure_id: int
    run_id: int
    cluster_id: int | None
    pathway_ids: tuple[int, ...]


def recorded_failure(error: BaseException) -> RecordedFailure | None:
    """Return this exception's inserted failure, including startup errors.

    Return None if insertion failed, was rolled back, or the error is
    consumer-owned. The same reference is available as stream.failure, even
    if event linking fails. Inside an open recording transaction it is
    provisional until commit. Enrich before serialization,
    or forward the reference explicitly; serializers may drop exception state.
    """
    return error.__dict__.get("_apitofsim_recorded_failure")


class RecordingMassSpecIterator:
    """Internal control stream; native code owns event capture and persistence.

    Python receives event batches only when requested. The outer workflow owns
    the experiment transaction; an individual invocation creates its own scope.
    """

    def __init__(
        self,
        db,
        run_id,
        mass_spec,
        substance,
        N,
        *,
        cluster_id=None,
        pathway_id=None,
        pathway_ids=(),
        tree_cluster_ids=None,
        max_batch_items=262144,
        max_pending_events=262144,
        record_events=None,
        python_events=False,
        ordered=None,
        _progress=True,
        _logs=True,
        **options,
    ):
        from apitofsim.api import _mass_spec_batches
        from apitofsim.events import MassSpecIterator, ordered_delivery

        self.max_batch_items = index(max_batch_items)
        max_pending_events = index(max_pending_events)
        if max_pending_events < 1:
            raise ValueError("max_pending_events must be positive")
        if self.max_batch_items <= 0:
            raise ValueError("max_batch_items must be positive")
        self.db, self.run_id = db, run_id
        self.cluster_id, self.pathway_id = cluster_id, pathway_id
        self.pathway_ids = tuple(pathway_ids)
        self.tree_result_id = None
        self.result = None
        self.failure = None
        self._ready = deque()
        self._done = False
        self._stream = None
        self._controller = None
        self._scope = None
        self.execution_id = None
        self._recording_transaction = None
        self.python_events = python_events
        self.record_events = (
            db.record_events if record_events is None else record_events
        )
        ordered = ordered_delivery(ordered, self.record_events)
        if hasattr(substance, "tree_nodes") and tree_cluster_ids is None:
            raise ValueError("Tree execution requires tree_cluster_ids")
        if (
            self.record_events
            and db._recording is not None
            and db._recording.active_execution is not None
        ):
            raise RuntimeError("Finish the active root before recording another root")
        try:
            if self.record_events and db._recording is None:
                self._scope = db.recording_transaction()
                self._scope.__enter__()
            if self.record_events:
                transaction = db._recording
                assert transaction is not None
                transaction.active_execution = self
                self._recording_transaction = transaction
            if tree_cluster_ids is not None:
                self.tree_result_id = db.start_tree_result(
                    run_id, cluster_id, substance, tree_cluster_ids, self.pathway_ids
                )
            self.execution_id = db.begin_execution(
                run_id,
                cluster_id,
                pathway_id,
                self.pathway_ids,
                result_id=self.tree_result_id,
                recorded=self.record_events,
            )
            level, _ = options.pop("logconf", (0, False))
            self._stream = _mass_spec_batches(
                mass_spec,
                substance,
                N,
                initial_batch_size=min(4096, self.max_batch_items),
                max_batch_size=self.max_batch_items,
                queue_capacity=max_pending_events
                if self.record_events or python_events
                else 0,
                logconf=(level, self.record_events or python_events),
                _deferred=True,
                **options,
            )
            self._controller = MassSpecIterator(
                self._stream,
                events="batches" if python_events else None,
                recording=db._recording if self.record_events else None,
                result_id=self.execution_id,
                ordered=ordered,
                progress=_progress,
                logs=_logs,
            )
        except BaseException as error:
            self._abort(error)
            raise

    def request_cancel(self):
        if self._stream is not None:
            self._stream.request_cancel()

    def __iter__(self):
        return self

    def _finish_scope(self, error=None):
        scope, self._scope = self._scope, None
        try:
            if scope is not None:
                scope.__exit__(
                    type(error) if error is not None else None,
                    error,
                    error.__traceback__ if error is not None else None,
                )
        finally:
            transaction, self._recording_transaction = self._recording_transaction, None
            if transaction is not None:
                transaction.active_execution = None

    def __next__(self):
        if self._done:
            raise StopIteration
        try:
            assert self._stream is not None
            assert self.execution_id is not None
            while not self._ready:
                batch = next(self._stream)
                if self.python_events and len(batch.events):
                    events = batch.events.rename_columns(
                        [
                            "realization_index" if name == "realization" else name
                            for name in batch.events.column_names
                        ]
                    )
                    self._ready.append(MassSpecBatch(events, []))
                self._ready.extend(batch.items)
            item = self._ready.popleft()
            if isinstance(item, MassSpecFinalResult):
                from apitofsim.events import MassSpecResult

                if self._controller is not None:
                    self._controller._completion(item)
                    result = self._controller.result
                    assert result is not None
                else:
                    result = MassSpecResult(
                        item.counters,
                        item.timings,
                        item.cancelled,
                        status="cancelled" if item.cancelled else "complete",
                    )
                if not result.cancelled:
                    self.db.record_result(
                        self.run_id,
                        item.counters,
                        item.timings,
                        cluster_id=self.cluster_id,
                        pathway_id=self.pathway_id,
                        pathway_ids=self.pathway_ids,
                        tree_result_id=self.tree_result_id,
                        result_id=self.execution_id,
                    )
                self.db.finish_execution(
                    self.execution_id, result.status, recording=result.recording
                )
                self._finish_scope()
                self.result = RecordedMassSpecResult(
                    self.run_id,
                    self.cluster_id,
                    self.pathway_ids,
                    self.execution_id,
                    result,
                )
                self._done = True
                return self.result
            return item
        except BaseException as error:
            self._abort(error)
            raise

    @staticmethod
    def _cleanup_step(error, action):
        try:
            return action()
        except BaseException as secondary:
            error.add_note(
                f"Simulation cleanup: {type(secondary).__name__}: {secondary}"
            )
            return None

    def _abort(self, error):
        if self._done:
            return
        self._done = True
        self._ready.clear()
        error.__dict__.pop("_apitofsim_recorded_failure", None)
        if self._stream is not None:
            self.request_cancel()
            try:
                while True:
                    next(self._stream)
            except StopIteration:
                pass
            except BaseException as secondary:
                if secondary is not error:
                    error.add_note(f"Draining simulation: {secondary}")
            self._cleanup_step(error, self._stream.join_if_joinable)
        info = None
        controller = self._controller
        if controller is not None:
            self._cleanup_step(error, lambda: controller._completion(error=error))
            if controller.result is not None:
                info = controller.result.recording
        failure_id = self._cleanup_step(
            error,
            lambda: self.db.record_failure(
                self.run_id,
                type(error).__name__,
                str(error),
                getattr(error, "current", None)
                if isinstance(error, ApiTofOverflowError)
                else None,
                cluster_id=self.cluster_id,
                pathway_id=self.pathway_id,
            ),
        )
        if self.execution_id is not None:
            self._cleanup_step(
                error,
                lambda: self.db.finish_execution(
                    self.execution_id, "failed", recording=info, failure_id=failure_id
                ),
            )
        if failure_id is not None:
            if self.tree_result_id is not None:
                self._cleanup_step(
                    error,
                    lambda: self.db.db.execute(
                        "update cluster_tree_experiment_result set failure_id=? where id=?",
                        [failure_id, self.tree_result_id],
                    ),
                )
            failure = RecordedFailure(
                failure_id, self.run_id, self.cluster_id, self.pathway_ids
            )
            self.failure = failure
            error.__dict__["_apitofsim_recorded_failure"] = failure
            transaction = self.db._recording
            if transaction is not None:

                def clear_failure():
                    if recorded_failure(error) is failure:
                        error.__dict__.pop("_apitofsim_recorded_failure", None)
                    self.failure = None

                transaction.failure_rollbacks.append(clear_failure)
        # A healthy writer retains partial events; I/O failure rolls back the
        # whole pending append, and remains visible to the caller.
        self._finish_scope(error)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        if not self._done:
            self.request_cancel()
            try:
                for _ in self:
                    pass
            except BaseException as cleanup:
                if exc is None:
                    raise
                exc.add_note(f"Simulation cleanup: {cleanup}")
        if self._stream is not None:
            self._stream.join_if_joinable()


class WorkflowEventIterator:
    """Option-1 event delivery around the workflow's metadata/control stream."""

    def __init__(self, stream, events):
        if events not in ("batches", "individual"):
            raise ValueError("events must be batches or individual")
        self.stream, self.events = stream, events
        self._batches = deque()
        self._batch, self._row = None, 0

    @property
    def result(self):
        return self.stream.result

    @property
    def failure(self):
        return self.stream.failure

    def request_cancel(self):
        self.stream.request_cancel()

    def __iter__(self):
        return self

    def __next__(self):
        from apitofsim.events import event_at

        if self.events == "individual" and self._batch is not None:
            value = event_at(self._batch, self._row)
            self._row += 1
            if self._row == self._batch.num_rows:
                self._batch = None
            return value
        while not self._batches:
            item = next(self.stream)
            if isinstance(item, MassSpecBatch):
                self._batches.extend(item.events.to_batches())
        batch = self._batches.popleft()
        if self.events == "individual":
            self._batch, self._row = batch, 0
            return next(self)
        return batch

    def __enter__(self):
        self.stream.__enter__()
        return self

    def __exit__(self, exc_type, exc, tb):
        return self.stream.__exit__(exc_type, exc, tb)


@dataclass(frozen=True)
class ExperimentResult:
    run_id: int
    roots: tuple[RecordedMassSpecResult, ...]
    status: str
    events_path: Any = None

    @property
    def cancelled(self):
        return self.status == "cancelled"


class ExperimentEventIterator:
    """One recording transaction over every root in prepared input order."""

    def __init__(
        self,
        runner,
        prepared,
        N,
        *,
        events,
        record_events=None,
        ordered=None,
        result_callback=None,
        log_callback=None,
        **options,
    ):
        if prepared._runner is not runner:
            raise ValueError("Prepared inputs must belong to this runner")
        self.runner, self.prepared = runner, prepared
        self.N = prepared.realizations if N is None else index(N)
        if self.N < 0:
            raise ValueError("N must be nonnegative")
        self.events = events
        self.record = (
            runner.db.record_events if record_events is None else record_events
        )
        self.result_callback, self.log_callback = result_callback, log_callback
        from apitofsim.events import ordered_delivery

        self.options = dict(
            options,
            ordered=ordered_delivery(ordered, self.record),
            _progress=result_callback is not None,
            _logs=log_callback is not None,
        )
        self.result = None
        self._groups = iter(prepared.groups)
        self._current = None
        self._scope = None
        self._started = False
        self._done = False
        self._cancel = False
        self._results = []
        self._batches = deque()
        self._batch, self._row = None, 0

    def request_cancel(self):
        self._cancel = True
        if self._current is not None:
            self._current.request_cancel()

    def _start(self):
        if not self._started:
            self._started = True
            if self.record:
                self._scope = self.runner.db.recording_transaction()
                self._scope.__enter__()
            self.runner.db.db.execute(
                "update experiment_run set execution_status='running' where id=?",
                [self.prepared.run_id],
            )

    def _complete(self, error=None):
        if self._done:
            return
        self._done = True
        status = "failed" if error else "cancelled" if self._cancel else "complete"
        self.runner.db.db.execute(
            "update experiment_run set execution_status=? where id=?",
            [status, self.prepared.run_id],
        )
        scope, self._scope = self._scope, None
        if scope is not None:
            scope.__exit__(
                type(error) if error else None,
                error,
                error.__traceback__ if error else None,
            )
        self.result = ExperimentResult(
            self.prepared.run_id,
            tuple(self._results),
            status,
            self.runner.db.events_path if self.record else None,
        )

    def __iter__(self):
        return self

    def __next__(self):
        from apitofsim.events import event_at

        if self.events == "individual" and self._batch is not None:
            value = event_at(self._batch, self._row)
            self._row += 1
            if self._row == self._batch.num_rows:
                self._batch = None
            return value
        try:
            self._start()
            while not self._batches:
                if self._done:
                    raise StopIteration
                if self._current is None:
                    group = None if self._cancel else next(self._groups, None)
                    if group is None:
                        self._complete()
                        raise StopIteration
                    self._current = self.runner._iter_mass_spec(
                        self.prepared,
                        group,
                        self.N,
                        record_events=self.record,
                        python_events=self.events is not None,
                        **self.options,
                    )
                item = next(self._current)
                if isinstance(item, MassSpecBatch):
                    self._batches.extend(item.events.to_batches())
                elif isinstance(item, RecordedMassSpecResult):
                    self._results.append(item)
                    self._current.__exit__(None, None, None)
                    self._current = None
                elif (
                    isinstance(item, MassSpecIntermediateCounter)
                    and self.result_callback
                ):
                    self.result_callback(item.counters)
                elif isinstance(item, MassSpecLogItem) and self.log_callback:
                    self.log_callback(item.type, item.name)
            batch = self._batches.popleft()
            if self.events == "individual":
                self._batch, self._row = batch, 0
                return next(self)
            return batch
        except StopIteration:
            raise
        except BaseException as error:
            if self._current is not None:
                self._current.__exit__(type(error), error, error.__traceback__)
            self._complete(error)
            raise

    def finish(self):
        for _ in self:
            pass
        return self.result

    def __enter__(self):
        self._start()
        return self

    def __exit__(self, exc_type, exc, tb):
        if not self._done:
            self.request_cancel()
            self._batch = None
            self._batches.clear()
            if self._current is not None:
                self._current.__exit__(exc_type, exc, tb)
                if self._current.result is not None:
                    self._results.append(self._current.result)
                self._current = None
            self._complete(exc)


class DerivedDataPreparer:
    def __init__(self, db: SuperClusterDatabase):
        self.db = db
        self.dos_histogram_id = None
        self.rate_histogram_id = None
        self.vib_indexes: dict[int, VibEnergyIndex] = {}
        self.rf_bases: dict[int, raw.RFBasis] = {}
        self.fragmentation_products: dict[
            tuple[int, int], raw.FragmentationProducts
        ] = {}

    def _prepare_fragmentation_products(
        self, config, clusters, pathways, densities, status_table=None
    ):
        import hashlib
        import json

        import pyarrow as pa

        self.fragmentation_products = {}
        if (
            simulation_option(config, "fragmentation_state_method")
            != raw.FragmentationStateMethod.phase_space
        ):
            return
        self.db.ensure_fragmentation_dos_cache()
        parents = {parent for parent, _, _ in pathways.values()}
        prepared = {}
        hits = misses = 0

        def cache_key(identity):
            return hashlib.sha256(json.dumps(identity).encode()).hexdigest()

        def prepare(identity, grid, build):
            nonlocal hits, misses
            key = cache_key(identity)
            if key not in prepared:
                rows = self.db.db.execute(
                    "select data from fragmentation_dos where key = ?", [key]
                ).to_arrow_table()
                if rows.num_rows:
                    prepared[key] = raw.FragmentationDos(
                        numpy.empty(0),
                        grid.bin_width,
                        grid.x_max,
                        rows["data"][0].values.to_numpy(),
                    )
                    hits += 1
                else:
                    prepared[key] = build()
                    data = prepared[key].log_counts
                    insert_via_arrow(
                        self.db.db,
                        "fragmentation_dos",
                        key=[key],
                        data=pa.FixedSizeListArray.from_arrays(data, len(data)),
                    )
                    misses += 1
            return key, prepared[key]

        for parent, first, second in pathways.values():
            if clusters[parent].charge == 0:
                continue
            if clusters[first].charge == 0:
                first, second = second, first
            if first not in parents or clusters[first].charge == 0:
                continue  # Terminal transition: no daughter state is sampled.
            if (first, second) in self.fragmentation_products:
                continue
            daughter = clusters[first].into_cpp()
            neutral = clusters[second].into_cpp()
            # Match the former tree-assembly SI grid, including selected caches.
            grid = scaled_density(self.histogram(config, densities[first])).into_cpp(
                mass_spec_stage=True
            )

            def marginal(cluster):
                return prepare(
                    (
                        2,
                        "marginal",
                        grid.bin_width.hex(),
                        grid.x_max.hex(),
                        cluster.frequencies.tolist(),
                    ),
                    grid,
                    lambda: raw.FragmentationDos(
                        cluster.frequencies, grid.bin_width, grid.x_max
                    ),
                )

            daughter_key, daughter_dos = marginal(daughter)
            neutral_dos = None
            combined_dos = daughter_dos
            if isinstance(neutral, raw.MolecularCluster):
                neutral_key, neutral_dos = marginal(neutral)
                _, combined_dos = prepare(
                    (2, "combined", daughter_key, neutral_key),
                    grid,
                    lambda: daughter_dos.extend(neutral.frequencies),
                )
            self.fragmentation_products[first, second] = raw.FragmentationProducts(
                daughter, neutral, daughter_dos, neutral_dos, combined_dos
            )
            if status_table is not None:
                status_table["Description"] = (
                    f"{hits} tables retrieved; {misses} computed"
                )

    def _fragmentation_preliminary(
        self,
        config,
        clusters,
        pathways,
        densities,
        table=None,
        pbar=None,
        step=1,
        total=1,
    ):
        from timeit import default_timer as timer

        if table is not None:
            table["#"] = f"{step}/{total}"
            table["Step"] = "Fragmentation DOS"
            table["Description"] = "Preparing product state counts"
            if pbar is not None:
                pbar.update()
        start = timer()
        self._prepare_fragmentation_products(
            config, clusters, pathways, densities, table
        )
        if table is not None:
            table["Time (s)"] = f"{timer() - start:.2f}"
            table.next_row()

    @staticmethod
    def _build_rf_basis(parameters):
        return raw.RFBasis(parameters[0], parameters[1], raw.RFBasis.default_cells)

    def _prepare_rf_bases(self, config, cluster_indexed, pathway_lookup):
        self.rf_bases = {}
        quadrupole = config.get("quadrupole")
        if (
            collision_time_options(config).method
            != raw.CollisionTimeMethod.integrated_hazard
            or quadrupole is None
        ):
            return
        import pyarrow as pa

        quadrupole = quadrupole.into_cpp()
        # The phase-scaled equation is independent of gas, DOS, absolute phase,
        # initial state, and simulation/recording options.
        cluster_keys = {}
        for cluster_id in {parent for parent, _, _ in pathway_lookup.values()}:
            parameters = raw.rf_parameters(
                quadrupole,
                raw.compute_derived(cluster_indexed[cluster_id].into_cpp()).mass,
            )
            if parameters is not None:
                cluster_keys[cluster_id] = tuple(parameters)
        keys = list(dict.fromkeys(cluster_keys.values()))
        if not keys:
            return
        cells, version = raw.RFBasis.default_cells, raw.RFBasis.format_version
        width = 16 * (cells + 1)
        cached = {}
        missing = list(
            get_through_join_else(
                self.db.db,
                self.db.db.table("rf_basis"),
                "data",
                cached,
                ac_scaled=[key[0] for key in keys],
                dc_scaled=[key[1] for key in keys],
                cells=cells,
                version=version,
            )
        )
        prepared = {}
        for key, data in cached.items():
            if len(data) != width:
                raise ValueError("Invalid persisted RF basis grid size")
            prepared[key] = raw.RFBasis(key[0], key[1], data)
        missed_keys = [(row["ac_scaled"], row["dc_scaled"]) for row in missing]
        for key in missed_keys:
            prepared[key] = self._build_rf_basis(key)
        if missed_keys:
            # Use the same columnar cache insertion as DOS, avoiding conversion
            # of each table's doubles through Python scalar SQL parameters.
            insert_via_arrow(
                self.db.db,
                "rf_basis",
                ac_scaled=[key[0] for key in missed_keys],
                dc_scaled=[key[1] for key in missed_keys],
                cells=[cells] * len(missed_keys),
                version=[version] * len(missed_keys),
                data=pa.FixedSizeListArray.from_arrays(
                    numpy.concatenate([prepared[key].data for key in missed_keys]),
                    width,
                ),
            )
        self.rf_bases = {
            cluster_id: prepared[key] for cluster_id, key in cluster_keys.items()
        }

    @staticmethod
    def _build_vib_index(mode, density, cluster_id):
        if not numpy.isfinite(density).all() or numpy.any(density < 0):
            raise ValueError(f"Invalid vibrational DOS for cluster {cluster_id}")
        if mode == VibEnergyMode.dos_only:
            with numpy.errstate(over="ignore"):
                data = numpy.cumsum(density, dtype=numpy.float64)
            if not numpy.isfinite(data[-1]):
                raise ValueError(
                    f"Vibrational DOS cumulative sum overflow for cluster {cluster_id}"
                )
            return data
        # Heap layout: root 1, children 2i and 2i+1, padded leaves have log mass -inf.
        leaves = 1 << (len(density) - 1).bit_length()
        data = numpy.full(2 * leaves, -numpy.inf)
        numpy.log(density, out=data[leaves : leaves + len(density)], where=density > 0)
        while leaves > 1:
            children = data[leaves : 2 * leaves]
            data[leaves // 2 : leaves] = numpy.logaddexp(children[::2], children[1::2])
            leaves //= 2
        return data

    def _prepare_vib_indexes(self, config, densities, pathway_lookup):
        mode = simulation_option(config, "vib_energy_sampler")
        self.vib_indexes = {}
        if mode == VibEnergyMode.direct:
            return
        conn = self.db.db
        self.db.ensure_vib_index_cache()
        parents = {parent for parent, _, _ in pathway_lookup.values()}
        for cluster_id in parents:
            rows = conn.execute(
                """
                select d.id, i.data from cluster_dos d left join vib_energy_index i
                on i.dos_id=d.id and i.mode=?
                where d.histogram_params_id=? and d.cluster_id=?
                """,
                [mode.name, self.dos_histogram_id, cluster_id],
            ).to_arrow_table()
            if rows.num_rows != 1:
                raise ValueError(f"Expected one source DOS for cluster {cluster_id}")
            stored = rows["data"][0]
            if stored.is_valid:
                data = stored.values.to_numpy()
            else:
                data = self._build_vib_index(mode, densities[cluster_id], cluster_id)
                conn.execute(
                    "insert into vib_energy_index values (?, ?, ?) on conflict do nothing",
                    [rows["id"][0].as_py(), mode.name, data],
                )
            self.vib_indexes[cluster_id] = VibEnergyIndex(mode, data)

    def histogram(self, config, data, *, rate=False):
        """Use the actual selected cache's mesh, including independent widths."""
        histogram_id = self.rate_histogram_id if rate else self.dos_histogram_id
        params = (
            self.db.get_histogram_params(histogram_id)
            if histogram_id is not None
            else (
                config["bin_width"],
                config["energy_max_rate" if rate else "energy_max"],
            )
        )
        if params is None:
            raise ValueError(f"Missing histogram configuration {histogram_id}")
        bin_width, maximum = params
        histogram = Histogram.from_mesh(bin_width, maximum, data)
        if len(histogram.x) != len(histogram.y):
            raise ValueError(
                f"Data does not match histogram configuration {histogram_id}"
            )
        return histogram

    def _run_skimmer(self, config):
        from apitofsim import skimmer

        return skimmer(config)

    def _get_cached_dos(
        self,
        cluster_indexed,
        histogram_id,
        pathway_lookup,
        include_cluster_dos=True,
        include_product_dos=True,
    ):
        num_misses = 0
        cluster_dos_dict = {}
        if include_cluster_dos:
            for cluster_id, cluster in self.db.get_cluster_dos_through_join_else(
                cluster_indexed,
                histogram_id,
                cluster_dos_dict,
            ):
                num_misses += 1

        if include_product_dos:
            for product_cluster in self.db.get_product_dos_through_join_else(
                cluster_indexed,
                histogram_id,
                cluster_dos_dict,
                pathway_lookup,
            ):
                num_misses += 1

        if num_misses > 0:
            raise ValueError(
                f"Expected to get all cluster DOS from the database, but got {num_misses} misses"
            )

        return cluster_dos_dict

    def _run_dos(
        self,
        cluster_indexed,
        histogram_id,
        bin_width,
        energy_max,
        pathway_lookup,
        status_table=None,
        use_old_impl=False,
    ):
        import pyarrow as pa

        from apitofsim.api import compute_density_of_states_batch

        num_clusters_missed = 0
        cluster_dos_dict = {}
        missed_cluster_ids = []
        density_of_states_inputs = []
        for cluster_id, cluster in self.db.get_cluster_dos_through_join_else(
            cluster_indexed,
            histogram_id,
            cluster_dos_dict,
        ):
            missed_cluster_ids.append(cluster_id)
            density_of_states_inputs.append(cluster)
            num_clusters_missed += 1

        density_of_states_inputs.extend(
            self.db.get_product_dos_through_join_else(
                cluster_indexed, histogram_id, cluster_dos_dict, pathway_lookup
            )
        )

        wanted_p1 = []
        wanted_p2 = []
        for _, product1_id, product2_id in pathway_lookup.values():
            product1_id, product2_id = sorted((product1_id, product2_id))
            if (product1_id, product2_id) not in cluster_dos_dict:
                wanted_p1.append(product1_id)
                wanted_p2.append(product2_id)

        if status_table is not None:
            status_table["Description"] = (
                f"{len(cluster_dos_dict)} retrieved; {len(density_of_states_inputs)} to compute"
            )

        if len(density_of_states_inputs) > 0:
            density_of_states = compute_density_of_states_batch(
                density_of_states_inputs,
                energy_max=energy_max,
                bin_width=bin_width,
                use_old_impl=use_old_impl,
            )
            assert density_of_states.flags.f_contiguous
            if num_clusters_missed > 0:
                arrow_arr = pa.FixedSizeListArray.from_arrays(
                    density_of_states[:, :num_clusters_missed].flatten("K"),
                    density_of_states.shape[0],
                )

                insert_via_arrow(
                    self.db.db,
                    "cluster_dos",
                    histogram_params_id=[histogram_id] * len(missed_cluster_ids),
                    cluster_id=missed_cluster_ids,
                    data=arrow_arr,
                )

            if len(density_of_states_inputs) > num_clusters_missed:
                arrow_arr = pa.FixedSizeListArray.from_arrays(
                    density_of_states[:, num_clusters_missed:].flatten("K"),
                    density_of_states.shape[0],
                )
                insert_via_arrow(
                    self.db.db,
                    "products_dos",
                    histogram_params_id=[histogram_id] * len(wanted_p1),
                    cluster1_id=wanted_p1,
                    cluster2_id=wanted_p2,
                    data=arrow_arr,
                )

            for cluster_id, v in zip(
                missed_cluster_ids, density_of_states[:, : len(missed_cluster_ids)].T
            ):
                cluster_dos_dict[cluster_id] = v

            for p1, p2, v in zip(
                wanted_p1, wanted_p2, density_of_states[:, len(missed_cluster_ids) :].T
            ):
                cluster_dos_dict[(p1, p2)] = v
        return cluster_dos_dict

    def _run_k_total(
        self,
        cluster_indexed,
        cluster_dos_dict,
        mesh,
        bin_width,
        energy_max,
        energy_max_rate,
        pathway_lookup,
        status_table=None,
        name_lookup=None,
        k_rate_cancellation=True,
    ):
        import pyarrow as pa

        from apitofsim import (
            FragmentationPathway,
            KTotalInput,
            compute_k_total_batch,
            consts,
        )
        from apitofsim.api import validate_max_energies

        histogram_id = get_or_insert(
            self.db.db,
            "histogram_params",
            bin_width=bin_width.to("K").magnitude,
            max=energy_max_rate.to("K").magnitude,
        )
        self.rate_histogram_id = histogram_id

        k_total_inputs = []
        k_total_keys = []
        k_total_dict = {}

        for (
            pathway_id,
            cluster_id,
            product1_id,
            product2_id,
        ) in self.db.get_k_rate_through_join_else(
            histogram_id, k_total_dict, pathway_lookup
        ):
            parent_cpp = cluster_indexed[cluster_id].into_cpp()
            product1_cpp = cluster_indexed[product1_id].into_cpp()
            product2_cpp = cluster_indexed[product2_id].into_cpp()
            fragmentation_energy = FragmentationPathway(
                parent_cpp, product1_cpp, product2_cpp
            ).fragmentation_energy_kelvin()
            try:
                validate_max_energies(
                    fragmentation_energy=fragmentation_energy,
                    bin_width=bin_width,
                    energy_max=energy_max,
                    energy_max_rate=energy_max_rate,
                    quantities_strict=False,
                )
            except ApiTofError as e:
                if name_lookup is not None:
                    parent_name = name_lookup[cluster_id]
                    product1_name = name_lookup[product1_id]
                    product2_name = name_lookup[product2_id]
                    e.add_note(
                        f"Pathway is {parent_name} -> {product1_name} + {product2_name}"
                    )
                    e.add_note(
                        f"{parent_name}: {parent_cpp.electronic_energy * consts.hartK}K"
                    )
                    e.add_note(
                        f"{product1_name}: {product1_cpp.electronic_energy * consts.hartK}K"
                    )
                    e.add_note(
                        f"{product2_name}: {product2_cpp.electronic_energy * consts.hartK}K"
                    )
                raise e
            product_id_tpl = tuple(sorted((product1_id, product2_id)))
            k_total_keys.append(pathway_id)
            assert cluster_dos_dict[cluster_id].flags.c_contiguous
            assert cluster_dos_dict[product_id_tpl].flags.c_contiguous
            k_total_inputs.append(
                KTotalInput(
                    product1_cpp,
                    product2_cpp,
                    fragmentation_energy,
                    cluster_dos_dict[cluster_id],
                    cluster_dos_dict[product_id_tpl],
                )
            )

        progress_callback: Callable[[int], None] | None = None
        if status_table is not None:
            status_table["Description"] = (
                f"{len(k_total_dict)} retrieved; {len(k_total_inputs)} to compute"
            )

        if not k_total_inputs:
            return k_total_dict

        if status_table is not None:
            inner_pbar = status_table(
                len(k_total_inputs),
                position=1,
                description="K total",
                show_throughput=False,
                show_progress=True,
                show_eta=True,
            )

            def _progress_callback(iters_done):
                inner_pbar.set_step(iters_done)

            progress_callback = _progress_callback

        k_rates = compute_k_total_batch(
            k_total_inputs,
            energy_max_rate=energy_max_rate,
            bin_width=bin_width,
            mesh=mesh,
            progress_callback=progress_callback,
            k_rate_cancellation=k_rate_cancellation,
        )

        insert_via_arrow(
            self.db.db,
            "k_rate",
            histogram_params_id=[histogram_id] * k_rates.shape[1],
            pathway_id=k_total_keys,
            data=pa.FixedSizeListArray.from_arrays(
                k_rates.flatten("K"), k_rates.shape[0]
            ),
        )
        for key, k_rates in zip(k_total_keys, k_rates.T):
            k_total_dict[key] = k_rates

        return k_total_dict

    def run_densityandrate(
        self,
        config,
        cluster_indexed,
        pathway_lookup,
        tablepbar=None,
        name_lookup=None,
        show_progress=True,
    ):
        from timeit import default_timer as timer

        table = pbar = None
        phase_space = (
            simulation_option(config, "fragmentation_state_method")
            == raw.FragmentationStateMethod.phase_space
        )
        total_steps, cur_step = 2 + phase_space, 1
        if show_progress and tablepbar is None:
            from progress_table import ProgressTable

            table = ProgressTable(default_column_alignment="left", refresh_rate=0)
            pbar = table(
                total_steps,
                description="Preliminary steps",
                show_throughput=False,
                show_progress=True,
                position=2,
            )
            table.add_column("#", alignment="right", width=3)
            table.add_column("Step", width=20)
            table.add_column("Description", width=40)
            table.add_column("Time (s)", alignment="right")
        elif show_progress and tablepbar is not None:
            table, pbar = tablepbar
            total_steps = 3 + phase_space
            cur_step = 2
        if table is not None and pbar is not None:
            table["#"] = f"{cur_step}/{total_steps}"
            table["Step"] = "Density of states"
            pbar.update()
        start = timer()

        histogram_id = get_or_insert(
            self.db.db,
            "histogram_params",
            bin_width=config["bin_width"].to("K").magnitude,
            max=config["energy_max"].to("K").magnitude,
        )
        self.dos_histogram_id = histogram_id

        cluster_dos_dict = self._run_dos(
            cluster_indexed,
            histogram_id,
            config["bin_width"],
            config["energy_max"],
            pathway_lookup=pathway_lookup,
            status_table=table,
            use_old_impl=config.get("use_old_dos_impl", False),
        )

        if table is not None and pbar is not None:
            table["Time (s)"] = f"{(timer() - start):.2f}"
            table.next_row()
            cur_step += 1

            table["#"] = f"{cur_step}/{total_steps}"
            table["Step"] = "K total"
            pbar.update()
        start = timer()

        k_rates = self._run_k_total(
            cluster_indexed,
            cluster_dos_dict,
            simulation_option(config, "k_total_mesh_mode"),
            config["bin_width"],
            config["energy_max"],
            config["energy_max_rate"],
            pathway_lookup=pathway_lookup,
            status_table=table,
            name_lookup=name_lookup,
            k_rate_cancellation=config.get("k_rate_cancellation", True),
        )

        if table is not None and pbar is not None:
            table["Time (s)"] = f"{(timer() - start):.2f}"
            if phase_space:
                table.next_row()
        if phase_space:
            self._fragmentation_preliminary(
                config,
                cluster_indexed,
                pathway_lookup,
                cluster_dos_dict,
                table,
                pbar,
                total_steps,
                total_steps,
            )
        else:
            self.fragmentation_products = {}
        if table is not None and pbar is not None:
            table.close()
            if pbar._is_active:
                # Not sure why this is sometimes needed
                pbar.close()

        self._prepare_vib_indexes(config, cluster_dos_dict, pathway_lookup)
        return k_rates, cluster_dos_dict

    def get_cached_densityandrate(
        self,
        cluster_indexed,
        dos_histogram_id,
        rate_histogram_id,
        pathway_lookup,
        config=None,
        tablepbar=None,
    ):
        num_missed = 0
        k_rates = {}
        for (
            pathway_id,
            cluster_id,
            product1_id,
            product2_id,
        ) in self.db.get_k_rate_through_join_else(
            rate_histogram_id, k_rates, pathway_lookup
        ):
            num_missed += 1

        if num_missed > 0:
            raise ValueError(
                f"Expected to get all k rates from the database, but got {num_missed} misses"
            )

        parents = {
            parent: cluster_indexed[parent] for parent, _, _ in pathway_lookup.values()
        }
        cluster_dos_dict = self._get_cached_dos(
            parents,
            dos_histogram_id,
            pathway_lookup,
            include_cluster_dos=True,
            include_product_dos=False,
        )
        self.dos_histogram_id = dos_histogram_id
        self.rate_histogram_id = rate_histogram_id
        self._prepare_vib_indexes(config or {}, cluster_dos_dict, pathway_lookup)
        table, pbar = tablepbar or (None, None)
        if (
            simulation_option(config or {}, "fragmentation_state_method")
            == raw.FragmentationStateMethod.phase_space
        ):
            self._fragmentation_preliminary(
                config or {},
                cluster_indexed,
                pathway_lookup,
                cluster_dos_dict,
                table,
                pbar,
                2,
                2,
            )
        else:
            self.fragmentation_products = {}
        if table is not None:
            table.close()
        if pbar is not None and pbar._is_active:
            pbar.close()
        return k_rates, cluster_dos_dict

    def run_preliminaries(
        self,
        config,
        cluster_indexed,
        pathway_lookup,
        use_cached=False,
        show_progress=True,
        cached_densityandrate: Optional[Tuple[int, int]] = None,
        name_lookup=None,
    ):
        from timeit import default_timer as timer

        prelim_table = outer_pbar = None
        phase_space = (
            simulation_option(config, "fragmentation_state_method")
            == raw.FragmentationStateMethod.phase_space
        )
        total_steps = (1 if cached_densityandrate else 3) + phase_space
        if show_progress:
            from progress_table import ProgressTable

            prelim_table = ProgressTable(
                default_column_alignment="left", refresh_rate=0
            )
            outer_pbar = prelim_table(
                total_steps,
                description="Preliminary steps",
                show_throughput=False,
                show_progress=True,
                position=2,
            )
            prelim_table.add_column("#", alignment="right", width=3)
            prelim_table.add_column("Step", width=20)
            prelim_table.add_column("Description", width=40)
            prelim_table.add_column("Time (s)", alignment="right")
            prelim_table["#"] = f"1/{total_steps}"
            prelim_table["Step"] = "Skimmer"
            outer_pbar.update()
            start = timer()

            skimmer_np = self._run_skimmer(config)

            prelim_table["Time (s)"] = f"{(timer() - start):.2f}"
            prelim_table.next_row()
        else:
            skimmer_np = self._run_skimmer(config)

        if cached_densityandrate:
            dos_histogram_id, rate_histogram_id = cached_densityandrate
            k_rates, cluster_dos = self.get_cached_densityandrate(
                cluster_indexed,
                dos_histogram_id,
                rate_histogram_id,
                pathway_lookup,
                config,
                tablepbar=(prelim_table, outer_pbar),
            )
        else:
            k_rates, cluster_dos = self.run_densityandrate(
                config,
                cluster_indexed,
                pathway_lookup,
                tablepbar=(prelim_table, outer_pbar) if show_progress else None,
                name_lookup=name_lookup,
                show_progress=show_progress,
            )

        self._prepare_rf_bases(config, cluster_indexed, pathway_lookup)
        return skimmer_np, k_rates, cluster_dos


class ExperimentRunner:
    """
    This class helps with running the simulation across configs and clusters/pathways in an ExperimentDatabase.

    It also records results and failures back into the database, and can optionally print progress tables to the terminal.

    It precomputes all histograms of density of states and rate constants as needed, and caches them in the database for future runs.
    """

    def __init__(self, db: ExperimentDatabase):
        self.db = db
        self.preparer = DerivedDataPreparer(db)
        self.current_run_id = None
        self._simulation_mode = None

    def _guard_run_started(self):
        if self.current_run_id is None:
            raise RuntimeError("No experiment run started; call start_run() first")

    def start_run(self, config_id=None, **kwargs):
        self.current_run_id = self.db.insert_run(config_id, **kwargs)
        self._simulation_mode = kwargs.get(
            "simulation_mode", SimulationMode.SINGLE_CLUSTER
        )

    def prepare_from_config(
        self,
        config,
        *,
        parent=None,
        pathways=None,
        cached_densityandrate=None,
        show_progress=False,
    ) -> PreparedSimulation:
        """Prepare SINGLE_CLUSTER groups or CLUSTER_TREE roots for a started run.

        Select a parent (ID/name) OR pathways (IDs/name triples), or omit both
        for all pathways. Empty, unknown and partially missing selections fail.
        Preparation saves effective config/cache provenance; it creates no run.
        Grouped inputs retain the existing electronic-energy bonding energies;
        the pathway-at-a-time fragmentation_energy override does not apply.
        """
        self._guard_run_started()
        if self._simulation_mode not in (
            SimulationMode.SINGLE_CLUSTER,
            SimulationMode.CLUSTER_TREE,
        ):
            raise ValueError(
                "Public preparation supports SINGLE_CLUSTER and CLUSTER_TREE"
            )
        if pathways is not None:
            pathways = list(pathways)
            if not pathways:
                raise ValueError("Pathway selection must not be empty")
        inputs = self._prepare_from_config(
            config,
            parent=parent,
            pathways=pathways,
            cached_densityandrate=cached_densityandrate,
            show_progress=show_progress,
            validate_selection=True,
        )
        mass_spec, clusters, names, lookup, rates, densities = inputs
        if self._simulation_mode == SimulationMode.CLUSTER_TREE:
            groups = []
            for (
                cluster_lookup,
                path_lookup,
                substance,
                root,
            ) in self._prepare_cluster_tree(
                config, clusters, names, lookup, rates, densities
            ):
                inverse_paths = {v: k for k, v in path_lookup.items()}
                paths = tuple(
                    inverse_paths[path.payload_idx] for path in substance.tree_pathways
                )
                products = tuple((lookup[path][1], lookup[path][2]) for path in paths)
                groups.append(
                    PreparedGroup(
                        root["cluster_id"],
                        paths,
                        products,
                        substance,
                        root["cluster_label"],
                        {v: k for k, v in cluster_lookup.items()},
                    )
                )
            groups = tuple(groups)
        else:
            groups = self._prepare_groups(
                config, clusters, names, lookup, rates, densities
            )
        assert self.current_run_id is not None
        assert self.preparer.dos_histogram_id is not None
        assert self.preparer.rate_histogram_id is not None
        return PreparedSimulation(
            self.current_run_id,
            mass_spec,
            groups,
            self.preparer.dos_histogram_id,
            self.preparer.rate_histogram_id,
            self,
            collision_time_options(config),
            config["N"],
        )

    def _iter_mass_spec(
        self,
        prepared,
        group,
        N,
        *,
        sample_mode=SampleMode.rejection,
        strict_dos=True,
        loglevel=0,
        seed=42,
        max_batch_items=262144,
        max_pending_events=262144,
        record_events=None,
        python_events=False,
        ordered=None,
        _progress=True,
        _logs=True,
    ) -> RecordingMassSpecIterator:
        """Stream a prepared group; see RecordingMassSpecIterator for cleanup.

        Counters, logs and native events are forwarded unchanged. The terminal
        RecordedMassSpecResult includes the original final result (and cancelled
        flag). Each invocation records a new result; consuming it twice does not.
        Use stream.result after early context exit to inspect saved completion.
        """
        if prepared._runner is not self or not any(group is g for g in prepared.groups):
            raise ValueError("Prepared inputs/group must belong to this runner")
        return RecordingMassSpecIterator(
            self.db,
            prepared.run_id,
            prepared.mass_spec,
            group.substance,
            N,
            tree_cluster_ids=group.tree_cluster_ids,
            cluster_id=group.cluster_id,
            pathway_ids=group.pathway_ids,
            max_batch_items=max_batch_items,
            max_pending_events=max_pending_events,
            record_events=record_events,
            python_events=python_events,
            ordered=ordered,
            _progress=_progress,
            _logs=_logs,
            sample_mode=sample_mode,
            collision_time=prepared.collision_time,
            strict=strict_dos,
            logconf=(loglevel, False),
            seed=seed,
        )

    def run_experiment(
        self, prepared, N=None, *, record_events=None, ordered=None, **options
    ):
        """Run every prepared root; return after requested persistence is durable."""
        with ExperimentEventIterator(
            self,
            prepared,
            N,
            events=None,
            record_events=record_events,
            ordered=ordered,
            **options,
        ) as stream:
            return stream.finish()

    def iter_experiment(
        self,
        prepared,
        N=None,
        *,
        events="batches",
        record_events=None,
        ordered=None,
        **options,
    ):
        """Receive events across all roots; one native IPC writer spans the run."""
        if events not in ("batches", "individual"):
            raise ValueError("events must be batches or individual")
        return ExperimentEventIterator(
            self,
            prepared,
            N,
            events=events,
            record_events=record_events,
            ordered=ordered,
            **options,
        )

    def iter_mass_spec(
        self,
        prepared,
        group,
        N,
        *,
        events="batches",
        record_events=None,
        ordered=None,
        **options,
    ):
        if events not in ("batches", "individual"):
            raise ValueError("events must be batches or individual")
        return WorkflowEventIterator(
            self._iter_mass_spec(
                prepared,
                group,
                N,
                python_events=True,
                record_events=record_events,
                ordered=ordered,
                _progress=False,
                _logs=False,
                **options,
            ),
            events,
        )

    def run_mass_spec(
        self,
        mass_spec,
        subs,
        N,
        *,
        pathway_id=None,
        cluster_id=None,
        pathway_ids=None,
        tree_cluster_ids=None,
        strict=False,
        strict_dos=True,
        loglevel=0,
        sample_mode=SampleMode.rejection,
        collision_time: CollisionTimeOptions | None = None,
        seed=42,
        max_batch_items=262144,
        result_callback=None,
        log_callback=None,
        record_events=None,
        ordered=None,
    ):
        """Run to completion using the same batched recorder as streaming runs."""
        self._guard_run_started()
        if pathway_id is None and cluster_id is None:
            raise ValueError("Either pathway_id or cluster_id must be provided")
        try:
            with RecordingMassSpecIterator(
                self.db,
                self.current_run_id,
                mass_spec,
                subs,
                N,
                cluster_id=cluster_id,
                pathway_id=pathway_id,
                pathway_ids=pathway_ids if pathway_ids is not None else (pathway_id,),
                tree_cluster_ids=tree_cluster_ids,
                max_batch_items=max_batch_items,
                record_events=record_events,
                ordered=ordered,
                _progress=result_callback is not None,
                _logs=log_callback is not None,
                sample_mode=sample_mode,
                collision_time=collision_time,
                seed=seed,
                strict=strict_dos,
                logconf=(loglevel, False),
            ) as stream:
                for item in stream:
                    if (
                        isinstance(item, MassSpecIntermediateCounter)
                        and result_callback
                    ):
                        result_callback(item.counters)
                    elif isinstance(item, MassSpecLogItem) and log_callback:
                        log_callback(item.type, item.name)
                    elif isinstance(item, RecordedMassSpecResult):
                        return item.result.counters
        except ApiTofError:
            if strict:
                raise
        return None

    def _prepare_from_config(
        self,
        config,
        parent=None,
        pathways=None,
        cached_densityandrate=None,
        show_progress=True,
        validate_selection=False,
    ):
        cluster_indexed, name_lookup, pathway_lookup = self.db.get_all_lookups(
            parent, pathways
        )

        if validate_selection:
            if not pathway_lookup:
                raise ValueError("No pathways found for the selection")
            if pathways is not None:
                if isinstance(pathways[0], tuple):
                    selected = {
                        (name_lookup[c], *sorted((name_lookup[p1], name_lookup[p2])))
                        for c, p1, p2 in pathway_lookup.values()
                    }
                    requested = {(c, *sorted((p1, p2))) for c, p1, p2 in pathways}
                else:
                    selected, requested = set(pathway_lookup), set(pathways)
                if selected != requested:
                    raise ValueError("Some selected pathways were not found")

        # The option has no effect outside tree continuation. Standalone DOS
        # preparation can still explicitly warm phase-space tables for later runs.
        preparation_config = (
            config
            if self._simulation_mode == SimulationMode.CLUSTER_TREE
            else {**config, "fragmentation_state_method": "zero"}
        )
        skimmer_np, k_rates, cluster_dos = self.preparer.run_preliminaries(
            preparation_config,
            cluster_indexed,
            pathway_lookup=pathway_lookup,
            name_lookup=name_lookup,
            cached_densityandrate=cached_densityandrate,
            show_progress=show_progress,
        )

        self.db.record_run_preparation(
            self.current_run_id,
            config,
            self.preparer.dos_histogram_id,
            self.preparer.rate_histogram_id,
        )

        assert isinstance(skimmer_np, numpy.ndarray)
        mass_spec = MassSpectrometer(
            skimmer_np,
            config["lengths"],
            config["voltages"],
            config["T"],
            config["pressures"],
            quadrupole=config.get("quadrupole"),
        )

        return (
            mass_spec,
            cluster_indexed,
            name_lookup,
            pathway_lookup,
            k_rates,
            cluster_dos,
        )

    """
    Run a `config` passed directly as a dict.
    """

    def run_from_config(self, config, *, record_events=None, **options):
        recording = self.db.record_events if record_events is None else record_events
        previous = self.db.record_events
        self.db.record_events = recording

        def execute():
            try:
                result = self._run_from_config(config, **options)
            except BaseException:
                if self.current_run_id is not None:
                    self.db.db.execute(
                        "update experiment_run set execution_status='failed' where id=?",
                        [self.current_run_id],
                    )
                raise
            statuses = {
                row[0]
                for row in self.db.db.execute(
                    "select status from event_execution where run_id=?",
                    [self.current_run_id],
                ).fetchall()
            }
            status = (
                "failed"
                if "failed" in statuses
                else "cancelled"
                if "cancelled" in statuses
                else "complete"
            )
            self.db.db.execute(
                "update experiment_run set execution_status=? where id=?",
                [status, self.current_run_id],
            )
            return result

        try:
            if recording:
                with self.db.recording_transaction():
                    return execute()
            return execute()
        finally:
            self.db.record_events = previous

    def _run_from_config(
        self,
        config,
        run_started=False,
        strict_dos=True,
        mode=None,
        parent=None,
        pathways=None,
        verbose=False,
        cached_densityandrate=None,
    ):
        mode = (
            mode if mode is not None else simulation_option(config, "simulation_mode")
        )
        if not run_started:
            config_id = self.db.insert_config(None, config)
            self.start_run(config_id, simulation_mode=mode)

        if mode == SimulationMode.SINGLE_CLUSTER:
            prepared = self.prepare_from_config(
                config,
                parent=parent,
                pathways=pathways,
                cached_densityandrate=cached_densityandrate,
                show_progress=True,
            )
            self._run_cluster_grouped(
                prepared, config, strict_dos=strict_dos, verbose=verbose
            )
            return

        (
            mass_spec,
            cluster_indexed,
            name_lookup,
            pathway_lookup,
            k_rates,
            cluster_dos,
        ) = self._prepare_from_config(
            config,
            parent=parent,
            pathways=pathways,
            cached_densityandrate=cached_densityandrate,
        )

        if mode == SimulationMode.PATHWAY_AT_A_TIME:
            self._run_pathways_at_a_time(
                mass_spec,
                config,
                cluster_indexed,
                name_lookup,
                pathway_lookup,
                k_rates,
                cluster_dos,
                strict_dos=strict_dos,
                verbose=verbose,
            )
        else:
            assert mode == SimulationMode.CLUSTER_TREE
            self._run_cluster_tree(
                mass_spec,
                config,
                cluster_indexed,
                name_lookup,
                pathway_lookup,
                k_rates,
                cluster_dos,
                strict_dos=strict_dos,
                verbose=verbose,
            )

    def _mk_update_from_counters(self, table, pbar):
        def update_from_counters(counters):
            fragmented_total = counter_fragmented_total(counters)
            realizations = counters.n_realizations
            table["Frags"] = int(fragmented_total)
            table["Intacts"] = int(counters.n_escaped_total)
            table["Avg colls"] = counters.ncoll_total / realizations
            table["PH rej"] = int(counters.counter_collision_rejections)
            table["Surv prob"] = counters.n_escaped_total / realizations
            table["Warns"] = int(counters.nwarnings)
            pbar.set_step(realizations)

        return update_from_counters

    def _run_pathways_at_a_time(
        self,
        mass_spec,
        config,
        cluster_indexed,
        name_lookup,
        pathway_lookup,
        k_rates,
        cluster_dos,
        strict_dos=True,
        parent=None,
        verbose=False,
    ):
        from os import environ
        from timeit import default_timer as timer

        from progress_table import ProgressTable

        mass_spec_table = ProgressTable(
            default_column_alignment="right",
            refresh_rate=0,
            interactive=0 if verbose else int(environ.get("PTABLE_INTERACTIVE", "2")),
        )
        mass_spec_table.add_column("#")
        mass_spec_table.add_column("Cluster", alignment="left")
        mass_spec_table.add_column("Products", width=16, alignment="left")
        mass_spec_table.add_columns(
            "Frags",
            "Intacts",
            "Avg colls",
            "PH rej",
        )
        mass_spec_table.add_column("Warns", width=5)
        mass_spec_table.add_columns(
            "Surv prob",
            "Time (s)",
        )
        cluster_seq = 1
        outer_pbar = mass_spec_table(
            pathway_lookup.items(),
            description="Pathway",
            show_throughput=False,
            show_progress=True,
            show_eta=True,
            position=2,
        )
        for pathway_id, (cluster_id, product1_id, product2_id) in outer_pbar:
            density_cluster = cluster_dos[cluster_id]
            rate_const = k_rates[pathway_id]
            cluster = cluster_indexed[cluster_id]
            product1 = cluster_indexed[product1_id]
            product2 = cluster_indexed[product2_id]
            mass_spec_table["#"] = f"{cluster_seq}/{len(pathway_lookup)}"
            mass_spec_table["Cluster"] = name_lookup[cluster_id]
            mass_spec_table["Products"] = (
                f"{name_lookup[product1_id]} + {name_lookup[product2_id]}"
            )
            start = timer()
            density_hist = scaled_density(
                self.preparer.histogram(config, density_cluster)
            )
            rate_hist = scaled_rate_const(
                self.preparer.histogram(config, rate_const, rate=True)
            )
            subs = MassSpecSubstanceSingleInput(
                cluster,
                product1,
                product2,
                config["gas"],
                density_hist,
                rate_hist,
                fragmentation_energy=config.get("fragmentation_energy"),
                cluster_charge_sign=cluster.charge,
                vib_index=self.preparer.vib_indexes.get(cluster_id),
                rf_basis=self.preparer.rf_bases.get(cluster_id),
            )
            realizations = (
                int(environ["N_OVERRIDE"]) if "N_OVERRIDE" in environ else config["N"]
            )
            inner_pbar = mass_spec_table(
                realizations, position=1, description="Realization"
            )

            def log_callback(typ, msg):
                msg = msg.rstrip()
                print(f"Channel: {typ}; Msg: {msg}")

            update_from_counters = self._mk_update_from_counters(
                mass_spec_table, inner_pbar
            )
            counters = self.run_mass_spec(
                mass_spec,
                subs,
                realizations,
                pathway_id=pathway_id,
                sample_mode=simulation_option(config, "sample_mode"),
                collision_time=collision_time_options(config),
                loglevel=1 if verbose else 0,
                strict="STRICT" in environ,
                strict_dos=strict_dos,
                result_callback=update_from_counters,
                log_callback=log_callback,
            )
            t_total = timer() - start
            if counters is None:
                for k in [
                    "Frags",
                    "Intacts",
                    "Avg colls",
                    "PH rej",
                    "Surv prob",
                    "Warns",
                ]:
                    mass_spec_table[k] = "FAIL"
            else:
                update_from_counters(counters)
            mass_spec_table["Time (s)"] = f"{t_total:.2f}"
            inner_pbar.close()
            mass_spec_table.next_row()
            cluster_seq += 1
        mass_spec_table.close()

    def _prepare_groups(
        self, config, cluster_indexed, name_lookup, pathway_lookup, k_rates, cluster_dos
    ):
        by_parent = {}
        for pathway_id, (cluster_id, p1, p2) in pathway_lookup.items():
            by_parent.setdefault(cluster_id, []).append((pathway_id, p1, p2))
        groups = []
        for cluster_id, rows in by_parent.items():
            cluster = cluster_indexed[cluster_id]
            density = scaled_density(
                self.preparer.histogram(config, cluster_dos[cluster_id])
            )
            pathways = [
                MassSpecInputFragmentationPathway(
                    cluster,
                    cluster_indexed[p1],
                    cluster_indexed[p2],
                    scaled_rate_const(
                        self.preparer.histogram(config, k_rates[pathway_id], rate=True)
                    ),
                )
                for pathway_id, p1, p2 in rows
            ]
            groups.append(
                PreparedGroup(
                    cluster_id,
                    tuple(pathway_id for pathway_id, _, _ in rows),
                    tuple((p1, p2) for _, p1, p2 in rows),
                    MassSpecSubstanceSingleInput(
                        cluster,
                        pathways,
                        config["gas"],
                        density,
                        cluster.charge,
                        vib_index=self.preparer.vib_indexes.get(cluster_id),
                        rf_basis=self.preparer.rf_bases.get(cluster_id),
                    ),
                    name_lookup[cluster_id],
                )
            )
        return tuple(groups)

    def _run_cluster_grouped(self, prepared, config, strict_dos=True, verbose=False):
        from os import environ
        from timeit import default_timer as timer

        from progress_table import ProgressTable

        mass_spec_table = ProgressTable(
            default_column_alignment="right",
            refresh_rate=0,
            interactive=0 if verbose else int(environ.get("PTABLE_INTERACTIVE", "2")),
        )
        mass_spec_table.add_column("Cluster", width=16, alignment="left")
        mass_spec_table.add_column("Paths", width=5, alignment="left")
        mass_spec_table.add_column("Frags", width=5)
        mass_spec_table.add_columns(
            "Intacts",
            "Avg colls",
        )
        mass_spec_table.add_column("PH rej", width=6)
        mass_spec_table.add_column("Warns", width=5)
        mass_spec_table.add_columns(
            "Surv prob",
            "Time (s)",
        )
        cluster_seq = 1
        outer_pbar = mass_spec_table(
            prepared.groups,
            description="Cluster",
            show_throughput=False,
            show_progress=True,
            show_eta=True,
            position=2,
        )
        for group in outer_pbar:
            mass_spec_table["Cluster"] = group.cluster_label
            mass_spec_table["Paths"] = str(len(group.pathway_ids))
            start = timer()
            realizations = (
                int(environ["N_OVERRIDE"]) if "N_OVERRIDE" in environ else config["N"]
            )
            inner_pbar = mass_spec_table(
                realizations, position=1, description="Realization"
            )

            update_from_counters = self._mk_update_from_counters(
                mass_spec_table, inner_pbar
            )
            counters = None
            try:
                with self._iter_mass_spec(
                    prepared,
                    group,
                    realizations,
                    sample_mode=simulation_option(config, "sample_mode"),
                    loglevel=1 if verbose else 0,
                    strict_dos=strict_dos,
                ) as stream:
                    for item in stream:
                        if isinstance(item, MassSpecIntermediateCounter):
                            update_from_counters(item.counters)
                        elif isinstance(item, RecordedMassSpecResult):
                            counters = item.result.counters
            except ApiTofError:
                if "STRICT" in environ:
                    raise
            t_total = timer() - start
            if counters is None:
                for k in [
                    "Frags",
                    "Intacts",
                    "Avg colls",
                    "PH rej",
                    "Surv prob",
                    "Warns",
                ]:
                    mass_spec_table[k] = "FAIL"
            else:
                update_from_counters(counters)
            mass_spec_table["Time (s)"] = f"{t_total:.2f}"
            inner_pbar.close()
            mass_spec_table.next_row()
            cluster_seq += 1
        mass_spec_table.close()

    def _prepare_cluster_tree_lookup(
        self,
        config,
        cluster_indexed,
        name_lookup,
        pathway_lookup,
        k_rates,
        cluster_dos,
    ):
        # These inputs are snapshots for this preparation only: neither database
        # metadata nor mutable ClusterData should be cached across runs.
        grids = {}

        def histogram(data, *, rate=False):
            if rate not in grids:
                source = self.preparer.histogram(config, data, rate=rate)
                result = scaled_rate_const(source) if rate else scaled_density(source)
                grids[rate] = result.x
            else:
                result = Histogram(
                    grids[rate], data if rate else data / raw.consts.boltzmann
                )
            if len(result.x) != len(result.y):
                histogram_id = (
                    self.preparer.rate_histogram_id
                    if rate
                    else self.preparer.dos_histogram_id
                )
                raise ValueError(
                    f"Data does not match histogram configuration {histogram_id}"
                )
            return result.into_cpp(mass_spec_stage=True)

        @cache
        def cluster_input(cluster_id):
            return cluster_indexed[cluster_id].into_cpp()

        @cache
        def density_input(cluster_id):
            return histogram(cluster_dos[cluster_id])

        sample_products = (
            simulation_option(config, "fragmentation_state_method")
            == raw.FragmentationStateMethod.phase_space
        )

        productive_cluster_ids = set()
        charge = None
        for _, (cluster_id, _, _) in pathway_lookup.items():
            cluster = cluster_indexed[cluster_id]
            if cluster.charge != 0:
                if charge is not None and cluster.charge != charge:
                    raise ValueError("Charge of all charged clusters must be the same")
                charge = cluster.charge
                productive_cluster_ids.add(cluster_id)

        cluster_lookup = {}
        for cluster_id in productive_cluster_ids:
            cluster_lookup[cluster_id] = {
                "pathways": [],
                "cluster": cluster_indexed[cluster_id],
                "cluster_id": cluster_id,
                "cluster_label": name_lookup[cluster_id],
                "payload": raw.MSSubstanceTreeCluster(
                    cluster_input(cluster_id),
                    density_input(cluster_id),
                    vib_index=self.preparer.vib_indexes.get(cluster_id),
                    rf_basis=self.preparer.rf_bases.get(cluster_id),
                ),
                "product_labels": [],
                "pathway_products": [],
                "pathway_ids": [],
            }

        pathway_input_lookup = {}

        for pathway_id, (
            cluster_id,
            product1_id,
            product2_id,
        ) in pathway_lookup.items():
            cluster = cluster_indexed[cluster_id]
            if cluster.charge == 0:
                continue
            rate_const = k_rates[pathway_id]
            cur_cluster_dict = cluster_lookup[cluster_id]
            product1 = cluster_indexed[product1_id]
            product2 = cluster_indexed[product2_id]
            productive_product = None
            if product1.charge != 0:
                productive_product = product1_id
            if product2.charge != 0:
                if productive_product is not None:
                    raise ValueError(
                        f"Multiple productive products found for pathway {pathway_id}"
                    )
                productive_product = product2_id
            cur_cluster_dict["pathway_products"].append(productive_product)
            cur_cluster_dict["pathway_ids"].append(pathway_id)
            products = None
            if sample_products and productive_product in cluster_lookup:
                assert productive_product is not None
                neutral_id = (
                    product2_id if productive_product == product1_id else product1_id
                )
                products = self.preparer.fragmentation_products[
                    productive_product, neutral_id
                ]
            pathway_input_lookup[pathway_id] = raw.MassSpecInputFragmentationPathway(
                cluster_input(cluster_id),
                cluster_input(product1_id),
                cluster_input(product2_id),
                histogram(rate_const, rate=True),
                products=products,
            )
        return cluster_lookup, pathway_input_lookup

    def _prepare_cluster_tree(
        self,
        config,
        cluster_indexed,
        name_lookup,
        pathway_lookup,
        k_rates,
        cluster_dos,
    ):
        cluster_lookup, pathway_input_lookup = self._prepare_cluster_tree_lookup(
            config,
            cluster_indexed,
            name_lookup,
            pathway_lookup,
            k_rates,
            cluster_dos,
        )

        for root in cluster_lookup.values():
            cluster_payload_lookup = {}
            pathway_payload_lookup = {}
            cluster_payloads = []
            pathway_payloads = []
            tree_nodes = []
            tree_pathways = []

            def walk(root, parents=()):
                cluster_id = root["cluster_id"]
                new_parents = (*parents, cluster_id)
                if cluster_id not in cluster_payload_lookup:
                    cluster_payload_lookup[cluster_id] = len(cluster_payloads)
                    cluster_payloads.append(root["payload"])
                cluster_payload_idx = cluster_payload_lookup[cluster_id]
                node_idx = len(tree_nodes)
                tree_nodes.append(None)
                pathway_indices = []
                for product_id, pathway_id in zip(
                    root["pathway_products"], root["pathway_ids"]
                ):
                    if pathway_id not in pathway_payload_lookup:
                        pathway_payload_lookup[pathway_id] = len(pathway_payloads)
                        pathway_payloads.append(pathway_input_lookup[pathway_id])
                    pathway_payload_idx = pathway_payload_lookup[pathway_id]
                    product_idx = None
                    if product_id in parents:
                        raise ValueError(
                            f"Cycle detected in cluster tree with parents {parents} and current product {product_id}"
                        )
                    if product_id in cluster_lookup:
                        product_idx = walk(cluster_lookup[product_id], new_parents)
                    pathway_indices.append(len(tree_pathways))
                    tree_pathways.append(
                        MSSubstanceTreePathway(pathway_payload_idx, product_idx)
                    )
                tree_nodes[node_idx] = MSSubstanceTreeNode(
                    cluster_payload_idx, pathway_indices
                )
                return node_idx

            walk(root)
            tree_input = MassSpecSubstanceTreeInput(
                root["cluster"].charge,
                config["gas"],
                cluster_payloads,
                pathway_payloads,
                tree_nodes,
                tree_pathways,
                fragmentation_state_method=simulation_option(
                    config, "fragmentation_state_method"
                ),
            )

            yield (cluster_payload_lookup, pathway_payload_lookup, tree_input, root)

    def _run_cluster_tree(
        self,
        mass_spec,
        config,
        cluster_indexed,
        name_lookup,
        pathway_lookup,
        k_rates,
        cluster_dos,
        strict_dos=True,
        verbose=False,
    ):
        from os import environ
        from timeit import default_timer as timer

        from progress_table import ProgressTable

        mass_spec_table = ProgressTable(
            default_column_alignment="right",
            refresh_rate=0,
            interactive=0 if verbose else int(environ.get("PTABLE_INTERACTIVE", "2")),
        )
        mass_spec_table.add_column("Cluster", width=16, alignment="left")
        mass_spec_table.add_column("Paths", width=5, alignment="left")
        mass_spec_table.add_column("Frags", width=5)
        mass_spec_table.add_columns(
            "Intacts",
            "Avg colls",
        )
        mass_spec_table.add_column("PH rej", width=6)
        mass_spec_table.add_column("Warns", width=5)
        mass_spec_table.add_columns(
            "Surv prob",
            "Time (s)",
        )
        cluster_seq = 1
        outer_pbar = mass_spec_table(
            self._prepare_cluster_tree(
                config,
                cluster_indexed,
                name_lookup,
                pathway_lookup,
                k_rates,
                cluster_dos,
            ),
            description="Cluster",
            show_throughput=False,
            show_progress=True,
            show_eta=True,
            position=2,
        )

        for cluster_payload_lookup, pathway_payload_lookup, subs, root in outer_pbar:
            mass_spec_table["Cluster"] = root["cluster_label"]
            mass_spec_table["Paths"] = str(len(pathway_payload_lookup))
            start = timer()
            realizations = (
                int(environ["N_OVERRIDE"]) if "N_OVERRIDE" in environ else config["N"]
            )

            inner_pbar = mass_spec_table(
                realizations, position=1, description="Realization"
            )

            update_from_counters = self._mk_update_from_counters(
                mass_spec_table, inner_pbar
            )
            inv_pathway_lookup = {v: k for k, v in pathway_payload_lookup.items()}
            counters = self.run_mass_spec(
                mass_spec,
                subs,
                realizations,
                cluster_id=root["cluster_id"],
                tree_cluster_ids={v: k for k, v in cluster_payload_lookup.items()},
                pathway_ids=[
                    inv_pathway_lookup[pathway.payload_idx]
                    for pathway in subs.tree_pathways
                ],
                sample_mode=simulation_option(config, "sample_mode"),
                collision_time=collision_time_options(config),
                loglevel=1 if verbose else 0,
                strict="STRICT" in environ,
                strict_dos=strict_dos,
                result_callback=update_from_counters,
            )
            t_total = timer() - start
            if counters is None:
                for k in [
                    "Frags",
                    "Intacts",
                    "Avg colls",
                    "PH rej",
                    "Surv prob",
                    "Warns",
                ]:
                    mass_spec_table[k] = "FAIL"
            else:
                update_from_counters(counters)
            mass_spec_table["Time (s)"] = f"{t_total:.2f}"
            inner_pbar.close()
            mass_spec_table.next_row()
            cluster_seq += 1
        mass_spec_table.close()

    def run_prepared_config(self, name=None, **kwargs):
        """
        Run from an experiment config that has been inserted into an ExperimentDatabase.

         * `name` is the name of an experiment config, a list thereof, or None to run all configs
        """
        from pprint import pprint

        configs = list(self.db.iter_configs(name))
        for idx, row in enumerate(configs):
            print(f"# Running experiment config: {row.name} [{idx + 1}/{len(configs)}]")
            pprint(row.config)
            print()
            self.start_run(
                row.id,
                simulation_mode=kwargs.get("mode")
                or simulation_option(row.config, "simulation_mode"),
            )
            self.run_from_config(row.config, run_started=True, **kwargs)
            print()
