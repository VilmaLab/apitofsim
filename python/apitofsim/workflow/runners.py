from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Tuple

import numpy
from pint import get_application_registry

from apitofsim.api import (
    ApiTofError,
    ApiTofOverflowError,
    Histogram,
    MassSpecFinalResult,
    MassSpecInputFragmentationPathway,
    MassSpecIntermediateCounter,
    MassSpecLogItem,
    MassSpecSubstanceSingleInput,
    MassSpecSubstanceTreeInput,
    MassSpectrometer,
    MeshMode,
    MSSubstanceTreeCluster,
    MSSubstanceTreeNode,
    MSSubstanceTreePathway,
    SampleMode,
    scaled_density,
    scaled_rate_const,
)

from .base import SimulationMode
from .db import (
    EventRecorder,
    ExperimentDatabase,
    RealizationDatabase,
    SuperClusterDatabase,
)
from .db_utils import get_or_insert, insert_via_arrow

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


@dataclass(frozen=True)
class PreparedSimulation:
    """Inputs bound to a started run and the runner that prepared them."""

    run_id: int
    mass_spec: MassSpectrometer
    groups: tuple[PreparedGroup, ...]
    dos_histogram_id: int
    rate_histogram_id: int
    _runner: object = field(repr=False, compare=False)


@dataclass(frozen=True)
class RecordedMassSpecResult:
    """Completion exposed only after result and event associations are saved."""

    run_id: int
    cluster_id: int
    pathway_ids: tuple[int, ...]
    result_id: int
    result: MassSpecFinalResult


@dataclass(frozen=True)
class RecordedFailure:
    """Immutable failure identity in the recording stream's result database."""

    failure_id: int
    run_id: int
    cluster_id: int
    pathway_ids: tuple[int, ...]


def recorded_failure(error: BaseException) -> RecordedFailure | None:
    """Return this exception's inserted failure, including startup errors.

    Return None if insertion failed or the error is consumer-owned. The same
    reference is available as stream.failure, even if event linking fails.
    Reusing an exception replaces its reference. Enrich before serialization,
    or forward the reference explicitly; serializers may drop exception state.
    """
    return error.__dict__.get("_apitofsim_recorded_failure")


class RecordingMassSpecIterator:
    """Consuming-thread persistence around the native iterator.

    Always use as a context manager. Early exit cancels, drains/records and
    joins, unlike the native iterator's join-only context exit. Only
    request_cancel() may be called from another thread; it never accesses DB.
    The caller retains ownership of the database. Execution/persistence errors
    record one failure when possible, retain partial events, then propagate.
    Consumers own diagnostics for errors in their own processing code.
    """

    def __init__(self, db, prepared, group, N, **options):
        from apitofsim.api import mass_spec_iter

        self.db = db
        self.prepared = prepared
        self.group = group
        self.result = None
        self.failure: RecordedFailure | None = None
        self._done = False
        self._stream = None
        self._recorder = (
            EventRecorder(db, group.pathway_ids)
            if isinstance(db, RealizationDatabase)
            else None
        )
        try:
            self._stream = mass_spec_iter(
                prepared.mass_spec, group.substance, N, **options
            )
        except BaseException as error:
            self._abort(error)
            raise

    def request_cancel(self):
        """Thread-safe, idempotent, non-blocking cooperative cancellation."""
        assert self._stream is not None
        self._stream.request_cancel()

    def __iter__(self):
        return self

    def _record_event(self, item):
        if self._recorder is not None and not isinstance(
            item, (MassSpecIntermediateCounter, MassSpecLogItem, MassSpecFinalResult)
        ):
            self._recorder(item)

    def __next__(self):
        if self._done:
            raise StopIteration
        assert self._stream is not None
        try:
            try:
                item = next(self._stream)
            except StopIteration:
                raise RuntimeError("Simulation ended without a final result") from None
            if isinstance(item, MassSpecFinalResult):
                result_id = self.db.record_result(
                    self.prepared.run_id,
                    item.counters,
                    item.timings,
                    cluster_id=self.group.cluster_id,
                    pathway_ids=self.group.pathway_ids,
                )
                if self._recorder is not None:
                    self._recorder.relate_realizations(result_id)
                self.result = RecordedMassSpecResult(
                    self.prepared.run_id,
                    self.group.cluster_id,
                    self.group.pathway_ids,
                    result_id,
                    item,
                )
                self._done = True
                return self.result
            self._record_event(item)
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
                f"Recording stream cleanup: {type(secondary).__name__}: {secondary}"
            )

    def _abort(self, error):
        # Do not retry a failed write or turn a later native final into success.
        self._done = True
        # A reused exception must never expose a previous attempt's reference.
        error.__dict__.pop("_apitofsim_recorded_failure", None)
        if self._stream is not None:
            self._cleanup_step(error, self.request_cancel)
            try:
                for item in self._stream:
                    self._cleanup_step(error, lambda: self._record_event(item))
            except BaseException as secondary:
                error.add_note(
                    f"Draining simulation: {type(secondary).__name__}: {secondary}"
                )
            self._cleanup_step(error, self._stream.join_if_joinable)
        failure_id = self._cleanup_step(
            error,
            lambda: self.db.record_failure(
                self.prepared.run_id,
                type(error).__name__,
                str(error),
                error.current if isinstance(error, ApiTofOverflowError) else None,  # pyright: ignore [reportAttributeAccessIssue]
                cluster_id=self.group.cluster_id,
            ),
        )
        if failure_id is not None:
            self.failure = RecordedFailure(
                failure_id,
                self.prepared.run_id,
                self.group.cluster_id,
                self.group.pathway_ids,
            )
            error.__dict__["_apitofsim_recorded_failure"] = self.failure
            if self._recorder is not None:
                recorder = self._recorder
                self._cleanup_step(
                    error, lambda: recorder.relate_realizations(failure_id)
                )

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        stream = self._stream
        assert stream is not None

        def finish():
            try:
                if not self._done:
                    self.request_cancel()
                    for _ in self:
                        pass
            except BaseException as error:
                self._cleanup_step(error, stream.join_if_joinable)
                raise
            else:
                stream.join_if_joinable()

        if exc_val is not None:
            self._cleanup_step(exc_val, finish)
        else:
            finish()


class DerivedDataPreparer:
    def __init__(self, db: SuperClusterDatabase):
        self.db = db
        self.dos_histogram_id = None
        self.rate_histogram_id = None

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

        wanted_p1 = []
        wanted_p2 = []
        for _, product1_id, product2_id in pathway_lookup.values():
            product1_id, product2_id = sorted((product1_id, product2_id))
            wanted_p1.append(product1_id)
            wanted_p2.append(product2_id)

        density_of_states_inputs.extend(
            self.db.get_product_dos_through_join_else(
                cluster_indexed, histogram_id, cluster_dos_dict, pathway_lookup
            )
        )

        if status_table is not None:
            status_table["Description"] = (
                f"{len(cluster_dos_dict)} retrieved; {len(density_of_states_inputs)} to compute"
            )

        if len(density_of_states_inputs) > 0:
            density_of_states = compute_density_of_states_batch(
                density_of_states_inputs,
                energy_max=energy_max,
                bin_width=bin_width,
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

        from apitofsim import precompute_mesh

        table = pbar = None
        total_steps, cur_step = 3, 1
        if show_progress and tablepbar is None:
            from progress_table import ProgressTable

            table = ProgressTable(default_column_alignment="left", refresh_rate=0)
            pbar = table(
                3,
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
            total_steps = 4
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
        )

        if table is not None and pbar is not None:
            table["Time (s)"] = f"{(timer() - start):.2f}"
            table.next_row()
            cur_step += 1

            table["#"] = f"{cur_step}/{total_steps}"
            table["Step"] = "Computing mesh"
            pbar.update()
        start = timer()

        if table is not None and pbar is not None:
            mesh_points = config["energy_max_rate"] / config["bin_width"]
            table["Description"] = f"Mesh of {mesh_points} pts"

        mesh = precompute_mesh(
            energy_max_rate=config["energy_max_rate"],
            bin_width=config["bin_width"],
            mesh_mode=MeshMode.compute_mesh_diagonal_multithreaded,
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
            mesh,
            config["bin_width"],
            config["energy_max"],
            config["energy_max_rate"],
            pathway_lookup=pathway_lookup,
            status_table=table,
            name_lookup=name_lookup,
        )

        if table is not None and pbar is not None:
            table["Time (s)"] = f"{(timer() - start):.2f}"
            table.close()
            if pbar._is_active:
                # Not sure why this is sometimes needed
                pbar.close()

        return k_rates, cluster_dos_dict

    def get_cached_densityandrate(
        self, cluster_indexed, dos_histogram_id, rate_histogram_id, pathway_lookup
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

        cluster_dos_dict = self._get_cached_dos(
            cluster_indexed,
            dos_histogram_id,
            pathway_lookup,
            include_cluster_dos=True,
            include_product_dos=False,
        )
        self.dos_histogram_id = dos_histogram_id
        self.rate_histogram_id = rate_histogram_id
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
        if show_progress:
            from progress_table import ProgressTable

            prelim_table = ProgressTable(
                default_column_alignment="left", refresh_rate=0
            )
            outer_pbar = prelim_table(
                4,
                description="Preliminary steps",
                show_throughput=False,
                show_progress=True,
                position=2,
            )
            prelim_table.add_column("#", alignment="right", width=3)
            prelim_table.add_column("Step", width=20)
            prelim_table.add_column("Description", width=40)
            prelim_table.add_column("Time (s)", alignment="right")
            prelim_table["#"] = "1/4"
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
                cluster_indexed, dos_histogram_id, rate_histogram_id, pathway_lookup
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
        """Prepare SINGLE_CLUSTER groups for an explicitly started run.

        Select a parent (ID/name) OR pathways (IDs/name triples), or omit both
        for all pathways. Empty, unknown and partially missing selections fail.
        Preparation saves effective config/cache provenance; it creates no run.
        Grouped inputs retain the existing electronic-energy bonding energies;
        the pathway-at-a-time fragmentation_energy override does not apply.
        """
        self._guard_run_started()
        if self._simulation_mode != SimulationMode.SINGLE_CLUSTER:
            raise ValueError("Public preparation supports SINGLE_CLUSTER only")
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
        groups = self._prepare_groups(config, clusters, names, lookup, rates, densities)
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
        )

    def iter_mass_spec(
        self,
        prepared,
        group,
        N,
        *,
        sample_mode=SampleMode.rejection,
        strict_dos=True,
        loglevel=0,
        seed=42,
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
            prepared,
            group,
            N,
            sample_mode=sample_mode,
            strict=strict_dos,
            logconf=(loglevel, isinstance(self.db, RealizationDatabase)),
            seed=seed,
        )

    def run_mass_spec(
        self,
        mass_spec,
        subs,
        *args,
        pathway_id=None,
        cluster_id=None,
        pathway_ids=None,
        product_ids=None,
        strict=False,
        strict_dos=True,
        loglevel=0,
        **kwargs,
    ):
        self._guard_run_started()
        if pathway_id is None and cluster_id is None:
            raise ValueError("Either pathway_id or cluster_id must be provided")
        from apitofsim.api import mass_spec as _mass_spec

        event_recorder = None

        if isinstance(self.db, RealizationDatabase):
            event_recorder = EventRecorder(
                self.db, pathway_ids if pathway_ids is not None else [pathway_id]
            )
            logconf = (loglevel, True)
        else:
            logconf = (loglevel, False)

        counters = None
        try:
            counters, timings = _mass_spec(
                mass_spec,
                subs,
                *args,
                **kwargs,
                logconf=logconf,
                event_callback=event_recorder,
                named_tuple_counters=True,
                output_timings=True,
                strict=strict_dos,
            )
        except ApiTofError as e:
            if strict:
                raise
            overflow_requested = None
            if isinstance(e, ApiTofOverflowError):
                overflow_requested = e.current  # pyright: ignore [reportAttributeAccessIssue]
            experiment_result_id = self.db.record_failure(
                self.current_run_id,
                type(e).__name__,
                str(e),
                overflow_requested,
                pathway_id=pathway_id,
                cluster_id=cluster_id,
            )
        else:
            experiment_result_id = self.db.record_result(
                self.current_run_id,
                counters,
                timings,
                pathway_id=pathway_id,
                pathway_ids=pathway_ids,
                cluster_id=cluster_id,
            )
        if event_recorder is not None:
            event_recorder.relate_realizations(experiment_result_id)
        return counters

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

        skimmer_np, k_rates, cluster_dos = self.preparer.run_preliminaries(
            config,
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

    def run_from_config(
        self,
        config,
        run_started=False,
        strict_dos=True,
        mode=SimulationMode.SINGLE_CLUSTER,
        parent=None,
        pathways=None,
        verbose=False,
        cached_densityandrate=None,
    ):
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
                sample_mode=SampleMode.rejection,
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
                        cluster, pathways, config["gas"], density, cluster.charge
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
                with self.iter_mass_spec(
                    prepared,
                    group,
                    realizations,
                    sample_mode=SampleMode.rejection,
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
            density_cluster = cluster_dos[cluster_id]
            density_hist = scaled_density(
                self.preparer.histogram(config, density_cluster)
            )
            cluster_lookup[cluster_id] = {
                "pathways": [],
                "cluster": cluster_indexed[cluster_id],
                "cluster_id": cluster_id,
                "cluster_label": name_lookup[cluster_id],
                "density_hist": density_hist,
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
            rate_hist = scaled_rate_const(
                self.preparer.histogram(config, rate_const, rate=True)
            )
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
            pathway_input_lookup[pathway_id] = MassSpecInputFragmentationPathway(
                cluster, product1, product2, rate_hist
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
                    cluster_payloads.append(
                        MSSubstanceTreeCluster(
                            root["cluster"],
                            root["density_hist"],
                        )
                    )
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
                pathway_ids=[
                    inv_pathway_lookup[pathway.payload_idx]
                    for pathway in subs.tree_pathways
                ],
                sample_mode=SampleMode.rejection,
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
                simulation_mode=kwargs.get("mode", SimulationMode.SINGLE_CLUSTER),
            )
            self.run_from_config(row.config, run_started=True, **kwargs)
            print()
