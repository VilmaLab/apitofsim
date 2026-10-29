"""Public grouped preparation, persistence, and worker-lifetime contracts."""

import json
import threading
import traceback
from dataclasses import FrozenInstanceError
from datetime import timedelta
from itertools import chain
from types import SimpleNamespace

import numpy as np
import pytest
from apitofsim import api
from apitofsim.workflow import (
    ExperimentDatabase,
    ExperimentRunner,
    RealizationDatabase,
    RecordedFailure,
    RecordedMassSpecResult,
    SimulationMode,
    recorded_failure,
)
from apitofsim.workflow.db import EventRecorder, connection_scope


@pytest.fixture(params=[ExperimentDatabase, RealizationDatabase])
def prepared_case(request, monkeypatch):
    with connection_scope(request.param, ":memory:") as db:
        db.create_tables()
        # Interleave parents in insertion order, with two outgoing pathways each.
        for name, mass, charge, energy in (
            ("parent", 200, -1, -1),
            ("other", 200, -1, -1),
            ("ion", 100, -1, -0.5),
            ("neutral", 100, 0, -0.49),
            ("other-neutral", 100, 0, -0.48),
        ):
            db.insert_cluster(name, mass, charge, energy, [1, 2, 3], [10, 20, 30])
        for parent, neutral in ((1, 4), (2, 4), (1, 5), (2, 5)):
            db.insert_pathway(parent, 3, neutral)
        db.db.execute(
            "insert into histogram_params (bin_width, max) values (100, 100000), (50, 100000), (50, 200)"
        )
        for cluster in (1, 2):
            db.db.execute(
                "insert into cluster_dos (cluster_id, histogram_params_id, data) values (?, 1, ?)",
                [cluster, np.ones(1000)],
            )
        for pathway in (1, 2, 3, 4):
            db.db.execute(
                "insert into k_rate (pathway_id, histogram_params_id, data) values (?, 2, ?)",
                [pathway, np.zeros(2000)],
            )
        config = dict(
            N=5,
            gas=api.Gas(api.Q_(2.46e-10, "m"), api.Q_(4.85e-26, "kg"), 1.4),
            lengths=api.Q_(np.full(5, 0.001), "m"),
            voltages=api.Q_(np.arange(5.0) * 10, "V"),
            T=api.Q_(300, "K"),
            pressures=api.Q_([100.0, 100.0], "Pa"),
            bin_width=api.Q_(999, "K"),
            energy_max=api.Q_(999, "K"),
            energy_max_rate=api.Q_(999, "K"),
            fragmentation_energy=api.Q_(1e9, "K"),
        )
        config_id = db.insert_config("stream", config)
        runner = ExperimentRunner(db)
        runner.start_run(config_id)
        skimmer = np.array(
            [[0.0, 0.0, 300.0, 100.0, 0.0, 0.0], [0.0005, 0.0, 300.0, 100.0, 0.0, 0.0]]
        )
        monkeypatch.setattr(runner.preparer, "_run_skimmer", lambda config: skimmer)
        yield db, runner, config, config_id


def prepare(case, **kwargs):
    _, runner, config, _ = case
    return runner.prepare_from_config(config, cached_densityandrate=(1, 2), **kwargs)


def assert_counters(actual, expected):
    for a, b in zip(actual, expected, strict=True):
        np.testing.assert_array_equal(a, b)


def test_preparation_and_provenance(prepared_case, capsys, monkeypatch):
    import progress_table

    monkeypatch.setattr(
        progress_table,
        "ProgressTable",
        lambda *a, **k: pytest.fail("terminal table constructed"),
    )
    db, runner, config, _ = prepared_case
    prepared = prepare(prepared_case, pathways=[4, 1, 3])
    assert [(g.cluster_id, g.pathway_ids) for g in prepared.groups] == [
        (1, (1, 3)),
        (2, (4,)),
    ]
    assert prepared.groups[0].product_ids == ((3, 4), (3, 5))
    assert (prepared.dos_histogram_id, prepared.rate_histogram_id) == (1, 2)
    for group in prepared.groups:
        substance = group.substance
        assert substance.cluster_charge_sign == -1
        np.testing.assert_allclose(
            api.Q_(substance.density_cluster.x, "J").to("K", "boltzmann").magnitude,
            np.arange(50, 100000, 100),
        )
        for pathway in substance.pathways:
            np.testing.assert_allclose(
                api.Q_(pathway.rate_const.x, "J").to("K", "boltzmann").magnitude,
                np.arange(25, 100000, 50),
            )
            assert pathway.bonding_energy < 1e9 * api.consts.boltzmann
    metadata = json.loads(
        db.db.execute(
            "select run_config from experiment_run where id = ?", [prepared.run_id]
        ).fetchone()[0]
    )
    assert (metadata["dos_histogram_id"], metadata["rate_histogram_id"]) == (1, 2)
    assert db.db.execute("select count(*) from experiment_run").fetchone() == (1,)
    with runner.iter_mass_spec(prepared, prepared.groups[0], N=0) as stream:
        terminal = list(stream)[-1]
    curves = db.pathway_rate_curves(prepared.run_id, 1, terminal.result_id)
    assert [c.pathway_id for c in curves] == [1, 3]
    assert all(c.participating for c in curves)
    assert all(len(c.histogram.x) == 2000 for c in curves)
    assert not capsys.readouterr().out


def test_selection_validation(prepared_case):
    db, runner, config, config_id = prepared_case
    unstarted = ExperimentRunner(db)
    with pytest.raises(RuntimeError, match="start_run"):
        unstarted.prepare_from_config(config)
    for options in (
        {"pathways": []},
        {"pathways": [999]},
        {"pathways": [1, 999]},
        {"parent": 999},
        {"parent": 1, "pathways": [1]},
    ):
        with pytest.raises(ValueError):
            prepare(prepared_case, **options)
    by_name = prepare(prepared_case, pathways=[("parent", "neutral", "ion")])
    assert by_name.groups[0].pathway_ids == (1,)
    with pytest.raises(ValueError, match="not found"):
        prepare(
            prepared_case,
            pathways=[("parent", "ion", "neutral"), ("missing", "ion", "neutral")],
        )
    assert len(prepare(prepared_case, parent="parent").groups) == 1
    runner.start_run(config_id, simulation_mode=SimulationMode.CLUSTER_TREE)
    with pytest.raises(ValueError, match="SINGLE_CLUSTER"):
        prepare(prepared_case)


@pytest.mark.parametrize("cache", [(999, 2), (1, 999), (1, 3)])
def test_missing_caches(prepared_case, cache):
    _, runner, config, _ = prepared_case
    with pytest.raises(ValueError, match="miss"):
        runner.prepare_from_config(config, cached_densityandrate=cache)


def test_invalid_mesh(prepared_case):
    db, _, _, _ = prepared_case
    db.db.execute("update k_rate set data = [1, 2] where pathway_id = 1")
    with pytest.raises(ValueError, match="histogram configuration"):
        prepare(prepared_case, pathways=[1])


@pytest.mark.parametrize("sign", [-1, 1])
def test_completion_matches_blocking_runner(prepared_case, sign):
    db, runner, config, _ = prepared_case
    if sign > 0:
        db.db.execute("update cluster set charge = 1 where charge = -1")
        config["voltages"] = -config["voltages"]
    prepared = prepare(prepared_case)
    for group in prepared.groups:
        assert group.substance.cluster_charge_sign == sign
        expected = runner.run_mass_spec(
            prepared.mass_spec,
            group.substance,
            5,
            cluster_id=group.cluster_id,
            pathway_ids=group.pathway_ids,
            seed=123,
            strict=True,
        )
        with runner.iter_mass_spec(
            prepared, group, N=5, seed=123, loglevel=1
        ) as stream:
            for item in stream:
                if isinstance(item, RecordedMassSpecResult):
                    terminal = item
                    # Associations and zero-count enabled pathways exist before completion is visible.
                    rows = db.db.execute(
                        "select pathway_id, count from pathway_fragmentation where experiment_result_id = ? order by pathway_id",
                        [item.result_id],
                    ).fetchall()
                    assert rows == [(p, 0) for p in group.pathway_ids]
                    if isinstance(db, RealizationDatabase):
                        assert db.db.execute(
                            "select count(*) from realization where experiment_result_id = ?",
                            [item.result_id],
                        ).fetchone() == (5,)
        assert terminal is stream.result
        assert terminal.run_id == prepared.run_id
        assert terminal.pathway_ids == group.pathway_ids
        assert_counters(terminal.result.counters, expected)
        assert not terminal.result.cancelled
        assert terminal.result.timings.total >= timedelta(0)
        assert terminal.result.timings.loop >= timedelta(0)
        assert list(stream) == []
        stream.request_cancel()
    assert db.db.execute("select count(*) from experiment_failure").fetchone() == (0,)
    assert db.db.execute(
        "select count(*) from multi_pathway_experiment_result"
    ).fetchone() == (4,)


def test_run_binding_and_foreign_group(prepared_case):
    db, runner, _, config_id = prepared_case
    prepared = prepare(prepared_case, pathways=[1])
    runner.start_run(config_id)
    with runner.iter_mass_spec(prepared, prepared.groups[0], N=0) as stream:
        terminal = list(stream)[-1]
    assert terminal.run_id == prepared.run_id != runner.current_run_id
    assert db.db.execute(
        "select experiment_run_id from multi_pathway_experiment_result where id = ?",
        [terminal.result_id],
    ).fetchone() == (prepared.run_id,)
    with pytest.raises(ValueError, match="belong"):
        ExperimentRunner(db).iter_mass_spec(prepared, prepared.groups[0], N=0)
    with pytest.raises(ValueError, match="belong"):
        runner.iter_mass_spec(prepared, prepare(prepared_case).groups[0], N=0)


@pytest.mark.parametrize("early", [True, False])
def test_cancellation_and_early_exit(prepared_case, early, monkeypatch):
    db, runner, _, _ = prepared_case
    prepared = prepare(prepared_case, pathways=[1])
    if not early:
        original = api.mass_spec_iter

        def prefetch(*args, **kwargs):
            native = original(*args, **kwargs)
            prefix = []
            # Observe a completion before starting slow DB writes, then request
            # cancellation through the wrapper. Avoid racing an unbounded event
            # producer against per-event persistence in this lifecycle test.
            for item in native:
                prefix.append(item)
                if isinstance(item, api.MassSpecIntermediateCounter):
                    break

            class PrefetchedStream:
                def __init__(self):
                    self.items = chain(prefix, native)

                def __iter__(self):
                    return self

                def __next__(self):
                    return next(self.items)

                def __getattr__(self, name):
                    return getattr(native, name)

            return PrefetchedStream()

        monkeypatch.setattr(api, "mass_spec_iter", prefetch)
    with runner.iter_mass_spec(prepared, prepared.groups[0], N=1_000_000) as stream:
        if not early:
            stream.request_cancel()
            stream.request_cancel()
        for item in stream:
            if early:
                break
            if isinstance(item, api.MassSpecIntermediateCounter):
                assert item.counters.n_realizations > 0
                stream.request_cancel()
                stream.request_cancel()
            if isinstance(item, RecordedMassSpecResult):
                break
    final = stream.result.result
    assert stream.result.requested_realizations == 1_000_000
    assert execution_metadata(db, prepared.run_id, stream.result.result_id) == dict(
        configured_realizations=5,
        requested_realizations=1_000_000,
        completed_realizations=int(final.counters.n_realizations),
    )
    assert final.cancelled
    assert final.counters.n_realizations < 1_000_000
    if not early:
        assert final.counters.n_realizations > 0
    stream.request_cancel()
    assert list(stream) == []
    if isinstance(db, RealizationDatabase):
        assert db.db.execute(
            "select count(*) from realization where experiment_result_id is null"
        ).fetchone() == (0,)
    assert db.db.execute(
        "select count(*) from multi_pathway_experiment_result"
    ).fetchone() == (1,)


class FakeStream:
    def __init__(self, items):
        self.items = iter(items)
        self.cancelled = False
        self.joined = False

    def __iter__(self):
        return self

    def __next__(self):
        item = next(self.items)
        if isinstance(item, BaseException):
            raise item
        return item

    def request_cancel(self):
        self.cancelled = True

    def join_if_joinable(self):
        self.joined = True


def test_errors_preserve_primary_and_join(prepared_case, monkeypatch):
    db, runner, _, _ = prepared_case
    prepared = prepare(prepared_case, pathways=[1])
    primary = RuntimeError("native failure")
    fake = FakeStream([primary])
    monkeypatch.setattr(api, "mass_spec_iter", lambda *a, **k: fake)
    with pytest.raises(RuntimeError, match="native failure") as caught:
        with runner.iter_mass_spec(prepared, prepared.groups[0], N=5) as stream:
            list(stream)
    assert caught.value is primary
    assert recorded_failure(primary) is stream.failure
    assert stream.failure == RecordedFailure(1, prepared.run_id, 1, (1,))
    assert "__next__" in [
        frame.name for frame in traceback.extract_tb(primary.__traceback__)
    ]
    assert fake.cancelled and fake.joined
    assert db.db.execute("select count(*) from experiment_failure").fetchone() == (1,)
    assert db.db.execute(
        "select count(*) from multi_pathway_experiment_result"
    ).fetchone() == (0,)
    assert db.db.execute("select 42").fetchone() == (42,)


def test_missing_final(prepared_case, monkeypatch):
    db, runner, _, _ = prepared_case
    prepared = prepare(prepared_case, pathways=[1])
    fake = FakeStream([])
    monkeypatch.setattr(api, "mass_spec_iter", lambda *a, **k: fake)
    with pytest.raises(RuntimeError, match="without a final"):
        with runner.iter_mass_spec(prepared, prepared.groups[0], N=5) as stream:
            list(stream)
    assert stream.result is None
    assert stream.failure == RecordedFailure(1, prepared.run_id, 1, (1,))
    assert fake.joined
    assert db.db.execute("select count(*) from experiment_failure").fetchone() == (1,)


@pytest.mark.parametrize("where", ["event", "result", "link"])
def test_persistence_error(prepared_case, monkeypatch, where):
    db, runner, _, _ = prepared_case
    if where != "result" and not isinstance(db, RealizationDatabase):
        pytest.skip("Realization persistence only")
    prepared = prepare(prepared_case, pathways=[1])
    primary = RuntimeError(f"{where} persistence failed")
    attempts = []

    def fail(*args, **kwargs):
        attempts.append(1)
        raise primary

    if where == "event":
        # Queue events before injecting the write failure: immediate cancellation
        # can otherwise race the native producer and leave nothing to drain.
        with api.mass_spec_iter(
            prepared.mass_spec, prepared.groups[0].substance, 5, logconf=(0, True)
        ) as native:
            fake = FakeStream(list(native))
        monkeypatch.setattr(api, "mass_spec_iter", lambda *a, **k: fake)
        # Fail once to verify subsequent queued events can still be saved.
        original = EventRecorder.__call__

        def record(self, item):
            if not attempts:
                fail()
            original(self, item)

        monkeypatch.setattr(EventRecorder, "__call__", record)
    elif where == "link":
        monkeypatch.setattr(EventRecorder, "relate_realizations", fail)
    else:
        monkeypatch.setattr(db, "record_result", fail)
    with pytest.raises(RuntimeError, match=f"{where} persistence failed") as caught:
        with runner.iter_mass_spec(prepared, prepared.groups[0], N=5) as stream:
            list(stream)
    assert caught.value is primary
    assert stream.result is None
    assert recorded_failure(primary) is stream.failure
    assert stream.failure.run_id == prepared.run_id
    assert stream.failure.cluster_id == 1
    if where == "link":
        assert any("link persistence failed" in note for note in primary.__notes__)
    assert db.db.execute("select count(*) from experiment_failure").fetchone() == (1,)
    assert db.db.execute(
        "select count(*) from multi_pathway_experiment_result"
    ).fetchone() == ((1 if where == "link" else 0),)
    if where == "result":
        assert len(attempts) == 1
    if isinstance(db, RealizationDatabase):
        assert db.db.execute("select count(*) from event_info").fetchone()[0] > 0
    assert db.db.execute("select 42").fetchone() == (42,)


def test_secondary_failure_does_not_hide_primary(prepared_case, monkeypatch):
    db, runner, _, _ = prepared_case
    prepared = prepare(prepared_case, pathways=[1])
    primary = ValueError("primary")
    fake = FakeStream([primary])
    monkeypatch.setattr(api, "mass_spec_iter", lambda *a, **k: fake)

    def fail(*args, **kwargs):
        raise RuntimeError("secondary")

    monkeypatch.setattr(db, "record_failure", fail)
    monkeypatch.setattr(fake, "join_if_joinable", fail)
    with pytest.raises(ValueError, match="primary") as caught:
        with runner.iter_mass_spec(prepared, prepared.groups[0], N=5) as stream:
            list(stream)
    assert caught.value is primary
    assert recorded_failure(primary) is stream.failure is None
    assert any("secondary" in note for note in primary.__notes__)


def test_consumer_exception_drains_without_failure_row(prepared_case):
    db, runner, _, _ = prepared_case
    prepared = prepare(prepared_case, pathways=[1])
    with pytest.raises(ValueError, match="consumer") as caught:
        with runner.iter_mass_spec(prepared, prepared.groups[0], N=1_000_000) as stream:
            next(stream)
            raise ValueError("consumer")
    assert stream.result is not None
    assert recorded_failure(caught.value) is stream.failure is None
    assert db.db.execute("select count(*) from experiment_failure").fetchone() == (0,)


def test_cancel_from_thread_while_blocked(prepared_case, monkeypatch):
    _, runner, _, _ = prepared_case
    prepared = prepare(prepared_case, pathways=[1])
    entered = threading.Event()
    released = threading.Event()
    final = api.MassSpecFinalResult(SimpleNamespace(n_realizations=0), None, True)
    # Only the cancellation delegate runs on the other thread. Persistence stays
    # on the owner thread even when next() is waiting for native output.
    owner = threading.get_ident()

    class BlockedStream(FakeStream):
        def __next__(self):
            assert threading.get_ident() == owner
            entered.set()
            assert released.wait(3)
            return super().__next__()

        def request_cancel(self):
            released.set()

    fake = BlockedStream([final])
    monkeypatch.setattr(api, "mass_spec_iter", lambda *a, **k: fake)
    monkeypatch.setattr(runner.db, "record_result", lambda *a, **k: 42)
    with runner.iter_mass_spec(prepared, prepared.groups[0], N=5) as stream:

        def cancel():
            assert entered.wait(3)
            stream.request_cancel()
            stream.request_cancel()

        thread = threading.Thread(target=cancel)
        thread.start()
        try:
            assert next(stream).result.cancelled
        finally:
            thread.join(3)
        assert not thread.is_alive()
    assert fake.joined


def test_uncached_preparation_is_silent(prepared_case, monkeypatch, capsys):
    import progress_table

    db, runner, config, _ = prepared_case
    config = dict(
        config,
        bin_width=api.Q_(100, "K"),
        energy_max=api.Q_(100000, "K"),
        energy_max_rate=api.Q_(50000, "K"),
    )
    monkeypatch.setattr(
        progress_table,
        "ProgressTable",
        lambda *a, **k: pytest.fail("terminal table constructed"),
    )
    prepared = runner.prepare_from_config(config, pathways=[1])
    assert prepared.dos_histogram_id == 1
    assert prepared.rate_histogram_id not in (1, 2, 3)
    with runner.iter_mass_spec(prepared, prepared.groups[0], N=0) as stream:
        terminal = list(stream)[-1]
    curves = db.pathway_rate_curves(prepared.run_id, 1, terminal.result_id)
    assert len(curves[0].histogram.x) == 500
    assert not capsys.readouterr().out


def test_interleaved_parents(prepared_case, monkeypatch):
    db, _, _, _ = prepared_case
    original = db.get_all_lookups

    def interleaved(*args, **kwargs):
        clusters, names, pathways = original(*args, **kwargs)
        return clusters, names, dict(sorted(pathways.items()))

    monkeypatch.setattr(db, "get_all_lookups", interleaved)
    prepared = prepare(prepared_case)
    assert [(g.cluster_id, g.pathway_ids) for g in prepared.groups] == [
        (1, (1, 3)),
        (2, (2, 4)),
    ]


def test_native_fragmentation_indices(prepared_case):
    db, runner, _, _ = prepared_case
    db.db.execute("update cluster set electronic_energy = -0.5 where id = 5")
    db.db.execute(
        "update k_rate set data = ? where pathway_id = 3", [np.full(2000, 1e12)]
    )
    prepared = prepare(prepared_case, pathways=[3, 1])
    group = prepared.groups[0]
    assert group.pathway_ids == (1, 3)
    events = []
    with runner.iter_mass_spec(prepared, group, N=5) as stream:
        for item in stream:
            if isinstance(item, api.FragmentationEvent):
                events.append(item)
                assert group.pathway_ids[item.pathway_index] == 3
    counts = db.db.execute(
        "select pathway_id, count from pathway_fragmentation where experiment_result_id = ? order by pathway_id",
        [stream.result.result_id],
    ).fetchall()
    assert counts == [(1, 0), (3, 5)]
    if isinstance(db, RealizationDatabase):
        assert len(events) == 5
        assert (
            db.db.execute("select pathway_id from fragmentation_event").fetchall()
            == [(3,)] * 5
        )
        assert db.db.execute(
            "select count(*) from realization where experiment_result_id = ?",
            [stream.result.result_id],
        ).fetchone() == (5,)


def test_cancel_before_completion(prepared_case):
    db, runner, _, _ = prepared_case
    prepared = prepare(prepared_case, pathways=[1])
    prepared.mass_spec.lengths = api.Q_([100.0, 0.001, 0.001, 0.001, 0.001], "m")
    with runner.iter_mass_spec(prepared, prepared.groups[0], N=1_000_000) as stream:
        if isinstance(db, RealizationDatabase):
            assert isinstance(next(stream), api.InitEvent)
        stream.request_cancel()
        stream.request_cancel()
        terminal = list(stream)[-1]
    assert terminal.result.cancelled
    assert terminal.result.counters.n_realizations == 0
    assert db.db.execute(
        "select count(*) from multi_pathway_experiment_result"
    ).fetchone() == (1,)
    if isinstance(db, RealizationDatabase):
        assert (
            db.db.execute(
                "select count(*) from realization where experiment_result_id = ?",
                [terminal.result_id],
            ).fetchone()[0]
            > 0
        )
        assert (
            db.db.execute(
                "select count(*) from event_info where event_type = 'init'"
            ).fetchone()[0]
            > 0
        )


def test_cli_adapter_parity(prepared_case, monkeypatch):
    db, runner, config, _ = prepared_case
    monkeypatch.delenv("N_OVERRIDE", raising=False)
    prepared = prepare(prepared_case)
    expected = []
    for group in prepared.groups:
        with runner.iter_mass_spec(prepared, group, N=config["N"]) as stream:
            expected.append(list(stream)[-1].result.counters)
    runner.run_from_config(config, run_started=True, cached_densityandrate=(1, 2))
    rows = db.db.execute(
        "select cluster_id, n_escaped_total, ncoll_total, counter_collision_rejections, nwarnings, loop_us, total_us from multi_pathway_experiment_result order by id"
    ).fetchall()
    assert len(rows) == 4
    for i, counters in enumerate(expected):
        a, b = rows[i], rows[i + 2]
        assert a[:5] == b[:5]
        assert b[1:5] == (
            counters.n_escaped_total,
            counters.ncoll_total,
            counters.counter_collision_rejections,
            counters.nwarnings,
        )
        assert b[5] >= 0 and b[6] >= 0


def test_startup_error_is_recorded_once(prepared_case, monkeypatch):
    db, runner, _, _ = prepared_case
    prepared = prepare(prepared_case, pathways=[1])
    primary = ValueError("invalid native input")

    def fail(*args, **kwargs):
        raise primary

    monkeypatch.setattr(api, "mass_spec_iter", fail)
    with pytest.raises(ValueError, match="invalid native input") as caught:
        runner.iter_mass_spec(prepared, prepared.groups[0], N=5)
    assert caught.value is primary
    assert recorded_failure(primary) == RecordedFailure(1, prepared.run_id, 1, (1,))
    assert db.db.execute("select count(*) from experiment_failure").fetchone() == (1,)
    assert db.db.execute(
        "select count(*) from multi_pathway_experiment_result"
    ).fetchone() == (0,)


def test_partial_result_is_not_retried(prepared_case, monkeypatch):
    db, runner, _, _ = prepared_case
    prepared = prepare(prepared_case, pathways=[1])
    original = db.record_result
    attempts = []

    def fail(*args, **kwargs):
        attempts.append(original(*args, **kwargs))
        raise RuntimeError("write failed after insertion")

    monkeypatch.setattr(db, "record_result", fail)
    with pytest.raises(RuntimeError, match="after insertion"):
        with runner.iter_mass_spec(prepared, prepared.groups[0], N=5) as stream:
            list(stream)
    assert stream.result is None
    assert len(attempts) == 1
    assert db.db.execute(
        "select count(*) from multi_pathway_experiment_result"
    ).fetchone() == (1,)
    assert db.db.execute("select count(*) from experiment_failure").fetchone() == (1,)
    if isinstance(db, RealizationDatabase):
        failure_id = db.db.execute("select id from experiment_failure").fetchone()[0]
        assert stream.failure.failure_id == failure_id
        assert db.db.execute(
            "select distinct experiment_result_id from realization"
        ).fetchall() == [(failure_id,)]


def test_native_error_retains_partial_events(prepared_case, monkeypatch):
    db, runner, _, _ = prepared_case
    prepared = prepare(prepared_case, pathways=[1])
    primary = RuntimeError("simulation interrupted by error")
    with api.mass_spec_iter(
        prepared.mass_spec, prepared.groups[0].substance, 1, logconf=(0, True)
    ) as native:
        events = [
            item
            for item in native
            if isinstance(
                item,
                (
                    api.InitEvent,
                    api.CollisionEvent,
                    api.PostCollisionEvent,
                    api.EscapeEvent,
                ),
            )
        ]
    fake = FakeStream([*events[:2], primary])
    monkeypatch.setattr(api, "mass_spec_iter", lambda *a, **k: fake)
    with pytest.raises(RuntimeError, match="simulation interrupted"):
        with runner.iter_mass_spec(prepared, prepared.groups[0], N=5) as stream:
            list(stream)
    assert fake.joined and fake.cancelled
    assert db.db.execute("select count(*) from experiment_failure").fetchone() == (1,)
    assert db.db.execute(
        "select count(*) from multi_pathway_experiment_result"
    ).fetchone() == (0,)
    reference = recorded_failure(primary)
    assert reference is stream.failure
    db.add_failure_diagnostics(reference.failure_id, {"stage": "iteration"})
    if isinstance(db, RealizationDatabase):
        assert db.db.execute("select count(*) from event_info").fetchone() == (2,)
        failure_id = db.db.execute("select id from experiment_failure").fetchone()[0]
        assert db.db.execute(
            "select distinct experiment_result_id from realization"
        ).fetchall() == [(failure_id,)]
        assert reference.failure_id == failure_id


def test_counters_and_logs_are_forwarded_unchanged(prepared_case, monkeypatch):
    _, runner, _, _ = prepared_case
    prepared = prepare(prepared_case, pathways=[1])
    # Native terminal data is retained intact by the wrapper too.
    with api.mass_spec_iter(
        prepared.mass_spec, prepared.groups[0].substance, 0
    ) as native:
        final = list(native)[-1]
    counter = api.MassSpecIntermediateCounter(final.counters)
    log = api.MassSpecLogItem("info", "message")
    fake = FakeStream([counter, log, final])
    options = {}

    def make(*args, **kwargs):
        options.update(kwargs)
        return fake

    monkeypatch.setattr(api, "mass_spec_iter", make)
    with runner.iter_mass_spec(
        prepared, prepared.groups[0], N=0, seed=123, strict_dos=False, loglevel=2
    ) as stream:
        assert next(stream) is counter
        assert next(stream) is log
        assert next(stream).result is final
        assert list(stream) == []
    assert options["seed"] == 123
    assert not options["strict"]
    assert options["logconf"] == (2, isinstance(runner.db, RealizationDatabase))


def test_failure_references_across_attempts(prepared_case, monkeypatch):
    db, runner, _, _ = prepared_case
    prepared = prepare(prepared_case)
    references = []
    # Reuse the exception too: each insertion belongs to this attempt.
    primary = RuntimeError("repeated failure")
    primary.add_note("existing diagnostic")
    for group in (*prepared.groups, prepared.groups[0]):
        fake = FakeStream([primary])
        monkeypatch.setattr(api, "mass_spec_iter", lambda *a, **k: fake)
        with pytest.raises(RuntimeError) as caught:
            with runner.iter_mass_spec(prepared, group, N=0) as stream:
                list(stream)
        assert caught.value is primary
        reference = recorded_failure(primary)
        assert reference is stream.failure
        assert reference.cluster_id == group.cluster_id
        assert reference.pathway_ids == group.pathway_ids
        assert reference.run_id == prepared.run_id
        assert primary.__notes__[0] == "existing diagnostic"
        with pytest.raises(FrozenInstanceError):
            reference.failure_id = 99
        references.append(reference)
    assert len({ref.failure_id for ref in references}) == 3
    assert db.db.execute(
        "select id, experiment_run_id, cluster_id from experiment_failure order by id"
    ).fetchall() == [(r.failure_id, r.run_id, r.cluster_id) for r in references]

    def fail(*args, **kwargs):
        raise OSError("cannot insert failure")

    monkeypatch.setattr(db, "record_failure", fail)
    monkeypatch.setattr(api, "mass_spec_iter", lambda *a, **k: FakeStream([primary]))
    with pytest.raises(RuntimeError) as caught:
        with runner.iter_mass_spec(prepared, prepared.groups[0], N=0) as stream:
            list(stream)
    assert caught.value is primary
    assert recorded_failure(primary) is stream.failure is None
    assert any("cannot insert failure" in note for note in primary.__notes__)


def test_startup_failure_insertion_error(prepared_case, monkeypatch):
    db, runner, _, _ = prepared_case
    prepared = prepare(prepared_case)
    primary = ValueError("startup")

    def startup(*args, **kwargs):
        raise primary

    def insert(*args, **kwargs):
        raise OSError("insertion")

    monkeypatch.setattr(api, "mass_spec_iter", startup)
    monkeypatch.setattr(db, "record_failure", insert)
    with pytest.raises(ValueError) as caught:
        runner.iter_mass_spec(prepared, prepared.groups[0], N=0)
    assert caught.value is primary
    assert recorded_failure(primary) is None
    assert "insertion" in primary.__notes__[0]
    assert db.db.execute("select count(*) from experiment_failure").fetchone() == (0,)


@pytest.mark.parametrize("attached", [False, True])
@pytest.mark.parametrize("legacy", [False, True])
def test_failure_diagnostics(tmp_path, attached, legacy):
    import duckdb
    from apitofsim.workflow import sql_files_nofk

    if attached:
        connection = duckdb.connect()
        connection.execute(f"attach '{tmp_path / 'results.duckdb'}' as output")
        connection.execute("use output")
        connection.execute(sql_files_nofk.experiment)
        filename = (connection, None)
    else:
        filename = ":memory:"
    with connection_scope(ExperimentDatabase, filename) as db:
        if not attached:
            db.create_tables()
        if legacy:
            db.db.execute("alter table experiment_failure drop column diagnostics")
        config_id = db.insert_config("diagnostics", "{}")
        run_id = db.insert_run(config_id)
        failure_id = db.record_failure(run_id, "OverflowError", "original", 123)
        primary_query = "select id, experiment_run_id, pathway_id, cluster_id, exc_name, msg, overflow_requested from experiment_failure"
        original = db.db.execute(primary_query).fetchall()
        assert db.get_failure_diagnostics(failure_id) == {}
        assert db.get_failure_diagnostics(999) == {}
        with pytest.raises(KeyError, match="Unknown failure"):
            db.add_failure_diagnostics(999, {})
        for diagnostics, error in (
            ([], TypeError),
            ({"bad": object()}, TypeError),
            ({"bad": float("nan")}, ValueError),
        ):
            with pytest.raises(error):
                db.add_failure_diagnostics(failure_id, diagnostics)
        if legacy:
            assert "diagnostics" not in db.db.table("experiment_failure").columns
        db.add_failure_diagnostics(
            failure_id, {"stage": "run", "counts": {"events": 2, "results": 0}}
        )
        for _ in range(2):
            db.add_failure_diagnostics(
                failure_id,
                {
                    "stage": "failed",
                    "counts": {"events": 3},
                    "traceback": ["failed"],
                    "optional": None,
                },
            )
        db.add_failure_diagnostics(failure_id, {})
        expected = {
            "stage": "failed",
            "counts": {"events": 3},
            "traceback": ["failed"],
            "optional": None,
        }
        assert db.get_failure_diagnostics(failure_id) == expected
        assert (
            json.loads(
                db.db.execute(
                    "select diagnostics from experiment_failure where id = ?",
                    [failure_id],
                ).fetchone()[0]
            )
            == expected
        )
        assert db.db.execute(primary_query).fetchall() == original
        # Recording remains compatible with both the old and migrated schema.
        second_id = db.record_failure(run_id, "ValueError", "second")
        assert second_id != failure_id
        assert db.get_failure_diagnostics(second_id) == {}
        if not attached:
            # Existing report views still bind after a legacy schema is enriched.
            assert db.report_df("experiment_summary").iloc[0].failures == 2
        if attached:
            assert db.db.execute(
                "select count(*) from output.experiment_failure where diagnostics is not null"
            ).fetchone() == (1,)
            db.forget(runs=True)
            assert db.get_failure_diagnostics(failure_id) == {}


@pytest.mark.parametrize("startup", [True, False])
def test_native_failure_reference(prepared_case, startup):
    db, runner, _, _ = prepared_case
    prepared = prepare(prepared_case, pathways=[1])
    if startup:
        error_type = TypeError
        options = {"seed": "invalid native seed"}
    else:
        error_type = api.ApiTofParticleEscapeError
        options = {}
        # Reverse field forces an upstream escape in the native worker.
        prepared.mass_spec.voltages = api.Q_(-np.arange(5.0) * 1000, "V")
        prepared.mass_spec.pressures = api.Q_([0.001, 0.001], "Pa")
    with pytest.raises(error_type) as caught:
        with runner.iter_mass_spec(
            prepared, prepared.groups[0], N=5, **options
        ) as stream:
            list(stream)
    reference = recorded_failure(caught.value)
    assert reference == RecordedFailure(1, prepared.run_id, 1, (1,))
    assert db.db.execute(
        "select exc_name, msg from experiment_failure where id = ?",
        [reference.failure_id],
    ).fetchone() == (error_type.__name__, str(caught.value))


def execution_metadata(db, run_id, result_id):
    metadata = json.loads(
        db.db.execute(
            "select run_config from experiment_run where id = ?", [run_id]
        ).fetchone()[0]
    )
    return metadata["executions"][str(result_id)]


def test_sessions_interleave_preparation_execution_and_failure(
    prepared_case, monkeypatch
):
    db, runner, config, config_id = prepared_case
    config = dict(config, quadrupole=None)
    first = runner.start_run(config_id)
    second = runner.start_run(config_id)
    assert first.run_id != second.run_id == runner.current_run_id
    with pytest.raises(FrozenInstanceError):
        first.run_id = second.run_id
    a = first.prepare(config, pathways=[1], cached_densityandrate=(1, 2))
    b = second.prepare(config, pathways=[2], cached_densityandrate=(1, 2))
    again = first.prepare(config, pathways=[3], cached_densityandrate=(1, 2))
    for session, prepared in ((second, b), (first, a), (first, again)):
        with session.iter_mass_spec(
            prepared, prepared.groups[0], realizations=0
        ) as stream:
            terminal = list(stream)[-1]
        assert terminal.run_id == session.run_id
        assert db.db.execute(
            "select experiment_run_id from experiment_result where id = ?",
            [terminal.result_id],
        ).fetchone() == (session.run_id,)
    with pytest.raises(ValueError, match="session"):
        second.iter_mass_spec(a, a.groups[0])
    with pytest.raises(ValueError, match="preparation"):
        a.iter_mass_spec(again.groups[0])
    error = RuntimeError("old session failure")
    monkeypatch.setattr(api, "mass_spec_iter", lambda *a, **k: FakeStream([error]))
    with pytest.raises(RuntimeError, match="old session failure"):
        with a.iter_mass_spec(a.groups[0]) as stream:
            list(stream)
    assert recorded_failure(error).run_id == first.run_id
    assert db.db.execute(
        "select experiment_run_id from experiment_failure where id = ?",
        [stream.failure.failure_id],
    ).fetchone() == (first.run_id,)
    assert (
        execution_metadata(db, first.run_id, stream.failure.failure_id)[
            "requested_realizations"
        ]
        == 5
    )


def test_session_mode_is_stable(prepared_case):
    _, runner, config, config_id = prepared_case
    first = runner.start_run(config_id)
    second = runner.start_run(config_id, simulation_mode=SimulationMode.CLUSTER_TREE)
    assert first.simulation_mode == SimulationMode.SINGLE_CLUSTER
    prepared = first.prepare(config, pathways=[1], cached_densityandrate=(1, 2))
    with prepared.iter_mass_spec(prepared.groups[0], realizations=0) as stream:
        assert list(stream)[-1].run_id == first.run_id
    with pytest.raises(ValueError, match="SINGLE_CLUSTER"):
        second.prepare(config)


def test_execution_counts_and_immutable_provenance(prepared_case):
    db, runner, config, config_id = prepared_case
    run = runner.start_run(config_id)
    prepared = run.prepare(config, pathways=[1], cached_densityandrate=(1, 2))
    # Execution defaults are snapshots; callers still own their config.
    config["N"] = 99
    snapshots = []
    for options, expected in (
        ({}, 5),
        ({"realizations": 2}, 2),
        ({"realizations": 0}, 0),
    ):
        with prepared.iter_mass_spec(prepared.groups[0], **options) as stream:
            terminal = list(stream)[-1]
        assert terminal.configured_realizations == prepared.configured_realizations == 5
        assert (
            terminal.requested_realizations
            == terminal.result.counters.n_realizations
            == expected
        )
        metadata = execution_metadata(db, run.run_id, terminal.result_id)
        assert metadata == dict(
            configured_realizations=5,
            requested_realizations=expected,
            completed_realizations=expected,
        )
        snapshots.append((terminal.result_id, metadata))
        assert db.db.execute(
            "select configured_realizations, requested_realizations, completed_realizations from experiment_pathway_report where result_id = ?",
            [terminal.result_id],
        ).fetchone() == (5, expected, expected)
    assert db.db.execute(
        "select requested_realizations, completed_realizations from experiment_cluster_report"
    ).fetchone() == (7, 7)
    for result_id, metadata in snapshots:
        assert execution_metadata(db, run.run_id, result_id) == metadata
    assert config["N"] == 99
    with pytest.raises(ValueError, match="provenance.*fixed"):
        run.prepare(config, pathways=[1], cached_densityandrate=(1, 2))
    config["N"] = 5
    db.db.execute(
        "insert into k_rate (pathway_id, histogram_params_id, data) values (1, 3, ?)",
        [np.zeros(4)],
    )
    with pytest.raises(ValueError, match="provenance.*fixed"):
        run.prepare(config, pathways=[1], cached_densityandrate=(1, 3))
    for result_id, metadata in snapshots:
        assert execution_metadata(db, run.run_id, result_id) == metadata
    assert runner.current_run_id == run.run_id


@pytest.mark.parametrize(
    "count", [-1, 1.5, True, float("nan"), float("inf"), "2", [2], api.Q_(2, "m")]
)
def test_invalid_execution_counts(prepared_case, count):
    db, runner, config, _ = prepared_case
    prepared = prepare(prepared_case, pathways=[1])
    with pytest.raises(ValueError):
        prepared.iter_mass_spec(prepared.groups[0], realizations=count)
    with pytest.raises(ValueError):
        runner.iter_mass_spec(prepared, prepared.groups[0], N=count)
    with pytest.raises(ValueError):
        runner.prepare_from_config(dict(config, N=count), cached_densityandrate=(1, 2))
    assert db.db.execute("select count(*) from experiment_failure").fetchone() == (0,)
    with pytest.raises(ValueError, match="only"):
        runner.iter_mass_spec(prepared, prepared.groups[0], N=1, realizations=2)


def test_zero_configured_count(prepared_case):
    db, runner, config, _ = prepared_case
    config["N"] = 0
    prepared = prepare(prepared_case, pathways=[1])
    with prepared.iter_mass_spec(prepared.groups[0]) as stream:
        terminal = list(stream)[-1]
    assert execution_metadata(db, prepared.run_id, terminal.result_id) == dict(
        configured_realizations=0, requested_realizations=0, completed_realizations=0
    )
