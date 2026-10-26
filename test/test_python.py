import os
import signal
import subprocess
import sys
import time

import pytest
from apitofsim.config import ConfigFile
from apitofsim.workflow import ExperimentDatabase, ExperimentRunner, ingest_legacy_one
from click.testing import CliRunner


@pytest.mark.parametrize("direct_import", [True, False])
@pytest.mark.parametrize("work", ["none", "mesh", "atexit"])
def test_native_interpreter_exit(direct_import, work):
    from apitofsim import apitofsimraw

    if direct_import:
        native_module = sys.modules[apitofsimraw.precompute_mesh.__module__]
        import_code = f"""
import importlib.util
spec = importlib.util.spec_from_file_location("apitofsimraw", {native_module.__file__!r})
r = importlib.util.module_from_spec(spec)
spec.loader.exec_module(r)
"""
    else:
        import_code = "from apitofsim import apitofsimraw as r"
    code = (
        import_code
        + """
import atexit
import time

def mesh():
    assert r.precompute_mesh(1000., 1., r.MeshMode.compute_mesh_diagonal_multithreaded).size == 1000
    time.sleep(.02)
    print("mesh done", flush=True)
"""
    )
    if work == "mesh":
        code += "\nmesh()\n"
    elif work == "atexit":
        code += "\natexit.register(mesh)\n"
    env = dict(os.environ, PYTHONPATH=os.pathsep.join(sys.path), MALLOC_PERTURB_="55")
    # The unload race needs repeated fresh processes, not repeated work in one pool.
    repetitions = 25 if direct_import and work != "none" else 1
    for _ in range(repetitions):
        child = subprocess.run(
            [sys.executable, "-X", "dev", "-c", code],
            env=env,
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert child.returncode == 0, child.stdout + child.stderr
        assert "could not finalize oneTBB" not in child.stderr
        if work != "none":
            assert child.stdout.strip() == "mesh done"


@pytest.mark.parametrize("signum", [signal.SIGINT, signal.SIGTERM, signal.SIGABRT])
def test_native_operation_signal_behavior(signum):
    signal_helper = os.environ.get("SIGNAL_HELPER")
    if signal_helper is None:
        pytest.skip("signal helper is only available in the Meson test suite")
    child = subprocess.Popen(
        [signal_helper],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    assert child.stdout is not None
    assert child.stdout.readline().strip() == "ready"

    started = time.monotonic()
    child.send_signal(signum)
    child.communicate(timeout=2)

    assert child.returncode == -signum
    assert time.monotonic() - started < 2


def test_legacy_atom_like_runner_functional():
    data_dir = os.environ.get("DATA_DIR")
    assert data_dir is not None, "DATA_DIR environment variable not set"
    config_filename = data_dir + "/raw/config.in"
    db = ExperimentDatabase(":memory:")
    db.create_tables()
    ingest_legacy_one(
        db,
        config_filename,
        {
            "sources": {
                "dat": {},
                "map": {
                    "1ABisopooh1brd1w-1100001000_1_129": {
                        "charge": -1,
                    },
                    "1ABisopooh1w-1010000_7_18-str7-str7": {
                        "charge": 0,
                    },
                    "1brd-1000_1_0": {
                        "charge": -1,
                    },
                },
            },
            "default_source": "dat",
            "charge": "map",
        },
    )
    config = ConfigFile(filename=config_filename)
    config = config.into_json_config()
    config["N"] = 2
    db.insert_config("test", config)
    runner = ExperimentRunner(db)
    runner.run_prepared_config()
    df = db.report_df("experiment_summary")
    if not (df["successes"].iloc[0] == 1 and df["failures"].iloc[0] == 0):
        if df["successes"].iloc[0] == 0 and df["failures"].iloc[0] == 1:
            fail_df = db.db.table("experiment_failure").fetchdf()
            exc_name = fail_df["exc_name"].iloc[0]
            msg = fail_df["msg"].iloc[0]
            pytest.fail(f"Test run failed with exception {exc_name}: {msg}")
        assert df["successes"].iloc[0] == 1 and df["failures"].iloc[0] == 0, (
            "Unexpected number of successes/failures"
        )


def test_cli_functional():
    from tempfile import TemporaryDirectory

    from apitofsim.cli import prepare, report, run
    from pandas import read_csv

    runner = CliRunner(catch_exceptions=False)
    data_dir = os.environ.get("DATA_DIR")
    assert data_dir is not None, "DATA_DIR environment variable not set"
    config_filename = data_dir + "/besel/config.toml"
    with TemporaryDirectory() as tmpdir:
        database_filename = tmpdir + "/testdb.duckdb"
        prepare_result = runner.invoke(
            prepare, ["create", config_filename, database_filename, "--ase"]
        )
        assert prepare_result.exit_code == 0
        initial_report = runner.invoke(
            report, ["pathway-report", database_filename, "pathway_report.csv"]
        )
        assert initial_report.exit_code == 0
        pathway_report = read_csv("pathway_report.csv")
        assert len(pathway_report) == 3, "Expected 3 pathways in initial report"
        run_result = runner.invoke(
            run, [database_filename, "--simulation-mode=single-cluster"]
        )
        assert run_result.exit_code == 0
        run_pathway_at_a_time_result = runner.invoke(
            run, [database_filename, "--simulation-mode=pathway-at-a-time"]
        )
        assert run_pathway_at_a_time_result.exit_code == 0
        run_cluster_tree_result = runner.invoke(
            run, [database_filename, "--simulation-mode=cluster-tree"]
        )
        assert run_cluster_tree_result.exit_code == 0
        experiment_summary_result = runner.invoke(
            report, ["experiment-summary", database_filename, "experiment_summary.csv"]
        )
        assert experiment_summary_result.exit_code == 0
        experiment_summary = read_csv("experiment_summary.csv")
        assert len(experiment_summary) == 3, (
            "Expected 3 experiments after conducting runs"
        )


def test_tree_building():
    from tempfile import TemporaryDirectory

    from apitofsim.cli import prepare

    runner = CliRunner(catch_exceptions=False)
    data_dir = os.environ.get("DATA_DIR")
    assert data_dir is not None, "DATA_DIR environment variable not set"
    config_filename = data_dir + "/besel/config.toml"
    with TemporaryDirectory() as tmpdir:
        database_filename = tmpdir + "/testdb.duckdb"
        runner.invoke(prepare, ["create", config_filename, database_filename, "--ase"])
        db = ExperimentDatabase(database_filename)
        runner = ExperimentRunner(db)
        configs = list(db.iter_configs())
        config = configs[0][2]
        (
            mass_spec,
            cluster_indexed,
            name_lookup,
            pathway_lookup,
            k_rates,
            cluster_dos,
        ) = runner._prepare_from_config(config)
        roots = runner._prepare_cluster_tree(
            config, cluster_indexed, name_lookup, pathway_lookup, k_rates, cluster_dos
        )
        for cluster_payload_lookup, pathway_payload_lookup, subs, root in roots:
            visited_cluster_payloads = set()
            visited_pathway_payloads = set()
            visited_cluster_indices = []
            visited_pathway_indices = []

            def visit(node_index):
                visited_cluster_indices.append(node_index)
                node = subs.tree_nodes[node_index]
                visited_cluster_payloads.add(node.payload_idx)
                for pathway_idx in node.pathway_indices:
                    visited_pathway_indices.append(pathway_idx)
                    pathway = subs.tree_pathways[pathway_idx]
                    visited_pathway_payloads.add(pathway.payload_idx)
                    if pathway.product_idx is not None:
                        visit(pathway.product_idx)

            visit(0)

            assert sorted(visited_cluster_indices) == list(
                range(len(subs.cluster_payloads))
            ), "Expected all tree nodes to be visited exactly once"

            assert sorted(visited_pathway_indices) == list(
                range(len(subs.pathway_payloads))
            ), "Expected all tree pathways to be visited exactly once"

            assert len(visited_cluster_payloads) == len(subs.cluster_payloads), (
                "Expected all cluster payloads to be visited"
            )

            assert len(visited_cluster_payloads) == len(cluster_payload_lookup), (
                "Expected all cluster payloads to be visited"
            )

            assert len(visited_pathway_payloads) == len(subs.pathway_payloads), (
                "Expected all pathway payloads to be visited"
            )

            assert len(visited_pathway_payloads) == len(pathway_payload_lookup), (
                "Expected all pathway payloads to be visited"
            )


def assert_event_row(connection, event, pathway_id=None):
    import apitofsim.api as api
    import numpy as np

    row = connection.execute(
        "select e.id, e.event_type, e.postime, e.velocity, e.omega, "
        "e.rot_energy, e.vibrational_energy, e.particle_index, "
        "c.id, c.theta, c.u_norm, c.accepted, f.id, f.pathway_id "
        "from event_info e "
        "left join collision_event c on c.id = e.id "
        "left join fragmentation_event f on f.id = e.id "
        "where e.id = (select max(id) from event_info)"
    ).fetchone()
    event_id, event_type, postime, velocity, omega, rot, vibrational, particle = row[:8]
    state = event.state
    assert event_type == type(event).__name__.removesuffix("Event").lower()
    np.testing.assert_array_equal(list(postime.values()), state.postime)
    np.testing.assert_array_equal(list(velocity.values()), state.velocity)
    np.testing.assert_array_equal(list(omega.values()), state.omega)
    assert (rot, vibrational, particle) == (
        state.rot_energy,
        state.vibrational_energy,
        state.particle_index,
    )
    if isinstance(event, api.CollisionEvent):
        assert row[8:12] == (event_id, event.theta, event.u_norm, event.accepted)
    else:
        assert row[8:12] == (None,) * 4
    if isinstance(event, api.FragmentationEvent):
        assert row[12] == event_id
        if pathway_id is not None:
            assert row[13] == pathway_id
    else:
        assert row[12:] == (None, None)


@pytest.mark.parametrize("mode", ["SINGLE_CLUSTER", "CLUSTER_TREE"])
def test_init_events_workflow(tmp_path, monkeypatch, mode):
    import apitofsim.api as api
    import duckdb
    import numpy as np
    from apitofsim.cli import prepare
    from apitofsim.workflow.base import SimulationMode
    from apitofsim.workflow.db import (
        EventRecorder,
        RealizationDatabase,
        connection_scope,
    )

    data_dir = os.environ["DATA_DIR"]
    filename = str(tmp_path / "realizations.duckdb")
    result = CliRunner(catch_exceptions=False).invoke(
        prepare,
        ["create", data_dir + "/besel/config.toml", filename, "--db-type=realization"],
    )
    assert result.exit_code == 0
    original_mass_spec = api.mass_spec
    num_runs = 0
    num_events = 0
    rollback_checked = False

    def check_mass_spec(ms, subs, n, **kwargs):
        nonlocal num_runs
        recorder = kwargs["event_callback"]
        initial_states = {}
        root = (
            subs
            if mode == "SINGLE_CLUSTER"
            else subs.cluster_payloads[subs.tree_nodes[0].payload_idx]
        )

        def record(event):
            nonlocal num_events, rollback_checked
            state = event.state
            if isinstance(event, api.InitEvent):
                assert state.realization not in initial_states
                initial_states[state.realization] = state
                np.testing.assert_array_equal(state.postime, np.zeros(4))
                assert np.isfinite(state.velocity).all()
                assert np.isfinite(state.omega).all()
                assert np.isfinite(state.rot_energy) and state.rot_energy >= 0
                assert (
                    np.isfinite(state.vibrational_energy)
                    and state.vibrational_energy >= 0
                )
                assert state.particle_index == 0
                assert state.rot_energy == pytest.approx(
                    0.2
                    * root.m_ion
                    * root.R_cluster**2
                    * np.dot(state.omega, state.omega),
                    rel=1e-12,
                    abs=0,
                )
            else:
                assert state.realization in initial_states
            if isinstance(event, api.FragmentationEvent) and mode == "CLUSTER_TREE":
                assert (
                    event.pathway_index
                    in subs.tree_nodes[state.particle_index].pathway_indices
                )
            if (
                isinstance(event, api.FragmentationEvent)
                and mode == "SINGLE_CLUSTER"
                and not rollback_checked
            ):
                invalid_recorder = EventRecorder(
                    recorder.db, [-1] * len(recorder.pathways)
                )
                invalid_recorder.realization_ids = recorder.realization_ids.copy()
                before = recorder.db.db.execute(
                    "select count(*) from event_info"
                ).fetchone()
                with pytest.raises(duckdb.ConstraintException):
                    invalid_recorder(event)
                assert (
                    recorder.db.db.execute("select count(*) from event_info").fetchone()
                    == before
                )
                rollback_checked = True
            recorder(event)
            pathway_id = (
                recorder.pathways[event.pathway_index]
                if isinstance(event, api.FragmentationEvent)
                else None
            )
            assert_event_row(recorder.db.db, event, pathway_id)
            num_events += 1

        counters = original_mass_spec(
            ms, subs, n, **{**kwargs, "event_callback": record}
        )
        assert set(initial_states) == set(range(n))
        events = []
        unlogged = original_mass_spec(
            ms,
            subs,
            n,
            **{**kwargs, "logconf": (0, False), "event_callback": events.append},
        )
        assert not events
        np.testing.assert_array_equal(counters[0][:-1], unlogged[0][:-1])
        np.testing.assert_array_equal(counters[0][-1], unlogged[0][-1])
        with api.mass_spec_iter(ms, subs, n, logconf=(0, True)) as stream:
            streamed_states = [
                event.state for event in stream if isinstance(event, api.InitEvent)
            ]
        assert len(streamed_states) == n
        streamed = {state.realization: state for state in streamed_states}
        assert set(streamed) == set(initial_states)
        for realization, state in streamed.items():
            np.testing.assert_array_equal(
                state.velocity, initial_states[realization].velocity
            )
            np.testing.assert_array_equal(
                state.omega, initial_states[realization].omega
            )
            assert state.rot_energy == initial_states[realization].rot_energy
            assert (
                state.vibrational_energy
                == initial_states[realization].vibrational_energy
            )
        if mode == "SINGLE_CLUSTER" and num_runs == 0:
            raw = api.apitofsimraw
            rate = subs.pathways[0].rate_const
            nonfragmenting = raw.MassSpecSubstanceSingleInput(
                subs.cluster_charge_sign,
                subs.m_ion,
                subs.R_cluster,
                subs.density_cluster,
                [
                    raw.MassSpecInputFragmentationPathway(
                        raw.Histogram(rate.x, np.zeros_like(rate.y)), 1e9
                    )
                ],
                subs.gas,
            )
            with connection_scope(RealizationDatabase, ":memory:") as escape_db:
                escape_db.create_tables()
                escape_recorder = EventRecorder(escape_db, [])
                escapes = 0
                with api.mass_spec_iter(
                    ms, nonfragmenting, n, logconf=(0, True), strict=False
                ) as stream:
                    for event in stream:
                        if isinstance(
                            event, (api.InitEvent, api.CollisionEvent, api.EscapeEvent)
                        ):
                            escape_recorder(event)
                            assert_event_row(escape_db.db, event)
                            escapes += isinstance(event, api.EscapeEvent)
                assert escapes == n
        num_runs += 1
        return counters

    monkeypatch.setattr(api, "mass_spec", check_mass_spec)
    with connection_scope(RealizationDatabase, filename) as db:
        runner = ExperimentRunner(db)
        runner.run_prepared_config(mode=SimulationMode[mode])
        assert num_runs > 0
        for result_id, run_id, parent, _, _ in db.db.table(
            "experiment_result"
        ).fetchall():
            curves = db.pathway_rate_curves(run_id, parent, result_id)
            assert curves and all(c.histogram is not None for c in curves)
        if mode == "SINGLE_CLUSTER":
            assert rollback_checked
        count = db.db.execute(
            "select count(*) from event_info where event_type = 'init'"
        ).fetchone()[0]
        assert count == db.db.execute("select count(*) from realization").fetchone()[0]
        assert (
            db.db.execute(
                "select count(*) from realization where experiment_result_id is null"
            ).fetchone()[0]
            == 0
        )
        assert (
            db.db.execute("select count(*) from event_info").fetchone()[0] == num_events
        )
        db.refresh_views()
        assert (
            db.db.execute("select count(*) from event_report").fetchone()[0]
            == num_events
        )
        assert (
            db.db.execute(
                "select count(*) from event_info e join event_report r on r.event_id = e.id "
                "left join collision_event c on c.id = e.id "
                "left join fragmentation_event f on f.id = e.id "
                "where e.event_type = r.event_type and e.postime.x = r.x "
                "and e.postime.y = r.y and e.postime.z = r.z and e.postime.t = r.t "
                "and e.velocity = r.velocity and e.omega = r.omega "
                "and e.rot_energy = r.rot_energy and e.vibrational_energy = r.vibrational_energy "
                "and e.particle_index = r.particle_index "
                "and c.theta is not distinct from r.theta "
                "and c.u_norm is not distinct from r.u_norm "
                "and c.accepted is not distinct from r.accepted "
                "and f.pathway_id is not distinct from r.pathway_id"
            ).fetchone()[0]
            == num_events
        )
        assert (
            db.db.execute(
                "select count(*) from event_report where event_type = 'init'"
            ).fetchone()[0]
            == count
        )


@pytest.mark.parametrize("accepted", [True, False])
def test_collision_details_and_escape(accepted, monkeypatch):
    from types import SimpleNamespace

    import apitofsim.api as api
    import numpy as np
    from apitofsim.workflow.db import (
        EventRecorder,
        RealizationDatabase,
        connection_scope,
    )

    # Native events are read-only; use payload fixtures to exercise both acceptance values.
    CollisionEvent = type("CollisionEvent", (SimpleNamespace,), {})
    EscapeEvent = type("EscapeEvent", (SimpleNamespace,), {})
    monkeypatch.setattr(api, "CollisionEvent", CollisionEvent)
    monkeypatch.setattr(api, "EscapeEvent", EscapeEvent)
    state = SimpleNamespace(
        realization=7,
        postime=np.array([1.123456789, 2.234567891, 3.345678912, 4.456789123]),
        velocity=np.array([5.567891234, -6.678912345, 7.789123456]),
        omega=np.array([8.891234567, 9.912345678, -10.123456789]),
        rot_energy=1.234567891234567e-21,
        vibrational_energy=2.345678912345678e-21,
        particle_index=3,
    )
    with connection_scope(RealizationDatabase, ":memory:") as db:
        db.create_tables()
        recorder = EventRecorder(db, [])
        collision = CollisionEvent(
            state=state, theta=0.123456789, u_norm=123.456789, accepted=accepted
        )
        recorder(collision)
        assert_event_row(db.db, collision)
        escape = EscapeEvent(state=state)
        recorder(escape)
        assert_event_row(db.db, escape)
        assert db.db.execute("select count(*) from event_info").fetchone()[0] == 2
        assert db.db.execute("select count(*) from collision_event").fetchone()[0] == 1
        assert (
            db.db.execute("select count(*) from fragmentation_event").fetchone()[0] == 0
        )


def test_sql_foreign_keys_declared_on_own_line():
    from importlib import resources

    import apitofsim.workflow
    import apitofsim.workflow.sql_files as sql_files
    import apitofsim.workflow.sql_files_nofk as sql_files_nofk

    sql_dir = resources.files(apitofsim.workflow) / "sql"
    for path in sorted(sql_dir.iterdir(), key=lambda p: p.name):
        if path.suffix != ".sql":
            continue
        name = path.name.removesuffix(".sql")
        raw = getattr(sql_files, name)
        stripped = getattr(sql_files_nofk, name)
        for line in raw.split("\n"):
            assert "references" not in line.lower() or "foreign key" in line.lower(), (
                f"{path.name}: foreign key must be declared on its own line, "
                f"not inline: {line.strip()!r}"
            )
        assert "references" not in stripped.lower(), (
            f"{path.name}: stripping foreign keys left references behind"
        )


@pytest.mark.parametrize(
    "cluster_dos, products_dos, rates",
    [
        ([], [], []),
        ([1], [], [1]),
        ([], [2, 2], []),
        ([1, 1, 2], [2, 3], [2, 2, 4]),
    ],
)
def test_available_histogram_queries(cluster_dos, products_dos, rates):
    import duckdb
    from apitofsim.workflow.db import SuperClusterDatabase, connection_scope

    with connection_scope(SuperClusterDatabase, ":memory:") as database:
        database.create_tables()
        db = database.db
        db.execute("insert into cluster (id) values (1)")
        db.execute("insert into pathway values (1, 1, 1, 1)")
        # Equal widths and differing maxima remain distinct configurations;
        # unreferenced parameter and mesh rows do not imply availability.
        db.execute(
            "insert into histogram_params values "
            "(1, 10, 100), (2, 1, 100), (3, 1, 200), (4, 5, 100), (5, 20, 100)"
        )
        db.execute("insert into k_rate_mesh (histogram_params_id) values (5)")
        for histogram_id in cluster_dos:
            db.execute(
                "insert into cluster_dos (histogram_params_id, cluster_id) values (?, 1)",
                (histogram_id,),
            )
        for histogram_id in products_dos:
            db.execute(
                "insert into products_dos (histogram_params_id, cluster1_id, cluster2_id) values (?, 1, 1)",
                (histogram_id,),
            )
        for histogram_id in rates:
            db.execute(
                "insert into k_rate (histogram_params_id, pathway_id) values (?, 1)",
                (histogram_id,),
            )
        widths = {1: 10.0, 2: 1.0, 3: 1.0, 4: 5.0}
        for query, expected_ids in (
            (
                database.available_dos_histograms_query(),
                set(cluster_dos + products_dos),
            ),
            (database.available_rate_histograms_query(), set(rates)),
        ):
            assert isinstance(query, duckdb.DuckDBPyRelation)
            assert query.columns == ["histogram_params_id", "bin_width"]
            assert set(query.fetchall()) == {(id, widths[id]) for id in expected_ids}


def test_experiment_metadata_queries():
    import json
    from datetime import datetime

    import duckdb
    from apitofsim.workflow.db import connection_scope

    with connection_scope(ExperimentDatabase, ":memory:") as database:
        database.create_tables()
        assert database.experiment_overview_query().fetchall() == []
        assert database.experiment_clusters_query(1).fetchall() == []
        db = database.db
        db.execute(
            "insert into cluster (id, common_name, atomic_mass) "
            "values (1, 'parent', 100), (2, 'other', 50)"
        )
        db.execute("insert into pathway values (1, 1, 2, 2), (2, 1, 1, 2)")
        db.execute(
            "insert into experiment_config values (1, 'raw config', '{\"N\": 42}')"
        )
        start_time = datetime(2026, 1, 2, 3, 4, 5)
        for run_id, mode in (
            (1, "SINGLE_PATHWAY"),
            (2, "SINGLE_CLUSTER"),
            (3, "SINGLE_CLUSTER"),
        ):
            db.execute(
                "insert into experiment_run values (?, 1, ?, ?)",
                (run_id, json.dumps({"simulation_mode": mode}), start_time),
            )
        db.execute(
            "insert into single_pathway_experiment_result (experiment_run_id, pathway_id) "
            "values (1, 1), (1, 2)"
        )
        result_id = db.execute(
            "insert into multi_pathway_experiment_result (experiment_run_id, cluster_id) "
            "values (2, 1) returning id"
        ).fetchone()[0]
        db.execute(
            "insert into pathway_fragmentation (experiment_result_id, pathway_id, count) "
            "values (?, 1, 1), (?, 2, 2)",
            (result_id, result_id),
        )
        database.record_failure(1, "ValueError", "test failure", pathway_id=1)

        assert database.experiment_overview_query(999).fetchall() == []
        assert database.experiment_clusters_query(999).fetchall() == []
        assert database.experiment_clusters_query(1, 999).fetchall() == []
        assert database.experiment_clusters_query(3).fetchall() == []
        assert len(database.experiment_overview_query().fetchall()) == 3
        for run_id, successes, failures, is_single in (
            (1, 2, 1, True),
            (2, 1, 0, False),
            (3, 0, 0, False),
        ):
            overview = database.experiment_overview_query(run_id)
            assert isinstance(overview, duckdb.DuckDBPyRelation)
            row = dict(zip(overview.columns, overview.fetchone(), strict=True))
            assert row == {
                "experiment_run_id": run_id,
                "config_name": "raw config",
                "start_time": start_time,
                "successes": successes,
                "failures": failures,
                "is_single_pathway": is_single,
                "config": '{"N": 42}',
                "run_config": json.dumps(
                    {
                        "simulation_mode": "SINGLE_PATHWAY"
                        if is_single
                        else "SINGLE_CLUSTER"
                    }
                ),
            }
        for run_id, is_single in ((1, True), (2, False)):
            clusters = database.experiment_clusters_query(run_id)
            assert isinstance(clusters, duckdb.DuckDBPyRelation)
            assert clusters.select(
                "cluster_id, cluster_common_name, cluster_atomic_mass, is_single_pathway"
            ).fetchall() == [(1, "parent", 100.0, is_single)]
            assert (
                database.experiment_clusters_query(run_id, 1).fetchall()
                == clusters.fetchall()
            )


@pytest.mark.parametrize("attached", [False, True])
def test_normalized_result_and_event_views(tmp_path, attached):
    import duckdb
    from apitofsim.workflow import sql_files, sql_files_nofk
    from apitofsim.workflow.db import RealizationDatabase, connection_scope

    if attached:
        clusters_path = str(tmp_path / "clusters.duckdb")
        results_path = str(tmp_path / "results.duckdb")
        with duckdb.connect(clusters_path) as clusters:
            clusters.execute(sql_files.pathway)
            clusters.execute(
                "insert into cluster (id, common_name) values (1, 'parent')"
            )
            clusters.execute("insert into pathway values (1, 1, 1, 1), (2, 1, 1, 1)")
        # Arbitrary aliases and read-only cluster attachment, as used by the web app.
        connection = duckdb.connect()
        connection.execute(f"attach '{clusters_path}' as molecules (read_only)")
        connection.execute(f"attach '{results_path}' as output")
        connection.execute("use output; set search_path = 'output,molecules'")
        connection.execute(sql_files_nofk.experiment)
        connection.execute(sql_files_nofk.realizations)
        filename = (connection, None)
    else:
        filename = ":memory:"

    with connection_scope(RealizationDatabase, filename) as database:
        db = database.db
        if not attached:
            database.create_tables()
            db.execute("insert into cluster (id, common_name) values (1, 'parent')")
            db.execute("insert into pathway values (1, 1, 1, 1), (2, 1, 1, 1)")
        database.refresh_views()
        for view in ("experiment_result", "event_details", "event_report"):
            assert db.execute(f"select count(*) from {view}").fetchone() == (0,)

        db.execute("insert into experiment_config (id) values (1)")
        db.execute(
            "insert into experiment_run (id, experiment_config_id) values (1, 1), (2, 1)"
        )
        db.execute(
            "insert into single_pathway_experiment_result (id, experiment_run_id, pathway_id) "
            "values (1, 1, 1)"
        )
        db.execute(
            "insert into multi_pathway_experiment_result (id, experiment_run_id, cluster_id) "
            "values (2, 2, 1)"
        )
        db.execute(
            "insert into pathway_fragmentation (experiment_result_id, pathway_id, count) "
            "values (2, 1, 1), (2, 2, 1)"
        )
        db.execute(
            "insert into experiment_failure (id, experiment_run_id, pathway_id) values (3, 1, 1)"
        )
        db.execute(
            "insert into realization values (1, 1), (2, 2), (3, null), (4, 99), (5, 3)"
        )
        event_id = 0
        for realization_id in range(1, 6):
            types = (
                ("init", "collision", "fragmentation", "escape")
                if realization_id <= 2
                else ("init",)
            )
            for event_type in types:
                event_id += 1
                db.execute(
                    "insert into event_info values (?, ?, ?, "
                    "{x: 1.0, y: 2.0, z: -3.0, t: 4.0}, "
                    "{x: 5.0, y: 6.0, z: 7.0}, {x: 8.0, y: 9.0, z: 10.0}, 11, 12, 13)",
                    (event_id, realization_id, event_type),
                )
                if event_type == "collision":
                    db.execute(
                        "insert into collision_event values (?, 0.5, 14, false)",
                        (event_id,),
                    )
                elif event_type == "fragmentation":
                    db.execute(
                        "insert into fragmentation_event values (?, 2)", (event_id,)
                    )

        # Refresh also upgrades existing files with stale report definitions.
        db.execute("create or replace view event_report as select 0 as obsolete")
        database.refresh_views()
        assert db.execute("select * from experiment_result order by id").fetchall() == [
            (1, 1, 1, 1, True),
            (2, 2, 1, None, False),
        ]
        assert db.execute("select count(*) from event_details").fetchone() == (11,)
        assert db.execute(
            "select count(*), count(distinct event_id) from event_report"
        ).fetchone() == (8, 8)
        assert db.execute(
            "select experiment_result_id, cluster_id, count(*) from event_report group by all order by 1"
        ).fetchall() == [(1, 1, 4), (2, 1, 4)]
        assert db.execute(
            "select count(*) from event_report where event_type = 'init'"
        ).fetchone() == (2,)
        assert db.execute(
            "select event_type, theta, u_norm, accepted, pathway_id from event_report "
            "where experiment_result_id = 1 order by event_id"
        ).fetchall() == [
            ("init", None, None, None, None),
            ("collision", 0.5, 14, False, None),
            ("fragmentation", None, None, None, 2),
            ("escape", None, None, None, None),
        ]
        assert db.execute(
            "select count(*) from event_details where postime.z = z and z = -3 "
            "and x = 1 and y = 2 and t = 4 and velocity.x = 5 and omega.z = 10 "
            "and rot_energy = 11 and vibrational_energy = 12 and particle_index = 13"
        ).fetchone() == (11,)
        assert db.execute(
            "select count(*) from event_report where z = -3"
        ).fetchone() == (8,)

    if attached:
        # Refreshed persisted views can be consumed through read-only attachments.
        with duckdb.connect() as readonly:
            readonly.execute(f"attach '{clusters_path}' as molecules (read_only)")
            readonly.execute(f"attach '{results_path}' as output (read_only)")
            readonly.execute("use output; set search_path = 'output,molecules'")
            assert readonly.execute(
                "select count(*) from experiment_result"
            ).fetchone() == (2,)
            assert readonly.execute(
                "select count(*) from event_details"
            ).fetchone() == (11,)
            assert readonly.execute("select count(*) from event_report").fetchone() == (
                8,
            )


@pytest.fixture
def query_db():
    from apitofsim.workflow.db import RealizationDatabase, connection_scope

    with connection_scope(RealizationDatabase, ":memory:") as db:
        db.create_tables()
        for name in ("parent", "other parent", "product"):
            db.insert_cluster(name, 1, 1, 0, [1, 1, 1], [])
        db.insert_pathway(1, 3, 3)
        db.insert_pathway(1, 2, 3)
        db.insert_pathway(2, 3, 3)
        config_id = db.insert_config("queries", "{}")
        db.insert_run(config_id)
        db.insert_run(config_id)
        # Two single results for one parent, plus multi results without pathway counts.
        db.db.execute("""
            insert into single_pathway_experiment_result (id, experiment_run_id, pathway_id)
            values (1, 1, 1), (2, 1, 2), (3, 2, 1), (4, 1, 3);
            insert into multi_pathway_experiment_result (id, experiment_run_id, cluster_id)
            values (5, 1, 1), (6, 2, 1), (7, 1, 2);
            insert into realization values
            (1, 1), (2, 2), (3, 3), (4, 4), (5, 5), (6, 6), (7, 7),
            (8, 5), (9, null);
        """)
        # IDs deliberately differ from chronological order; equal times use ID ties.
        for event_id, realization_id, event_type, t in (
            (1, 1, "escape", 3),
            (2, 1, "collision", 2),
            (3, 1, "init", 0),
            (4, 2, "fragmentation", 1),
            (5, 3, "escape", 1),
            (6, 4, "escape", 1),
            (7, 5, "fragmentation", 1),
            (8, 5, "collision", 1),
            (9, 5, "escape", 2),
            (10, 6, "escape", 1),
            (11, 7, "escape", 1),
            (12, 9, "init", 0),
        ):
            db.db.execute(
                "insert into event_info values (?, ?, ?, {'x': 0, 'y': 0, 'z': 0, 't': ?}, "
                "{'x': 0, 'y': 0, 'z': 0}, {'x': 0, 'y': 0, 'z': 0}, 0, 0, 0)",
                (event_id, realization_id, event_type, t),
            )
        db.db.execute("""
            insert into collision_event values (2, 0.5, 2, false);
            insert into fragmentation_event values (7, null);
        """)
        yield db


@pytest.mark.parametrize(
    "scope, result_ids, realization_ids",
    [
        ({}, list(range(1, 8)), list(range(1, 10))),
        ({"run_id": 1, "parent": 1}, [1, 2, 5], [1, 2, 5, 8]),
        ({"run_id": 2, "parent": 1}, [3, 6], [3, 6]),
        ({"run_id": 1, "parent": 2}, [4, 7], [4, 7]),
        ({"result_id": 5}, [5], [5, 8]),
        ({"run_id": 2, "result_id": 5}, [], []),
        ({"parent": 2, "result_id": 5}, [], []),
        ({"run_id": 999}, [], []),
        ({"parent": 999}, [], []),
        ({"result_id": 999}, [], []),
    ],
)
def test_realization_query_scopes(query_db, scope, result_ids, realization_ids):
    import duckdb

    results = query_db.results_query(**scope)
    realizations = query_db.realizations_query(**scope)
    events = query_db.events_query(**scope)
    assert isinstance(results, duckdb.DuckDBPyRelation)
    assert isinstance(realizations, duckdb.DuckDBPyRelation)
    assert isinstance(events, duckdb.DuckDBPyRelation)
    assert results.order("experiment_result_id").select(
        "experiment_result_id"
    ).fetchall() == [(i,) for i in result_ids]
    assert realizations.order("realization_id").select("realization_id").fetchall() == [
        (i,) for i in realization_ids
    ]
    expected_events = query_db.db.sql("select * from event_info").fetchall()
    assert sorted(events.select("event_id").fetchall()) == sorted(
        (e[0],) for e in expected_events if e[1] in realization_ids
    )


def test_recorded_output_queries(query_db):
    assert query_db.results_query().order("experiment_result_id").fetchall() == (
        query_db.db.table("experiment_result").order("id").fetchall()
    )
    assert query_db.events_query().filter(
        "postime.x = x and postime.y = y and postime.z = z and postime.t = t"
    ).count("*").fetchone() == (12,)

    def counts(**scope):
        df = query_db.recorded_output_query(**scope).fetchdf()
        assert len(df) == 1
        return df.iloc[0].to_dict()

    assert counts(run_id=1, parent=1) == {
        "realization_count": 4,
        "event_count": 7,
        "escape_count": 2,
        "collision_event_count": 2,
        "collision_subtype_count": 1,
        "fragmentation_event_count": 2,
        "fragmentation_subtype_count": 1,
        "terminal_realization_count": 3,
    }
    assert counts()["realization_count"] == 9
    assert counts()["event_count"] == 12
    assert counts(run_id=2)["realization_count"] == 2
    assert counts(result_id=5)["terminal_realization_count"] == 1
    assert all(value == 0 for value in counts(run_id=999).values())
    rows = (
        query_db.events_query(run_id=1, parent=1, sort=True)
        .select(
            "event_id, realization_id, collision_event_id, fragmentation_event_id, pathway_id, accepted"
        )
        .fetchall()
    )
    assert rows == [
        (3, 1, None, None, None, None),
        (2, 1, 2, None, None, False),
        (1, 1, None, None, None, None),
        (4, 2, None, None, None, None),
        (7, 5, None, 7, None, None),
        (8, 5, None, None, None, None),
        (9, 5, None, None, None, None),
    ]
    # Lazy relations retain their own scope when other queries run on the connection.
    first = query_db.events_query(run_id=1, parent=1, sort=True)
    query_db.events_query(run_id=2).fetchall()
    assert first.select("event_id").fetchall() == [(row[0],) for row in rows]


@pytest.mark.parametrize("parameter", ["run_id", "parent", "result_id"])
def test_query_scope_parameters_are_values(query_db, parameter):
    import duckdb

    for method in (
        query_db.results_query,
        query_db.realizations_query,
        query_db.events_query,
        query_db.recorded_output_query,
    ):
        with pytest.raises(duckdb.ConversionException):
            method(**{parameter: "1 OR 1=1"}).fetchall()


def test_recorded_output_database_scope(query_db):
    from apitofsim.workflow.db import RealizationDatabase, connection_scope

    # A second database must not share aliases or counts with the populated one.
    with connection_scope(RealizationDatabase, ":memory:") as empty_db:
        empty_db.create_tables()
        assert empty_db.results_query().fetchall() == []
        assert empty_db.events_query(sort=True).fetchall() == []
        assert empty_db.recorded_output_query().fetchone() == (0,) * 8
        empty_db.insert_realization(None)
        assert empty_db.recorded_output_query().fetchone() == (1,) + (0,) * 7
        assert query_db.recorded_output_query().select(
            "realization_count"
        ).fetchone() == (9,)


@pytest.fixture
def collision_inputs():
    import numpy as np
    from apitofsim import Gas
    from apitofsim.api import Q_

    config = dict(
        lengths=Q_([1, 1, 1, 1, 1], "mm"),
        pressures=Q_([100, 2], "Pa"),
        T=Q_(300, "K"),
        gas=Gas(Q_(0.2, "nm"), Q_(4.8e-26, "kg"), 1.4),
    )
    # Includes a final held cell, just like the numerical skimmer output.
    skimmer = np.array([[100, 250, 80], [300, 150, 20]], dtype=float)
    return config, Q_(0.5, "nm"), skimmer


@pytest.mark.parametrize(
    "velocity", [[0, 0, 0], [200, 0, 0], [0, 0, 200], [0, 0, -200]]
)
def test_collision_profile_physics(collision_inputs, velocity):
    import math

    import numpy as np
    from apitofsim import collision_profile, consts
    from apitofsim.api import Q_

    config, radius, skimmer = collision_inputs
    profile = collision_profile(
        config, radius, skimmer=skimmer, mesh_skimmer=Q_(0.5, "mm")
    )
    # Unsorted, repeated, both endpoints, interpolated and held-cell positions.
    z = np.array([5, 1, 1.25, 0, 2, 1.25, 1.75, 2.001])
    actual = profile.frequency(Q_(z, "mm"), Q_(velocity, "m/s"))
    expected = []
    for position in z:
        if 1 <= position <= 2:
            vgas, temperature, pressure = [
                np.interp(position, [1, 1.5], skimmer[:, i]) for i in range(3)
            ]
        else:
            vgas, temperature, pressure = 0, 300, (100 if position < 1 else 2)
        relative = np.asarray(velocity, dtype=float) - [0, 0, vgas]
        speed = np.linalg.norm(relative)
        kT = consts.boltzmann * temperature
        mobility = kT / config["gas"].mass.to("kg").magnitude
        n = pressure / kT
        R = (radius + config["gas"].radius).to("m").magnitude
        if speed == 0:
            frequency = 2 * math.sqrt(2 * math.pi * mobility) * n * R**2
        else:
            frequency = (
                2
                * math.pi
                * n
                * R**2
                * (
                    0.5
                    * (mobility / speed + speed)
                    * math.erf(speed / math.sqrt(2 * mobility))
                    + math.sqrt(mobility / (2 * math.pi))
                    * math.exp(-(speed**2) / (2 * mobility))
                )
            )
        expected.append(frequency)
    np.testing.assert_allclose(actual.to("1/s").magnitude, expected, rtol=1e-12)
    vectors = np.tile(velocity, (len(z), 1))
    np.testing.assert_allclose(
        profile.frequency(Q_(z / 10, "cm"), Q_(vectors * 100, "cm/s")).magnitude,
        expected,
    )
    np.testing.assert_allclose(
        profile.frequency(
            z * 1e-3, np.array(velocity), quantities_strict=False
        ).magnitude,
        expected,
    )


def test_collision_profile_validation_and_snapshot(collision_inputs):
    import numpy as np
    from apitofsim import collision_profile
    from apitofsim.api import Q_
    from pint import DimensionalityError

    config, radius, skimmer = collision_inputs
    profile = collision_profile(
        config, radius, skimmer=skimmer, mesh_skimmer=Q_(0.5, "mm")
    )
    z = Q_([0, 1.25, 4], "mm")
    baseline = profile.frequency(z)
    skimmer[:] = 0
    config["pressures"] *= 2
    np.testing.assert_array_equal(profile.frequency(z).magnitude, baseline.magnitude)
    assert profile.frequency(Q_([], "m")).shape == (0,)
    assert profile.frequency(Q_([], "m"), Q_(np.empty((0, 3)), "m/s")).shape == (0,)
    for invalid in (
        Q_(1, "m"),
        Q_([[0]], "m"),
        Q_([-1], "mm"),
        Q_([6], "mm"),
        Q_([np.nan], "m"),
    ):
        with pytest.raises(ValueError):
            profile.frequency(invalid)
    for invalid in (
        Q_(0, "m/s"),
        Q_([1, 2], "m/s"),
        Q_(np.zeros((1, 3)), "m/s"),
        Q_([0, 0, np.inf], "m/s"),
    ):
        with pytest.raises(ValueError):
            profile.frequency(z, invalid)
    with pytest.raises(ValueError, match="pint.Quantity"):
        profile.frequency([0])
    with pytest.raises(DimensionalityError):
        profile.frequency(Q_([1], "s"))
    with pytest.raises(DimensionalityError):
        profile.frequency(z, Q_([0, 0, 0], "m"))
    with pytest.raises(ValueError, match="Prepared skimmer"):
        collision_profile(config, radius)
    with pytest.raises(ValueError, match="Missing.*gas"):
        collision_profile(
            {k: v for k, v in config.items() if k != "gas"}, radius, skimmer=skimmer
        )
    with pytest.raises(ValueError, match="temperature"):
        collision_profile(config, radius, skimmer=skimmer, mesh_skimmer=Q_(0.5, "mm"))


def test_collision_profile_readonly_inputs(tmp_path, collision_inputs):
    import duckdb
    import numpy as np
    from apitofsim import collision_profile
    from apitofsim.api import Q_

    config, _, skimmer = collision_inputs
    path = str(tmp_path / "experiment.duckdb")
    db = ExperimentDatabase(path)
    db.create_tables()
    config_id = db.insert_config("profile", config)
    run_id = db.insert_run(config_id)
    db.db.execute(
        "insert into cluster values (1, 'cluster', 100, -1, 0, [1, 2, 3], [10, 20], null)"
    )
    db.db.execute(
        "insert into multi_pathway_experiment_result values (1, ?, 1, 0, 0, 0, 1, 0, 0)",
        [run_id],
    )
    db.close()
    for attached in (False, True):
        conn = (
            duckdb.connect(":memory:")
            if attached
            else duckdb.connect(path, read_only=True)
        )
        if attached:
            conn.execute(f"attach '{path}' as results (read_only)")
            conn.execute("use results")
        db = ExperimentDatabase((conn, None))
        resolved, cluster = db.collision_profile_inputs(result_id=1)
        profile = collision_profile(
            resolved, cluster, skimmer=skimmer, mesh_skimmer=Q_(0.5, "mm")
        )
        assert np.isfinite(profile.frequency(Q_([0, 1.25, 2, 5], "mm")).magnitude).all()
        direct, _ = db.collision_profile_inputs(run_id=run_id, cluster_id=1)
        assert direct["T"] == resolved["T"]
        with pytest.raises(ValueError, match="Unknown experiment result"):
            db.collision_profile_inputs(result_id=99)
        with pytest.raises(ValueError, match="No stored configuration"):
            db.collision_profile_inputs(run_id=99, cluster_id=1)
        with pytest.raises(ValueError, match="Unknown cluster"):
            db.collision_profile_inputs(run_id=run_id, cluster_id=99)
        db.close()


def test_collision_profile_prepared_skimmer(collision_inputs):
    import numpy as np
    from apitofsim import collision_profile
    from apitofsim import skimmer as compute_skimmer
    from apitofsim.api import Q_

    config = ConfigFile(
        filename=os.path.join(os.environ["DATA_DIR"], "raw/config.in")
    ).into_json_config()
    skimmer = compute_skimmer(config)
    radius = collision_inputs[1]
    profile = collision_profile(config, radius, skimmer=skimmer)
    start = config["lengths"][0].to("m").magnitude
    end = start + config["lengths"][-1].to("m").magnitude
    z = Q_([start, (start + end) / 2, end], "m")
    velocities = Q_([[0, 0, 0], [100, 200, -300], [0, 0, 400]], "m/s")
    expected = [
        profile.frequency(z[i : i + 1], velocities[i]).magnitude[0] for i in range(3)
    ]
    np.testing.assert_allclose(profile.frequency(z, velocities).magnitude, expected)
    np.testing.assert_allclose(
        profile.frequency(z).magnitude,
        profile.frequency(z, Q_([0, 0, 0], "m/s")).magnitude,
    )


def test_skimmer_config_dispatch():
    import numpy as np
    import orjson
    from apitofsim import skimmer
    from apitofsim.config import dump_to_raw

    config = ConfigFile(
        filename=os.path.join(os.environ["DATA_DIR"], "raw/config.in")
    ).into_json_config()
    parameters = dict(
        T0=config["T"],
        P0=config["pressures"][0],
        rmax=config["lengths"][-1],
        dc=config["dc"],
        alpha_factor=config["alpha_factor"],
        gas=config["gas"],
        N=config["N_iter"],
        M=config["M_iter"],
        resolution=config["resolution"],
        tolerance=config["tolerance"],
    )
    expected = skimmer(**parameters)
    np.testing.assert_array_equal(skimmer(*parameters.values()), expected)
    np.testing.assert_array_equal(skimmer(config), expected)
    np.testing.assert_array_equal(skimmer(orjson.loads(dump_to_raw(config))), expected)
    np.testing.assert_array_equal(
        skimmer(config, output_pandas=True).to_numpy(), expected
    )
    with pytest.raises(TypeError, match="cannot be combined"):
        skimmer(config, N=10)
    with pytest.raises(TypeError, match="cannot be combined"):
        skimmer(config, 10)


def test_quantity_energy_helpers():
    import numpy as np
    import pandas as pd
    from apitofsim import (
        available_fragmentation_energy,
        energy_in_joules,
        internal_energy,
    )
    from apitofsim.api import Q_, ureg
    from pint import DimensionalityError

    joules = energy_in_joules(Q_(300, "K"))
    assert joules.magnitude == pytest.approx(
        (Q_(300, "K") * ureg.boltzmann_constant).to("J").magnitude, abs=0
    )
    assert energy_in_joules(joules.to("eV")).magnitude == pytest.approx(
        joules.magnitude
    )
    molar = (joules * ureg.avogadro_constant).to("kcal/mol")
    assert energy_in_joules(molar).magnitude == pytest.approx(joules.magnitude, abs=0)
    assert internal_energy(Q_(100, "K"), molar).to(
        "K", "boltzmann"
    ).magnitude == pytest.approx(400)
    assert available_fragmentation_energy(joules, molar * 2).magnitude == pytest.approx(
        -joules.magnitude, abs=0
    )
    assert internal_energy(None, joules) is None
    assert internal_energy(joules, pd.NA) is None
    assert available_fragmentation_energy(joules, None) is None
    assert np.isnan(internal_energy(float("nan"), 1).magnitude)
    assert internal_energy(1, 2).magnitude == 3
    with pytest.raises(DimensionalityError):
        energy_in_joules(Q_(1, "s"))


@pytest.fixture
def rate_curve_db():
    from apitofsim.api import Q_
    from apitofsim.workflow.db import RealizationDatabase, connection_scope

    with connection_scope(RealizationDatabase, ":memory:") as db:
        db.create_tables()
        for name, energy, frequencies, charge in (
            ("parent", -10.0, [100.0, 200.0, 300.0], -1),
            ("product", -6.0, [100.0, 200.0, 300.0], -1),
            ("neutral", -3.9, [], 0),
        ):
            db.insert_cluster(name, 100, charge, energy, [1.0, 2.0, 3.0], frequencies)
        db.insert_pathway(1, 2, 3)
        db.insert_pathway(1, 2, 3)
        db.insert_pathway(1, 2, 3)
        # Independent DOS/rate widths; a decoy with the same width must not win.
        db.db.execute(
            "insert into histogram_params values (1, 20, 100), (2, 10, 40), (3, 10, 60)"
        )
        db.db.execute(
            "insert into k_rate (pathway_id, histogram_params_id, data) values (1, 2, [10,20,40,80]), (2, 2, [1,2,3,4]), (1, 3, [9,9,9,9,9,9])"
        )
        config = {"fragmentation_energy": Q_(150, "K"), "bin_width": Q_(99, "K")}
        config_id = db.insert_config("selected", config)
        for rate_id in (2, 3):
            run = db.insert_run(
                config_id, dos_histogram_id=1, rate_histogram_id=rate_id
            )
            db.db.execute(
                "insert into single_pathway_experiment_result (experiment_run_id, pathway_id) values (?, 1)",
                (run,),
            )
        db.db.execute(
            "insert into multi_pathway_experiment_result (experiment_run_id, cluster_id) values (1, 1)"
        )
        db.db.execute(
            "insert into pathway_fragmentation (experiment_result_id, pathway_id, count) values (3, 1, 7), (3, 2, 0)"
        )
        yield db


def test_pathway_rate_curves(rate_curve_db):
    import numpy as np
    from apitofsim.api import (
        Q_,
        ApiTofRateConstantOverflow,
        Gas,
        MassSpecInputFragmentationPathway,
        MassSpecSubstanceSingleInput,
    )

    db = rate_curve_db
    single = db.pathway_rate_curves(1, 1, 1)
    assert len(single) == 1
    curve = single[0]
    assert (
        curve.pathway_id,
        curve.parent_id,
        curve.product1_id,
        curve.product2_id,
    ) == (1, 1, 2, 3)
    assert (curve.dos_histogram_id, curve.rate_histogram_id) == (1, 2)
    np.testing.assert_allclose(
        curve.histogram.x.to("K", "boltzmann").magnitude, [5, 15, 25, 35]
    )
    np.testing.assert_equal(curve.rates.to("1/s").magnitude, [10, 20, 40, 80])
    assert curve.bonding_energy.to("K", "boltzmann").magnitude == pytest.approx(150)
    assert db.pathway_rate_curves(2, 1)[0].rates[0].magnitude == 9
    np.testing.assert_allclose(
        curve.rate_at(Q_([-1, 0, 2, 5, 10, 15, 20, 35, 39.9], "K")).magnitude,
        [0, 0, 10, 10, 15, 20, 30, 80, 80],
    )
    with pytest.raises(ApiTofRateConstantOverflow):
        curve.rate_at(Q_(41, "K"))
    assert curve.rate_at(Q_(41, "K"), strict=False).magnitude == 80
    upper_edge = (curve.histogram.x[1] - curve.histogram.x[0]) * len(curve.histogram.x)
    assert curve.rate_at(upper_edge).magnitude == 80
    assert curve.rate_at(None) is None
    assert np.isnan(curve.rate_at(Q_(float("nan"), "J")).magnitude)

    multi = db.pathway_rate_curves(1, 1, 3)
    assert [c.pathway_id for c in multi] == [
        1,
        2,
    ]  # Includes zero-event enabled pathway.
    clusters = db.clusters_objects_indexed()
    native = MassSpecInputFragmentationPathway(
        clusters[1], clusters[2], clusters[3], curve.histogram
    )
    assert multi[0].bonding_energy.magnitude == native.bonding_energy
    # The configured override is used by single-pathway preparation only.
    gas = Gas(Q_(1e-10, "m"), Q_(1e-26, "kg"), 1.4)
    native_single = MassSpecSubstanceSingleInput(
        clusters[1],
        clusters[2],
        clusters[3],
        gas,
        curve.histogram,
        curve.histogram,
        fragmentation_energy=Q_(150, "K"),
    )
    assert curve.bonding_energy.magnitude == native_single.pathways[0].bonding_energy
    all_curves = db.pathway_rate_curves(1, 1, 3, all_pathways=True)
    assert [c.participating for c in all_curves] == [True, True, False]
    assert all_curves[2].histogram is None
    assert all_curves[2].rates is None
    with pytest.raises(ValueError, match="Missing rate curve"):
        all_curves[2].rate_at(Q_(100, "K"))
    # Removing an enabled curve stays explicit, with no compute-on-read.
    db.db.execute("delete from k_rate where pathway_id = 2")
    assert db.pathway_rate_curves(1, 1, 3)[1].histogram is None


def test_rate_curve_provenance_errors(rate_curve_db):
    from apitofsim.workflow import RateCurveProvenanceError

    db = rate_curve_db
    for args in ((999, 1), (1, 1), (1, 2, 1), (1, 1, 999), (2, 1, 1)):
        with pytest.raises(RateCurveProvenanceError):
            db.pathway_rate_curves(*args)
    db.db.execute("update experiment_run set run_config = '{}' where id = 2")
    with pytest.raises(RateCurveProvenanceError, match="Missing DOS/rate"):
        db.pathway_rate_curves(2, 1)
    db.db.execute(
        "insert into k_rate (pathway_id, histogram_params_id, data) values (1, 2, [1,2,3,4])"
    )
    with pytest.raises(RateCurveProvenanceError, match="Ambiguous rate cache"):
        db.pathway_rate_curves(1, 1, 1)
    db.db.execute("delete from pathway_fragmentation")
    with pytest.raises(RateCurveProvenanceError, match="enabled-pathway"):
        db.pathway_rate_curves(1, 1, 3)


@pytest.mark.parametrize("attached", [False, True])
def test_rate_curves_readonly(rate_curve_db, tmp_path, attached):
    import duckdb
    from apitofsim.workflow.db import RealizationDatabase, connection_scope

    path = str(tmp_path / "curves.duckdb")
    source = rate_curve_db.db
    clusters_path = str(tmp_path / "molecules.duckdb")
    source.execute(f"attach '{path}' as saved")
    if attached:
        source.execute(f"attach '{clusters_path}' as molecules")
    cluster_tables = {
        "cluster",
        "pathway",
        "histogram_params",
        "cluster_dos",
        "products_dos",
        "k_rate_mesh",
        "k_rate",
    }
    for (name,) in source.execute(
        "select table_name from duckdb_tables() where database_name = 'memory'"
    ).fetchall():
        destination = "molecules" if attached and name in cluster_tables else "saved"
        source.execute(
            f'create table {destination}.main."{name}" as select * from "{name}"'
        )
    source.execute("detach saved")
    if attached:
        source.execute("detach molecules")
        connection = duckdb.connect()
        connection.execute(f"attach '{path}' as output (read_only)")
        connection.execute(f"attach '{clusters_path}' as molecules (read_only)")
        connection.execute("set search_path = 'output,molecules'")
    else:
        connection = duckdb.connect(path, read_only=True)
    with connection_scope(RealizationDatabase, (connection, None)) as db:
        before = connection.execute("select count(*) from k_rate").fetchone()
        assert len(db.pathway_rate_curves(1, 1, 3)) == 2
        assert before == connection.execute("select count(*) from k_rate").fetchone()
        assert connection.execute(
            "select count(*) from duckdb_views() where not internal"
        ).fetchone() == (0,)


def test_recorded_run_preparation(rate_curve_db):
    import json

    import numpy as np
    from apitofsim.api import Q_
    from apitofsim.workflow.runners import DerivedDataPreparer

    db = rate_curve_db
    config = {"fragmentation_energy": Q_(2, "kJ/mol")}
    db.record_run_preparation(1, config, 1, 3)
    metadata = json.loads(
        db.db.execute("select run_config from experiment_run where id = 1").fetchone()[
            0
        ]
    )
    assert metadata["simulation_mode"] == "SINGLE_CLUSTER"
    assert (metadata["dos_histogram_id"], metadata["rate_histogram_id"]) == (1, 3)
    curve = db.pathway_rate_curves(1, 1, 1)[0]
    assert curve.bonding_energy.to("eV").magnitude == pytest.approx(0.0207285393)
    preparer = DerivedDataPreparer(db)
    preparer.dos_histogram_id, preparer.rate_histogram_id = 1, 3
    np.testing.assert_equal(
        preparer.histogram({}, np.ones(5)).x.magnitude, [10, 30, 50, 70, 90]
    )
    np.testing.assert_equal(
        preparer.histogram({}, np.ones(6), rate=True).x.magnitude,
        [5, 15, 25, 35, 45, 55],
    )


def test_cached_run_preparation(rate_curve_db, monkeypatch):
    import numpy as np
    from apitofsim.api import Q_
    from apitofsim.workflow import ExperimentRunner, runners

    db = rate_curve_db
    db.db.execute(
        "insert into cluster_dos (histogram_params_id, cluster_id, data) values (1, 1, [1,2,3,4,5]), (1, 2, [1,2,3,4,5])"
    )
    runner = ExperimentRunner(db)
    runner.start_run(1)
    monkeypatch.setattr(
        runner.preparer, "_run_skimmer", lambda config: np.zeros((2, 6))
    )
    monkeypatch.setattr(runners, "MassSpectrometer", lambda *args, **kwargs: None)
    config = dict(
        next(db.iter_configs()).config,
        lengths=None,
        voltages=None,
        T=Q_(300, "K"),
        pressures=None,
    )
    _, _, _, _, rates, densities = runner._prepare_from_config(
        config,
        pathways=[1],
        cached_densityandrate=(1, 3),
    )
    result_id = db.db.execute(
        "insert into single_pathway_experiment_result (experiment_run_id, pathway_id) values (?, 1) returning id",
        (runner.current_run_id,),
    ).fetchone()[0]
    curve = db.pathway_rate_curves(runner.current_run_id, 1, result_id)[0]
    prepared = runner.preparer.histogram(config, rates[1], rate=True)
    np.testing.assert_equal(
        curve.histogram.x.to("K", "boltzmann").magnitude, prepared.x.magnitude
    )
    assert len(runner.preparer.histogram(config, densities[1]).x) == 5
    assert len(curve.histogram.x) == 6


def test_serial_rate_curve_parent(rate_curve_db):
    from apitofsim.workflow import RateCurveProvenanceError

    db = rate_curve_db
    db.insert_pathway(2, 2, 3)
    db.db.execute(
        "insert into pathway_fragmentation (experiment_result_id, pathway_id, count) values (3, 4, 0)"
    )
    # A known downstream parent uses its own outgoing pathways, not the root's.
    downstream = db.pathway_rate_curves(1, 2, 3)
    assert [(c.pathway_id, c.parent_id) for c in downstream] == [(4, 2)]
    assert downstream[0].histogram is None
    with pytest.raises(RateCurveProvenanceError, match="Parent is not identified"):
        db.pathway_rate_curves(1, 3, 3)
