import os
import signal
import subprocess
import time

import pytest
from apitofsim.config import ConfigFile
from apitofsim.workflow import ExperimentDatabase, ExperimentRunner, ingest_legacy_one
from click.testing import CliRunner


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


@pytest.mark.parametrize("mode", ["SINGLE_CLUSTER", "CLUSTER_TREE"])
def test_init_events_workflow(tmp_path, monkeypatch, mode):
    import apitofsim.api as api
    import numpy as np
    from apitofsim.cli import prepare
    from apitofsim.workflow.base import SimulationMode
    from apitofsim.workflow.db import RealizationDatabase, connection_scope

    data_dir = os.environ["DATA_DIR"]
    filename = str(tmp_path / "realizations.duckdb")
    result = CliRunner(catch_exceptions=False).invoke(
        prepare,
        ["create", data_dir + "/besel/config.toml", filename, "--db-type=realization"],
    )
    assert result.exit_code == 0
    original_mass_spec = api.mass_spec
    num_runs = 0

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
            state = event.state
            if isinstance(event, api.InitEvent):
                assert state.realization not in initial_states
                initial_states[state.realization] = state
                np.testing.assert_array_equal(state.postime, np.zeros(4))
                assert np.isfinite(state.velocity).all()
                assert np.isfinite(state.omega).all()
                assert np.isfinite(state.rot_energy) and state.rot_energy >= 0
                assert np.isfinite(state.internal_energy) and state.internal_energy >= 0
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
            recorder(event)

        counters = original_mass_spec(
            ms, subs, n, **{**kwargs, "event_callback": record}
        )
        assert set(initial_states) == set(range(n))
        for realization, state in initial_states.items():
            stored = recorder.db.db.execute(
                "select postime, velocity, omega, rot_energy, internal_energy, particle_index "
                "from init_event where realization_id = ?",
                (recorder.realization_ids[realization],),
            ).fetchall()
            assert len(stored) == 1
            postime, velocity, omega, rot, internal, particle = stored[0]
            np.testing.assert_array_equal(list(postime.values()), state.postime)
            np.testing.assert_array_equal(list(velocity.values()), state.velocity)
            np.testing.assert_array_equal(list(omega.values()), state.omega)
            assert (rot, internal, particle) == (
                state.rot_energy,
                state.internal_energy,
                state.particle_index,
            )

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
            assert state.internal_energy == initial_states[realization].internal_energy
        num_runs += 1
        return counters

    monkeypatch.setattr(api, "mass_spec", check_mass_spec)
    with connection_scope(RealizationDatabase, filename) as db:
        runner = ExperimentRunner(db)
        runner.run_prepared_config(mode=SimulationMode[mode])
        assert num_runs > 0
        count = db.db.execute("select count(*) from init_event").fetchone()[0]
        assert count == db.db.execute("select count(*) from realization").fetchone()[0]
        assert (
            db.db.execute(
                "select count(*) from realization where experiment_result_id is null"
            ).fetchone()[0]
            == 0
        )
        db.refresh_views()
        assert (
            db.db.execute(
                "select count(*) from event_report where event_type = 'init'"
            ).fetchone()[0]
            == count
        )
