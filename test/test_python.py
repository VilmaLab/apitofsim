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
