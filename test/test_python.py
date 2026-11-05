import os
import signal
import subprocess
import sys
import time

import pytest
import stream_snapshots as snapshot
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


def _cancellation_stream(tree=False, slow=False, n=1_000_000, events=False):
    import numpy as np
    from apitofsim import api

    raw = api.apitofsimraw
    ms = raw.MassSpectrometer(
        np.array([[0.0, 300.0, 100.0], [0.0, 300.0, 100.0]]),
        0.0005,
        np.array([0.1 if slow else 0.001, 0.001, 0.001, 0.001, 0.001]),
        np.arange(5.0) * 10,
        300.0,
        np.array([100.0, 100.0]),
        None,
        None,
    )
    energies = np.arange(0.5, 1000.0) * 100 * api.consts.boltzmann
    density = raw.Histogram(energies, np.ones(1000))
    pathways = [
        raw.MassSpecInputFragmentationPathway(
            raw.Histogram(energies, np.zeros(1000)), 1e10
        )
    ]
    gas = raw.Gas(2.46e-10, 4.85e-26, 1.4)
    if tree:
        subs = raw.MassSpecSubstanceTreeInput(
            -1,
            gas,
            [raw.MSSubstanceTreeCluster(3.6e-25, 5e-10, density)],
            pathways,
            [raw.MSSubstanceTreeNode(0, [0])],
            [raw.MSSubstanceTreePathway(0, None)],
        )
    else:
        subs = raw.MassSpecSubstanceSingleInput(
            -1, 3.6e-25, 5e-10, density, pathways, gas
        )
    return api._MassSpecBatchIterator(ms, subs, n, logconf=(0, events))


@pytest.mark.parametrize("tree", [False, True])
@pytest.mark.parametrize("early", [False, True])
def test_iterator_cancellation(tree, early):
    import numpy as np
    from apitofsim import api

    with _cancellation_stream(tree, slow=early, events=True) as stream:
        latest = None
        initial = 0
        finals = []
        requested = False
        for item in snapshot.flatten_batches(stream):
            if isinstance(item, snapshot.InitEvent):
                initial += 1
            if isinstance(item, api.MassSpecIntermediateCounter):
                latest = item.counters
            if not requested and (
                (early and initial) or (not early and latest is not None)
            ):
                stream.request_cancel()
                stream.request_cancel()
                requested = True
            if isinstance(item, api.MassSpecFinalResult):
                finals.append(item)
        assert requested
        assert len(finals) == 1
        final = finals[0]
        assert final.cancelled and stream.cancelled
        assert final.counters.n_realizations < 1_000_000
        if early:
            assert final.counters.n_realizations == 0
            assert initial > 0  # Events from an interrupted realization are retained.
        else:
            assert final.counters.n_realizations > 0
            for actual, expected in zip(final.counters, latest):
                np.testing.assert_array_equal(actual, expected)
        stream.request_cancel()
        stream.join_if_joinable()
        stream.join_if_joinable()
        assert list(snapshot.flatten_batches(stream)) == []


@pytest.mark.parametrize("tree", [False, True])
@pytest.mark.parametrize("n", [0, 3])
def test_iterator_cancellation_after_completion(tree, n):
    from apitofsim import api

    with _cancellation_stream(tree, n=n) as stream:
        stream.join_if_joinable()  # Complete before draining the queue.
        stream.request_cancel()
        items = list(snapshot.flatten_batches(stream))
        final = items[-1]
        assert isinstance(final, api.MassSpecFinalResult)
        assert final.counters.n_realizations == n
        assert not final.cancelled and not stream.cancelled
        stream.request_cancel()
        assert not stream.cancelled


def test_iterator_cancellation_isolation():
    with (
        _cancellation_stream(slow=True) as cancelled,
        _cancellation_stream(n=3) as independent,
    ):
        cancelled.request_cancel()
        assert list(snapshot.flatten_batches(cancelled))[-1].cancelled
        final = list(snapshot.flatten_batches(independent))[-1]
        assert final.counters.n_realizations == 3
        assert not final.cancelled


@pytest.mark.parametrize("action", ["next", "read_batch", "join", "sigint"])
@pytest.mark.parametrize("tree", [False, True])
def test_iterator_cancellation_blocking(action, tree):
    # A subprocess bounds failures if a blocking binding accidentally keeps the GIL.
    code = f"""
import os, runpy, signal, threading, time
import stream_snapshots as snapshot
make_stream = runpy.run_path({__file__!r})['_cancellation_stream']
stream = make_stream(tree={tree!r})
# Ensure completed work is present before interrupting the remaining simulation.
first = next(snapshot.flatten_batches(stream))
assert first.counters.n_realizations > 0
def interrupt():
    time.sleep(.05)
    if {action!r} == 'sigint':
        os.kill(os.getpid(), signal.SIGINT)
    else:
        stream.request_cancel()
        stream.request_cancel()
        stream.join_if_joinable()
requester = threading.Thread(target=interrupt)
requester.start()
try:
    with stream:
        if {action!r} == 'join':
            stream.join_if_joinable()
        if {action!r} == 'read_batch':
            output = []
            while True:
                try:
                    output.extend(stream.read_batch(512).items)
                except StopIteration:
                    break
            final = output[-1]
        else:
            final = list(snapshot.flatten_batches(stream))[-1]
        assert {action!r} != 'sigint', 'SIGINT must raise KeyboardInterrupt'
        assert final.cancelled
        assert first.counters.n_realizations <= final.counters.n_realizations < 1_000_000
        assert list(snapshot.flatten_batches(stream)) == []
except KeyboardInterrupt:
    assert {action!r} == 'sigint'
requester.join()
print('done')
"""
    child = subprocess.run(
        [sys.executable, "-X", "dev", "-c", code],
        env=dict(os.environ, PYTHONPATH=os.pathsep.join(sys.path)),
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert child.returncode == 0, child.stdout + child.stderr
    assert child.stdout.strip() == "done"


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


@pytest.mark.parametrize(
    "edges",
    [
        [(0, 1), (1, 2), (2, None)],
        [(0, 1), (0, 2), (1, 3), (2, 3), (3, None)],
    ],
)
def test_tree_indices(edges, monkeypatch):
    from types import SimpleNamespace

    from apitofsim.workflow import runners

    lookup = {
        cluster_id: dict(
            cluster_id=cluster_id,
            cluster=SimpleNamespace(charge=-1),
            payload=object(),
            pathway_ids=[],
            pathway_products=[],
        )
        for cluster_id, _ in edges
    }
    for pathway_id, (parent, product) in enumerate(edges):
        lookup[parent]["pathway_ids"].append(pathway_id)
        lookup[parent]["pathway_products"].append(product)

    runner = ExperimentRunner(None)
    monkeypatch.setattr(
        runner,
        "_prepare_cluster_tree_lookup",
        lambda *args: (lookup, dict(enumerate(edges))),
    )
    monkeypatch.setattr(
        runners,
        "MassSpecSubstanceTreeInput",
        lambda charge, gas, clusters, pathways, nodes, tree_pathways, **kwargs: (
            SimpleNamespace(
                cluster_payloads=clusters,
                pathway_payloads=pathways,
                tree_nodes=nodes,
                tree_pathways=tree_pathways,
            )
        ),
    )
    for cluster_ids, _, tree, root in runner._prepare_cluster_tree(
        {"gas": None}, None, None, None, None, None
    ):
        payload_clusters = {index: cluster for cluster, index in cluster_ids.items()}
        visited_nodes, visited_pathways = [], []

        def visit(node_index, expected_cluster):
            visited_nodes.append(node_index)
            node = tree.tree_nodes[node_index]
            assert payload_clusters[node.payload_idx] == expected_cluster
            assert (
                tree.cluster_payloads[node.payload_idx]
                is lookup[expected_cluster]["payload"]
            )
            assert len(node.pathway_indices) == len(
                lookup[expected_cluster]["pathway_ids"]
            )
            for pathway_index in node.pathway_indices:
                visited_pathways.append(pathway_index)
                pathway = tree.tree_pathways[pathway_index]
                parent, product = tree.pathway_payloads[pathway.payload_idx]
                assert parent == expected_cluster
                if product is None:
                    assert pathway.product_idx is None
                else:
                    visit(pathway.product_idx, product)

        visit(0, root["cluster_id"])
        assert sorted(visited_nodes) == list(range(len(tree.tree_nodes)))
        assert sorted(visited_pathways) == list(range(len(tree.tree_pathways)))
        assert set(payload_clusters.values()) == {
            payload_clusters[tree.tree_nodes[index].payload_idx]
            for index in visited_nodes
        }


def test_particle_snapshot_properties_and_array_lifetimes():
    import gc

    import numpy as np
    from apitofsim import apitofsimraw as raw

    ms = raw.MassSpectrometer(
        np.array([[0, 300, 1e-12], [0, 300, 1e-12]]),
        0.0005,
        np.full(5, 0.001),
        np.zeros(5),
        300,
        np.full(2, 1e-12),
        radius_pinhole=None,
    )
    density = raw.Histogram(1e-21, 32, np.ones(32))
    subs = raw.MassSpecSubstanceSingleInput(
        -1, 1e-25, 5e-10, density, [], raw.Gas(2e-10, 4.8e-26, 1.4)
    )
    arrays = []
    stream = raw.MassSpecIterator(
        ms, subs, 2, seed=42, logconf=(0, True), initial_batch_size=1, max_batch_size=1
    )
    try:
        while True:
            try:
                batch = next(stream)
                columns = batch.columns
                arrays.extend(columns.values())
                for column in columns.values():
                    assert not column.flags.writeable
            except StopIteration:
                break
    finally:
        stream.join_if_joinable()
    expected = [array.copy() for array in arrays]
    del batch, columns, stream
    gc.collect()
    for array, values in zip(arrays, expected, strict=True):
        np.testing.assert_array_equal(array, values)


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
            (1, "PATHWAY_AT_A_TIME"),
            (2, "SINGLE_CLUSTER"),
            (3, "SINGLE_CLUSTER"),
        ):
            db.execute(
                "insert into experiment_run (id,experiment_config_id,simulation_mode,run_config,start_time) values (?, 1, ?, '{}', ?)",
                (run_id, mode, start_time),
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
                "simulation_mode": "PATHWAY_AT_A_TIME"
                if is_single
                else "SINGLE_CLUSTER",
                "run_config": "{}",
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
    from apitofsim.workflow.db import ExperimentDatabase, connection_scope

    with connection_scope(ExperimentDatabase, ":memory:") as db:
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
    from apitofsim.workflow.db import ExperimentDatabase, connection_scope

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
    with connection_scope(ExperimentDatabase, (connection, None)) as db:
        before = connection.execute("select count(*) from k_rate").fetchone()
        assert len(db.pathway_rate_curves(1, 1, 3)) == 2
        assert before == connection.execute("select count(*) from k_rate").fetchone()
        assert connection.execute(
            "select count(*) from duckdb_views() where not internal and not temporary"
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
    assert "simulation_mode" not in metadata
    assert db.db.execute(
        "select simulation_mode from experiment_run where id = 1"
    ).fetchone() == ("SINGLE_CLUSTER",)
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


@pytest.mark.parametrize("tree", [False, True])
@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize("strict", [False, True])
def test_upstream_escape_invalidates_run(tree, streaming, strict):
    import numpy as np
    from apitofsim import api

    raw = api.apitofsimraw
    # A strong reverse field turns every launched particle back before chamber 1 ends.
    ms = api.MassSpectrometer(
        np.array([[0.0, 300.0, 100.0], [0.0, 300.0, 100.0]]),
        api.Q_(np.full(5, 0.001), "m"),
        api.Q_(-np.arange(5.0) * 1000, "V"),
        api.Q_(300.0, "K"),
        api.Q_(np.array([0.001, 0.001]), "Pa"),
        mesh_skimmer=api.Q_(0.0005, "m"),
        radius_pinhole=None,
    )
    energies = np.arange(0.5, 1000.0) * 100 * api.consts.boltzmann
    density = raw.Histogram(energies, np.ones(1000))
    pathways = [
        raw.MassSpecInputFragmentationPathway(
            raw.Histogram(energies, np.zeros(1000)), 1e10
        )
    ]
    gas = raw.Gas(2.46e-10, 4.85e-26, 1.4)
    if tree:
        subs = raw.MassSpecSubstanceTreeInput(
            -1,
            gas,
            [raw.MSSubstanceTreeCluster(3.6e-25, 5e-10, density)],
            pathways,
            [raw.MSSubstanceTreeNode(0, [0])],
            [raw.MSSubstanceTreePathway(0, None)],
        )
    else:
        subs = raw.MassSpecSubstanceSingleInput(
            -1, 3.6e-25, 5e-10, density, pathways, gas
        )
    initial = []
    with pytest.raises(
        api.ApiTofParticleEscapeError, match="axial velocity.*acceleration"
    ):
        if streaming:
            with api._MassSpecBatchIterator(
                ms.into_cpp(), subs, 8, strict=strict, seed=42, logconf=(0, True)
            ) as stream:
                for item in snapshot.flatten_batches(stream):
                    assert not isinstance(item, api.MassSpecFinalResult)
                    if isinstance(item, snapshot.InitEvent):
                        initial.append(item.state)
        else:
            api._mass_spec_legacy(
                ms,
                subs,
                8,
                strict=strict,
                seed=42,
                logconf=(0, True),
            )
    assert bool(initial) == streaming
    assert all(state.velocity[2] >= 0 for state in initial)
    assert issubclass(api.ApiTofParticleEscapeError, api.ApiTofError)
