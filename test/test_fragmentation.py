"""Opt-in continuation, including native ownership and recorded after-states."""

import gc

import numpy as np
import pytest
from apitofsim import api
from apitofsim import apitofsimraw as raw
from apitofsim.config import import_raw_config, simulation_option


@pytest.mark.parametrize("atomic", [False, True])
def test_fragmentation_cache_persists_and_tracks_inputs(tmp_path, monkeypatch, atomic):
    from contextlib import closing

    from apitofsim.workflow import DerivedDataPreparer, ExperimentDatabase

    def cluster(mass, charge, atom=False):
        return api.ClusterData(
            api.Q_(mass, "amu"),
            api.Q_(0, "hartree"),
            np.ones(3),
            np.array([] if atom else [20.0, 36.0]),
            charge,
        )

    clusters = {
        1: cluster(120, -1),
        2: cluster(100, -1),
        3: cluster(20, 0, atomic),
        4: cluster(80, -1),
    }
    if not atomic:
        clusters[3].frequencies = np.array([40.0, 50.0])
    pathways = {1: (1, 2, 3), 2: (2, 4, 3), 3: (1, 2, 3)}
    densities = {1: np.ones(100), 2: np.ones(100)}
    config = {
        "fragmentation_state_method": "phase-space",
        "bin_width": api.Q_(10, "K"),
        "energy_max": api.Q_(1000, "K"),
    }
    path = str(tmp_path / "cache.duckdb")
    with closing(ExperimentDatabase(path)) as db:
        db.create_tables()
        db.db.execute("drop table fragmentation_dos")  # Upgrade an older database.
        preparer = DerivedDataPreparer(db)
        preparer._prepare_fragmentation_products(config, clusters, pathways, densities)
        assert list(preparer.fragmentation_products) == [(2, 3)]
        snapshot = preparer.fragmentation_products[2, 3].log_dos
        assert db.db.execute("select count(*) from fragmentation_dos").fetchone() == (
            1 if atomic else 3,
        )

    real_dos = raw.FragmentationDos
    calls = []

    def construct(*args):
        calls.append(len(args))
        return real_dos(*args)

    monkeypatch.setattr(raw, "FragmentationDos", construct)
    with closing(ExperimentDatabase(path)) as db:
        preparer = DerivedDataPreparer(db)
        preparer._prepare_fragmentation_products(config, clusters, pathways, densities)
        assert calls == [4] * (1 if atomic else 3)
        np.testing.assert_array_equal(
            preparer.fragmentation_products[2, 3].log_dos, snapshot
        )
        # The cached histogram's actual grid overrides requested config defaults.
        preparer.dos_histogram_id = db.db.execute(
            "insert into histogram_params values (default, 10, 1000) returning id"
        ).fetchone()[0]
        config["bin_width"] = api.Q_(5, "K")
        preparer._prepare_fragmentation_products(config, clusters, pathways, densities)
        assert calls[-1] == 4
        calls.clear()
        clusters[2].frequencies = np.array([20.0, 47.0])
        preparer._prepare_fragmentation_products(config, clusters, pathways, densities)
        assert calls == ([3] if atomic else [3, 4])
        # The same number of bins at another width is a different cache entry.
        preparer.dos_histogram_id = None
        config["energy_max"] = api.Q_(500, "K")
        preparer._prepare_fragmentation_products(config, clusters, pathways, densities)
        assert calls[-1] == 3
        assert db.db.execute("select count(*) from fragmentation_dos").fetchone() == (
            3 if atomic else 8,
        )
        preparer._prepare_fragmentation_products({}, clusters, pathways, densities)
        assert preparer.fragmentation_products == {}

        # A second neutral pair shares the daughter's allocation and cache row.
        clusters[5] = cluster(20, 0)
        clusters[5].frequencies = np.array([60.0, 70.0])
        pathways[4] = (1, 2, 5)
        preparer._prepare_fragmentation_products(config, clusters, pathways, densities)
        assert (
            preparer.fragmentation_products[2, 3].daughter_dos
            is preparer.fragmentation_products[2, 5].daughter_dos
        )
        if atomic:
            assert (
                preparer.fragmentation_products[2, 3].combined_dos
                is preparer.fragmentation_products[2, 3].daughter_dos
            )
        assert db.db.execute("select count(*) from fragmentation_dos").fetchone() == (
            5 if atomic else 10,
        )


@pytest.mark.parametrize("atomic", [False, True])
def test_fragmentation_snapshot_validation_and_ownership(atomic):
    daughter = raw.MolecularCluster(100, 0, np.ones(3), np.array([1.6, 2.3]), -1)
    neutral = (
        raw.AtomLikeCluster(20, 0, 0)
        if atomic
        else raw.MolecularCluster(20, 0, np.ones(3), np.array([3.7, 1000.0]), 0)
    )
    width = raw.consts.boltzmann
    original = raw.FragmentationProducts(daughter, neutral, width, 100 * width)
    snapshot = original.log_dos
    restored = raw.FragmentationProducts(
        daughter, neutral, width, 100 * width, snapshot
    )
    snapshot[:] = np.nan  # Snapshots and restored payloads own their arrays.
    np.testing.assert_array_equal(restored.log_dos, original.log_dos)
    for invalid in (snapshot, original.log_dos[:-1], original.log_dos[:, :2]):
        with pytest.raises(raw.ApiTofArgumentError, match="Invalid prepared"):
            raw.FragmentationProducts(daughter, neutral, width, 100 * width, invalid)


@pytest.mark.parametrize("batch_bytes", [64 * 1024 * 1024, 1500])
def test_fragmentation_cache_batches_different_grids(
    tmp_path, monkeypatch, batch_bytes
):
    from contextlib import closing

    import pyarrow as pa
    from apitofsim.workflow import DerivedDataPreparer, ExperimentDatabase, runners

    def cluster(mass, charge):
        return api.ClusterData(
            api.Q_(mass, "amu"),
            api.Q_(0, "hartree"),
            np.ones(3),
            np.array([float(mass), float(mass + 20)]),
            charge,
        )

    clusters = {i: cluster(140 - 20 * i, -1) for i in (1, 2, 4, 5)}
    clusters[3] = cluster(20, 0)
    pathways = {1: (1, 2, 3), 2: (2, 4, 3), 3: (4, 5, 3)}
    densities = {2: np.ones(100), 4: np.ones(50)}
    config = {"fragmentation_state_method": "phase-space"}
    calls = []
    real_insert = runners.insert_via_arrow

    def insert(conn, table, **kwargs):
        if table == "fragmentation_dos":
            assert kwargs["data"].type == pa.list_(pa.float64())
            calls.append((list(kwargs["key"]), kwargs["data"].to_pylist()))
        return real_insert(conn, table, **kwargs)

    monkeypatch.setattr(runners, "insert_via_arrow", insert)
    monkeypatch.setattr(runners, "_FRAGMENTATION_CACHE_BATCH_BYTES", batch_bytes)
    path = str(tmp_path / "cache.duckdb")
    snapshots = {}
    for warm in (False, True):
        with closing(ExperimentDatabase(path)) as db:
            if not warm:
                db.create_tables()
            preparer = DerivedDataPreparer(db)
            monkeypatch.setattr(
                preparer,
                "histogram",
                lambda config, data: api.Histogram.from_mesh(
                    api.Q_(10, "K"), api.Q_(10 * len(data), "K"), data
                ),
            )
            calls.clear()
            preparer._prepare_fragmentation_products(
                config, clusters, pathways, densities
            )
            if warm:
                assert calls == []
                for key, product in preparer.fragmentation_products.items():
                    np.testing.assert_array_equal(product.log_dos, snapshots[key])
            else:
                assert len(calls) == (1 if batch_bytes > 1500 else 4)
                assert all(
                    8 * sum(len(row) for row in data) <= batch_bytes or len(keys) == 1
                    for keys, data in calls
                )
                rows = [row for keys, data in calls for row in zip(keys, data)]
                assert len(rows) == len({key for key, _ in rows}) == 6
                assert sorted(len(data) for _, data in rows) == [50] * 3 + [100] * 3
                for key, data in rows:
                    stored = db.db.execute(
                        "select data from fragmentation_dos where key = ?", [key]
                    ).fetchone()[0]
                    np.testing.assert_array_equal(stored, data)
                snapshots = {
                    key: product.log_dos
                    for key, product in preparer.fragmentation_products.items()
                }


def test_shared_fragmentation_marginals_and_lifetime():
    daughter = raw.MolecularCluster(100, 0, np.ones(3), np.array([1.6, 2.3]), -1)
    neutral = raw.MolecularCluster(20, 0, np.ones(3), np.array([3.7, 1000.0]), 0)
    width = raw.consts.boltzmann
    d = raw.FragmentationDos(daughter.frequencies, width, 100 * width)
    n = raw.FragmentationDos(neutral.frequencies, width, 100 * width)
    combined = d.extend(neutral.frequencies)
    first = raw.FragmentationProducts(daughter, neutral, d, n, combined)
    second = raw.FragmentationProducts(daughter, neutral, d, n, combined)
    reference = raw.FragmentationProducts(daughter, neutral, width, 100 * width)
    assert first.daughter_dos is second.daughter_dos
    assert first.neutral_dos is second.neutral_dos
    assert first.combined_dos is second.combined_dos
    np.testing.assert_array_equal(first.log_dos, reference.log_dos)
    d.log_counts[:] = np.nan  # Exposed arrays are independent snapshots.
    del d, n, combined, first
    gc.collect()
    np.testing.assert_array_equal(second.log_dos, reference.log_dos)
    wrong_grid = raw.FragmentationDos(daughter.frequencies, width, 50 * width)
    with pytest.raises(raw.ApiTofArgumentError, match="grid"):
        raw.FragmentationProducts(
            daughter, neutral, wrong_grid, second.neutral_dos, second.combined_dos
        )


def make_tree(method=None, *, atomic=False):
    def molecule(mass, charge):
        return raw.MolecularCluster(mass, 0, np.ones(3), np.array([100.0]), charge)

    clusters = [molecule(mass, -1) for mass in (120, 100, 80)]
    neutral = raw.AtomLikeCluster(20, 0, 0) if atomic else molecule(20, 0)
    width = 10 * raw.consts.boltzmann
    density = raw.Histogram(width, 1000, np.ones(1000))
    rate = raw.Histogram(width, 1000, np.full(1000, 1e12))
    pathways = [
        raw.MassSpecInputFragmentationPathway(
            rate,
            0,
            products=raw.FragmentationProducts(daughter, neutral, width, 1000 * width),
        )
        for daughter in clusters[1:]
    ]
    # A final zero-rate channel lets the last daughter escape.
    pathways.append(
        raw.MassSpecInputFragmentationPathway(
            raw.Histogram(width, 1000, np.zeros(1000)), 0
        )
    )
    return raw.MassSpecSubstanceTreeInput(
        -1,
        raw.Gas(2e-10, 4.8e-26, 1.4),
        [raw.MSSubstanceTreeCluster(cluster, density) for cluster in clusters],
        pathways,
        [raw.MSSubstanceTreeNode(i, [i]) for i in range(3)],
        [raw.MSSubstanceTreePathway(i, i + 1 if i < 2 else None) for i in range(3)],
        **({} if method is None else {"fragmentation_state_method": method}),
    )


def events(tree):
    instrument = raw.MassSpectrometer(
        np.array([[0, 300, 1e-12], [0, 300, 1e-12]]),
        0.0005,
        np.full(5, 0.001),
        np.arange(5.0) * 10,
        300,
        np.full(2, 1e-12),
        radius_pinhole=None,
    )
    stream = raw.MassSpecIterator(instrument, tree, 8, seed=42, logconf=(0, True))
    rows = []
    try:
        while True:
            try:
                batch = next(stream)
            except StopIteration:
                break
            columns = batch.columns
            rows.extend(dict(zip(columns, values)) for values in zip(*columns.values()))
    finally:
        stream.join_if_joinable()
    return sorted(
        rows, key=lambda row: (row["realization"], row["t"], row["event_kind"])
    )


def test_default_continuation_is_exactly_explicit_zero():
    implicit = make_tree()
    assert implicit.fragmentation_state_method == raw.FragmentationStateMethod.zero
    first = events(implicit)
    second = events(make_tree(raw.FragmentationStateMethod.zero))
    # Compare the motion columns; detail columns contain NaN for other events.
    keys = (
        "realization",
        "event_kind",
        "t",
        "vx",
        "vy",
        "vz",
        "ox",
        "oy",
        "oz",
        "rot_energy",
        "vibrational_energy",
    )
    assert [[row[k] for k in keys] for row in first] == [
        [row[k] for k in keys] for row in second
    ]
    after = [row for row in first if row["event_kind"] == 4]
    assert after
    assert all(row[k] == 0 for row in after for k in keys[3:])


@pytest.mark.parametrize("atomic", [False, True])
def test_phase_space_serial_continuation_and_lifetime(atomic):
    tree = make_tree(raw.FragmentationStateMethod.phase_space, atomic=atomic)
    gc.collect()  # Product descriptors are owned by native pathway inputs.
    rows = events(tree)
    previous = {}
    after_count = 0
    for row in rows:
        realization = row["realization"]
        if row["event_kind"] == 4:
            before = previous[realization]
            assert before["event_kind"] == 3
            assert row["particle_index"] == before["next_particle_index"]
            assert row["t"] == before["t"]
            assert row["rot_energy"] > 0
            assert 0 <= row["vibrational_energy"]
            assert row["rot_energy"] + row["vibrational_energy"] <= (
                before["rot_energy"] + before["vibrational_energy"]
            ) * (1 + 1e-12)
            assert np.isfinite(
                [row[k] for k in ("vx", "vy", "vz", "ox", "oy", "oz")]
            ).all()
            assert any(row[k] != 0 for k in ("vx", "vy", "vz"))
            after_count += 1
        previous[realization] = row
    assert after_count == 16  # Two successive breakups for each realization.


def test_phase_space_requires_products_but_zero_does_not():
    tree = make_tree()
    arguments = (
        tree.cluster_charge_sign,
        tree.gas,
        tree.cluster_payloads,
        [
            raw.MassSpecInputFragmentationPathway(p.rate_const, 0)
            for p in tree.pathway_payloads
        ],
        tree.tree_nodes,
        tree.tree_pathways,
    )
    raw.MassSpecSubstanceTreeInput(*arguments)
    with pytest.raises(raw.ApiTofArgumentError, match="product data"):
        raw.MassSpecSubstanceTreeInput(
            *arguments, raw.FragmentationStateMethod.phase_space
        )


def test_phase_space_rejects_mismatched_transition_mass():
    tree = make_tree()
    with pytest.raises(raw.ApiTofArgumentError, match="masses and daughter inertia"):
        raw.MassSpecSubstanceTreeInput(
            tree.cluster_charge_sign,
            tree.gas,
            [tree.cluster_payloads[1], *tree.cluster_payloads[1:]],
            tree.pathway_payloads,
            tree.tree_nodes,
            tree.tree_pathways,
            raw.FragmentationStateMethod.phase_space,
        )


@pytest.mark.parametrize("spelling", ["phase-space", "phase_space", "PHASE_SPACE"])
def test_fragmentation_config_normalization(spelling):
    config = import_raw_config({"simulation": {"fragmentation_state_method": spelling}})
    assert config == {"fragmentation_state_method": "phase-space"}
    assert (
        simulation_option(config, "fragmentation_state_method")
        == api.FragmentationStateMethod.phase_space
    )
    assert (
        simulation_option({}, "fragmentation_state_method")
        == api.FragmentationStateMethod.zero
    )
