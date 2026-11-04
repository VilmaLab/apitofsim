"""Opt-in continuation, including native ownership and recorded after-states."""

import gc

import numpy as np
import pytest
from apitofsim import api
from apitofsim import apitofsimraw as raw
from apitofsim.config import import_raw_config, simulation_option


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
