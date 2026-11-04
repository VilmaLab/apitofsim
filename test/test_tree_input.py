"""Native tree views, topology column ownership and transition identity."""

import gc
from collections import Counter

import numpy as np
import pytest
from apitofsim import api
from apitofsim import apitofsimraw as raw


def make_tree(nodes=None, pathways=None):
    density = raw.Histogram(np.arange(4.0), np.ones(4))
    return raw.MassSpecSubstanceTreeInput(
        -1,
        raw.Gas(1e-10, 1e-26, 1.4),
        [raw.MSSubstanceTreeCluster(1e-25, 1e-10, density)],
        [raw.MassSpecInputFragmentationPathway(density, 1.0)],
        [
            raw.MSSubstanceTreeNode(p, indices)
            for p, indices in (
                [(0, [2]), (0, [0]), (0, [1])] if nodes is None else nodes
            )
        ],
        [
            raw.MSSubstanceTreePathway(p, dst)
            for p, dst in (
                [(0, 2), (0, None), (0, 1)] if pathways is None else pathways
            )
        ],
    )


def test_read_only_vector_views():
    tree = make_tree()
    views = [
        tree.tree_nodes,
        tree.tree_pathways,
        tree.cluster_payloads,
        tree.pathway_payloads,
        tree.tree_nodes[0].pathway_indices,
    ]
    for view in views:
        assert not isinstance(view, list)
        assert len(list(view)) == len(view)
        assert len(view[::-1]) == len(view)
        assert len(view[:0]) == 0
        with pytest.raises(IndexError):
            view[len(view)]
        with pytest.raises(IndexError):
            view[-len(view) - 1]
        with pytest.raises(TypeError):
            view[0] = view[0]
        with pytest.raises(TypeError):
            del view[0]
        assert not hasattr(view, "append")
        assert not hasattr(view, "clear")
    assert tree.tree_nodes[0] is tree.tree_nodes[0]
    assert tree.tree_pathways[-1] is tree.tree_pathways[2]
    assert list(tree.tree_nodes[0].pathway_indices) == [2]
    assert [n.payload_idx for n in tree.tree_nodes[::-1]] == [0, 0, 0]
    # Returned members also expose no setters.
    with pytest.raises(AttributeError):
        tree.tree_nodes[0].payload_idx = 1


@pytest.mark.parametrize(
    "access",
    [
        lambda t: t.tree_nodes,
        lambda t: t.tree_nodes[0],
        lambda t: t.tree_nodes[0].pathway_indices,
        lambda t: iter(t.tree_nodes),
        lambda t: t.tree_pathways[2],
        lambda t: t.cluster_payloads[0],
        lambda t: t.pathway_payloads[0],
    ],
)
def test_view_owner_lifetime(access):
    tree = make_tree()
    item = access(tree)
    del tree
    gc.collect()
    # Exercise the complete chain even when intermediate views were temporary.
    if isinstance(item, raw.TreeNodeView):
        assert list(item[0].pathway_indices) == [2]
    elif isinstance(item, raw.MSSubstanceTreeNode):
        assert list(item.pathway_indices) == [2]
    elif isinstance(item, raw.IndexView):
        assert list(item) == [2]
    elif isinstance(item, raw.MSSubstanceTreePathway):
        assert item.product_idx == 1
    elif isinstance(item, raw.MSSubstanceTreeCluster):
        assert item.m_ion == 1e-25
    elif isinstance(item, raw.MassSpecInputFragmentationPathway):
        assert item.bonding_energy == raw.consts.boltzmann
    else:
        assert list(next(item).pathway_indices) == [2]


def test_topology_columns_are_owned_and_indexed_by_native_transition():
    tree = make_tree()
    columns = tree.topology_columns()
    del tree
    gc.collect()
    expected = {
        "node_payload": [0, 0, 0],
        "transition_source": [1, 2, 0],
        "transition_payload": [0, 0, 0],
        "transition_destination": [2, -1, 1],
    }
    assert columns.keys() == expected.keys()
    for name, values in expected.items():
        assert columns[name].dtype == np.int64
        np.testing.assert_array_equal(columns[name], values)
    # Snapshots may be modified without changing the native input.
    tree = make_tree()
    columns = tree.topology_columns()
    columns["node_payload"][0] = 99
    assert tree.tree_nodes[0].payload_idx == 0


def test_empty_topology_columns():
    tree = make_tree(nodes=[], pathways=[])
    assert all(len(v) == 0 for v in tree.topology_columns().values())
    assert list(tree.tree_nodes) == []


@pytest.mark.parametrize(
    "nodes,pathways",
    [
        ([(0, [1])], [(0, None)]),
        ([(0, [0, 0])], [(0, None)]),
        ([(0, [])], [(0, None)]),
        ([(0, [0])], [(0, 1)]),
    ],
)
def test_topology_export_rejects_invalid_transition_identity(nodes, pathways):
    with pytest.raises(ValueError, match="Tree transition"):
        make_tree(nodes=nodes, pathways=pathways).topology_columns()


def test_constructor_accepts_native_views():
    source = make_tree()
    clone = raw.MassSpecSubstanceTreeInput(
        source.cluster_charge_sign,
        source.gas,
        source.cluster_payloads,
        source.pathway_payloads,
        source.tree_nodes,
        source.tree_pathways,
    )
    del source
    gc.collect()
    assert list(clone.tree_nodes[0].pathway_indices) == [2]
    assert clone.topology_columns()["transition_source"].tolist() == [1, 2, 0]
    np.testing.assert_array_equal(
        clone.cluster_payloads[0].density_cluster.y, np.ones(4)
    )
    np.testing.assert_array_equal(
        clone.pathway_payloads[0].rate_const.x, np.arange(4.0)
    )


def test_topology_export_rejects_payload_overflow():
    if np.dtype(np.uintp).itemsize == 8:
        with pytest.raises(ValueError, match="int64 range"):
            make_tree(nodes=[(2**63, [])], pathways=[]).topology_columns()


@pytest.mark.parametrize("mode", list(api.VibEnergyMode))
def test_tree_preparation_reuses_inputs_and_preserves_meshes(
    prepared_case, monkeypatch, mode
):
    db, runner, config, _ = prepared_case
    config["vib_energy_sampler"] = mode
    _, clusters, names, pathways, rates, densities = runner._prepare_from_config(
        config, cached_densityandrate=(1, 2), show_progress=False
    )
    params_calls = []
    conversions = Counter()
    get_params = db.get_histogram_params
    into_cpp = api.ClusterData.into_cpp

    def params(histogram_id):
        params_calls.append(histogram_id)
        return get_params(histogram_id)

    def convert(cluster):
        conversions[id(cluster)] += 1
        return into_cpp(cluster)

    with monkeypatch.context() as patch:
        patch.setattr(db, "get_histogram_params", params)
        patch.setattr(api.ClusterData, "into_cpp", convert)
        trees = list(
            runner._prepare_cluster_tree(
                config, clusters, names, pathways, rates, densities
            )
        )
    assert params_calls == [1, 2]
    assert conversions == Counter({id(cluster): 1 for cluster in clusters.values()})
    for cluster_ids, pathway_ids, tree, _ in trees:
        for cluster_id, index in cluster_ids.items():
            expected = api.scaled_density(
                api.Histogram.from_mesh(
                    api.Q_(100, "K"), api.Q_(100000, "K"), densities[cluster_id]
                )
            )
            actual = tree.cluster_payloads[index].density_cluster
            np.testing.assert_array_equal(actual.x, expected.x.magnitude)
            np.testing.assert_array_equal(actual.y, expected.y)
        for pathway_id, index in pathway_ids.items():
            parent, p1, p2 = pathways[pathway_id]
            expected = api.MassSpecInputFragmentationPathway(
                clusters[parent],
                clusters[p1],
                clusters[p2],
                api.scaled_rate_const(
                    api.Histogram.from_mesh(
                        api.Q_(50, "K"), api.Q_(100000, "K"), rates[pathway_id]
                    )
                ),
            )
            actual = tree.pathway_payloads[index]
            np.testing.assert_array_equal(actual.rate_const.x, expected.rate_const.x)
            np.testing.assert_array_equal(actual.rate_const.y, expected.rate_const.y)
            assert actual.bonding_energy == expected.bonding_energy

    # A new preparation must observe both changed mesh selection and mutable
    # cluster data; previously returned native inputs must retain their values.
    old_mass = trees[0][2].cluster_payloads[0].m_ion
    clusters[1].mass *= 2
    runner.preparer.dos_histogram_id = 3
    runner.preparer.rate_histogram_id = 1
    runner.preparer.vib_indexes = {}
    new_densities = {key: np.ones(4) for key in densities}
    new_rates = {key: np.ones(1000) for key in rates}
    fresh = list(
        runner._prepare_cluster_tree(
            config, clusters, names, pathways, new_rates, new_densities
        )
    )
    assert trees[0][2].cluster_payloads[0].m_ion == old_mass
    for cluster_ids, _, tree, _ in fresh:
        if 1 in cluster_ids:
            assert tree.cluster_payloads[cluster_ids[1]].m_ion == 2 * old_mass
        assert len(tree.cluster_payloads[0].density_cluster.x) == 4
        assert len(tree.pathway_payloads[0].rate_const.x) == 1000


@pytest.mark.parametrize("rate", [False, True])
def test_tree_preparation_checks_every_histogram_length(prepared_case, rate):
    _, runner, config, _ = prepared_case
    _, clusters, names, pathways, rates, densities = runner._prepare_from_config(
        config, cached_densityandrate=(1, 2), show_progress=False
    )
    # Corrupt a later payload, after the grid has already been cached.
    if rate:
        rates[3] = np.ones(2)
    else:
        densities[2] = np.ones(2)
    with pytest.raises(ValueError, match="histogram configuration"):
        list(
            runner._prepare_cluster_tree(
                config, clusters, names, pathways, rates, densities
            )
        )
