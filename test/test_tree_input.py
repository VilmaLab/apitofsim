"""Native tree views, topology column ownership and transition identity."""

import gc

import numpy as np
import pytest
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


def test_topology_export_rejects_payload_overflow():
    if np.dtype(np.uintp).itemsize == 8:
        with pytest.raises(ValueError, match="int64 range"):
            make_tree(nodes=[(2**63, [])], pathways=[]).topology_columns()
