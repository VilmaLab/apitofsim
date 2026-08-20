def draw_graph_node(builder, node, parent):
    node_name = f"treenode{node.index}"
    builder.node(
        node_name,
        "|".join(
            [
                f"index: {node.index}",
                f"cluster_charge_sign: {node.cluster_charge_sign}",
                f"m_ion: {node.m_ion}",
                f"R_cluster: {node.R_cluster}",
            ]
        ),
    )
    builder.edge(parent, node_name)
    for pathway_idx, (pathway, (product1, product2, count)) in enumerate(
        zip(node.pathways, node.pathway_products)
    ):
        pathway_name = f"{node_name}_{pathway_idx}"
        builder.node(
            pathway_name,
            "|".join([f"bonding_energy: {pathway.bonding_energy}", f"count: {count}"]),
        )
        builder.edge(node_name, pathway_name)
        draw_graph_node(builder, product1, pathway_name)
        draw_graph_node(builder, product2, pathway_name)


def mk_tree_input_graph(root):
    import graphviz

    builder = graphviz.Digraph(
        "inputtreerecords",
        filename="inputtreerecords.gv",
        node_attr={"shape": "record"},
        graph_attr={"rankdir": "LR"},
    )

    builder.node(
        "treeinput",
        "|".join(
            [
                f"gas.radius: {root.gas.radius}",
                f"gas.mass: {root.gas.mass}",
                f"gas.adiabatic_index: {root.gas.adiabatic_index}",
                f"count: {root.count}",
                f"pathway_count: {root.pathway_count}",
            ]
        ),
    )

    draw_graph_node(builder, root.root, "treeinput")

    return builder
