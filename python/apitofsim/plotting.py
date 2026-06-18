def get_joint_survivals(db, er_id):
    import duckdb

    return dict(
        (
            db.db.table("experiment_cluster_report")
            .filter(
                (
                    duckdb.ColumnExpression("experiment_run_id")
                    == duckdb.ConstantExpression(er_id)
                )
            )
            .select("cluster_common_name", "survival_rate")
        ).fetchall()
    )


def make_survival_plot(outf, cluster_names, values):
    try:
        import matplotlib.pyplot as plt  # pyright: ignore[reportMissingImports]
    except ImportError:
        raise ImportError("Plotting requires holoviews and matplotlib; please install")
    import numpy as np

    # Bar positions
    x = np.arange(len(cluster_names))
    width = 0.2

    fig, ax = plt.subplots(figsize=(10, 6))

    # Create bars
    ax.bar(
        x,
        values,
        width,
        edgecolor="none",
    )

    # Customize axes
    ax.set_xlabel("Cluster", fontsize=12)
    ax.set_ylabel("Survival Probability", fontsize=12)
    ax.set_xticks(x)
    ax.tick_params(axis="x", labelrotation=90)
    ax.set_xticklabels(cluster_names)
    ax.set_yticks([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])

    # Add horizontal gridlines only
    ax.yaxis.grid(True, linestyle="-", alpha=0.7, color="gray")
    ax.xaxis.grid(False)
    ax.set_axisbelow(True)

    # Style spines
    ax.spines["left"].set_color("gray")
    ax.spines["right"].set_color("gray")
    ax.spines["top"].set_color("gray")
    ax.spines["bottom"].set_color("gray")

    # Legend in upper right
    fig.legend(loc="upper right", frameon=True, facecolor="white", edgecolor="gray")

    plt.tight_layout()
    plt.savefig(outf, dpi=150, facecolor=fig.get_facecolor(), bbox_inches="tight")


def _get_intensities_helper(db, sql, er_id, cluster_id=None, qual=""):
    if cluster_id is None:
        args = (
            sql
            + f"""
        where
            experiment_run_id = ?
        order by
            {qual}parent_id
        """,
            (er_id,),
        )
    else:
        args = (
            sql
            + f"""
        where
            experiment_run_id = ? and
            {qual}parent_id = ?
        """,
            (er_id, cluster_id),
        )
    return db.db.execute(*args).fetchdf()


def get_intensities_multipathway(db, er_id, cluster_id=None):
    return _get_intensities_helper(
        db,
        """
        with
            pathway_products as (
                select id as pathway_id, product1_id as product_id from pathway
                union
                select id as pathway_id, product2_id as product_id from pathway
            ),
            cluster_counts as (
                select
                    multi_pathway_experiment_result.experiment_run_id as experiment_run_id,
                    multi_pathway_experiment_result.cluster_id as parent_id,
                    multi_pathway_experiment_result.id as experiment_result_id,
                    pathway_products.product_id as product_id,
                    pathway_fragmentation.count as count
                from
                    multi_pathway_experiment_result
                inner join
                    pathway_fragmentation on multi_pathway_experiment_result.id = pathway_fragmentation.experiment_result_id
                inner join
                    pathway_products on pathway_products.pathway_id = pathway_fragmentation.pathway_id
                union
                select
                    multi_pathway_experiment_result.experiment_run_id as experiment_run_id,
                    multi_pathway_experiment_result.cluster_id as parent_id,
                    multi_pathway_experiment_result.id as experiment_result_id,
                    multi_pathway_experiment_result.cluster_id as product_id,
                    multi_pathway_experiment_result.n_escaped_total as count
                from
                    multi_pathway_experiment_result
            ),
            experiment_counts as (
                select
                    cluster_counts.parent_id as parent_id,
                    cluster_counts.experiment_result_id as experiment_result_id,
                    sum(cluster_counts.count) as count
                from
                    cluster_counts
                group by
                    parent_id,
                    experiment_result_id
            )
        select
            cluster_counts.experiment_run_id as experiment_run_id,
            cluster_counts.parent_id as parent_id,
            parent_cluster.common_name as parent_name,
            product_cluster.common_name as product_name,
            product_cluster.atomic_mass,
            cluster_counts.count / experiment_counts.count as relative_count,
            abs(relative_count * product_cluster.charge) as intensity
        from
            cluster_counts
        inner join
            cluster as parent_cluster
            on parent_cluster.id = cluster_counts.parent_id
        inner join
            cluster as product_cluster
            on product_cluster.id = cluster_counts.product_id
        inner join
            experiment_counts
            on experiment_counts.parent_id = cluster_counts.parent_id
            and experiment_counts.experiment_result_id = cluster_counts.experiment_result_id
        """,
        er_id,
        cluster_id=cluster_id,
        qual="cluster_counts.",
    )


def get_intensities_singlepathway(db, er_id, cluster_id=None):
    import pandas

    df = _get_intensities_helper(
        db,
        """
        select
            single_pathway_experiment_result.experiment_run_id as experiment_run_id,
            pathway_report.cluster_id as parent_id,
            pathway_report.*,
            single_pathway_experiment_result.n_fragmented_total / (single_pathway_experiment_result.n_fragmented_total + single_pathway_experiment_result.n_escaped_total) as fragmentation_prob,
            single_pathway_experiment_result.n_escaped_total / (single_pathway_experiment_result.n_fragmented_total + single_pathway_experiment_result.n_escaped_total) as survival_prob,
        from
            single_pathway_experiment_result
        inner join
            pathway_report on pathway_report.pathway_id = single_pathway_experiment_result.pathway_id
        """,
        er_id,
        cluster_id=cluster_id,
    )

    def check_charges(charged, uncharged, expected):
        if charged != expected:
            raise ValueError(
                f"Single-pathway requires product has same charge as parent; Got {expected} but got {charged}"
            )
        if uncharged != 0:
            raise ValueError(
                f"Single-pathway requires only one product charged; Expected uncharged product to have 0 charge but got {uncharged}"
            )

    new_df = {
        "experiment_run_id": [],
        "parent_id": [],
        "parent_name": [],
        "product_name": [],
        "atomic_mass": [],
        "intensity": [],
    }
    for (experiment_run_id, parent_id), group in df.groupby(
        ["experiment_run_id", "parent_id"]
    ):
        cluster_name = None
        cluster_atomic_mass = None
        product_names = []
        product_masses = []
        survival_probs = []
        fragmentation_probs = []
        for row in group.itertuples():
            if row.product1_charge != 0:
                check_charges(
                    row.product1_charge, row.product2_charge, row.cluster_charge
                )
                product_name = row.product1_common_name
                atomic_mass = row.product1_atomic_mass
            else:
                check_charges(
                    row.product2_charge, row.product1_charge, row.cluster_charge
                )
                product_name = row.product2_common_name
                atomic_mass = row.product2_atomic_mass
            cluster_name = row.cluster_common_name
            cluster_atomic_mass = row.cluster_atomic_mass
            product_names.append(product_name)
            product_masses.append(atomic_mass)
            survival_probs.append(row.survival_prob)
            fragmentation_probs.append(row.fragmentation_prob)
        probabilities = {name: 0.0 for name in (cluster_name, *product_names)}
        for combination in range(2 << len(survival_probs)):
            # Step 1. Find probability of this combination of survivals and fragmentations
            prob = 1.0
            for product_idx in range(len(survival_probs)):
                if combination & (1 << product_idx) > 0:
                    prob *= fragmentation_probs[product_idx]
                else:
                    prob *= survival_probs[product_idx]
            if combination == 0:
                # Special case: all survive, so we add this probability to the parent cluster
                probabilities[cluster_name] = prob
            else:
                # Step 2. Redistribute this probability across the products that are fragmented to in this combination
                denom = 0.0
                for product_idx in range(len(survival_probs)):
                    if combination & (1 << product_idx) > 0:
                        denom += fragmentation_probs[product_idx]
                if denom == 0.0:
                    continue
                for product_idx, name in enumerate(product_names):
                    probabilities[name] += (
                        fragmentation_probs[product_idx] / denom * prob
                    )
        for product_name, product_mass in zip(
            (*product_names, cluster_name), (*product_masses, cluster_atomic_mass)
        ):
            new_df["experiment_run_id"].append(experiment_run_id)
            new_df["parent_id"].append(parent_id)
            new_df["parent_name"].append(cluster_name)
            new_df["product_name"].append(product_name)
            new_df["atomic_mass"].append(product_mass)
            new_df["intensity"].append(probabilities[product_name])
    return pandas.DataFrame(new_df)


def get_intensities(db, *, experiment_id, cluster_id=None, is_single_pathway=False):
    if is_single_pathway:
        return get_intensities_singlepathway(db, experiment_id, cluster_id)
    else:
        return get_intensities_multipathway(db, experiment_id, cluster_id)


def rotmat(deg):
    import numpy as np

    theta = np.radians(deg)
    c, s = np.cos(theta), np.sin(theta)
    return np.array(((c, -s), (s, c)))


ccw90_mat = rotmat(45)
cw90_mat = rotmat(-45)


def draw_boxes(boxes_np, lines_np, outfn):
    # This function is not used in the final code, but is useful for debugging the relayout_labels(...)
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_aspect("equal")
    plt.title("Rectangles")

    xmin = 0
    xmax = 1
    ymin = 0
    ymax = 1
    for box in boxes_np:
        x0, y0, x1, y1 = box
        if x0 < xmin:
            xmin = x0
        if y0 < ymin:
            ymin = y0
        if x1 > xmax:
            xmax = x1
        if y1 > ymax:
            ymax = y1
        width = x1 - x0
        height = y1 - y0
        rect = Rectangle((x0, y0), width, height, color="blue", fc="none", lw=2)
        ax.add_patch(rect)

    for line in lines_np:
        ax.plot(line[[0, 2]], line[[1, 3]], color="red", lw=1)

    plt.xlim((xmin, xmax))
    plt.ylim((ymin, ymax))
    plt.xlabel("x - axis")
    plt.ylabel("y - axis")

    fig.savefig(outfn, dpi=300)


def relayout_labels(spectrogram, labels, fig_inches, aspect):
    import holoviews
    import numpy as np

    bounding_boxes = []
    xlims = None
    ylims = None
    aspect_ratio = None
    collected_bboxes = False
    box_xs = []

    def collect_normal_bboxes(plot, element):
        nonlocal collected_bboxes
        if collected_bboxes:
            return
        nonlocal xlims, ylims, aspect_ratio
        from matplotlib.text import Text
        from textalloc import _get_renderer

        fig = plot.state
        ax = plot.handles["axis"]
        renderer = _get_renderer(fig)
        artists = plot.handles["artist"]
        xlims = ax.get_xlim()
        ylims = ax.get_ylim()
        aspect_ratio = fig.get_size_inches()[0] / fig.get_size_inches()[1]
        idx = 0
        for artist in artists:
            assert isinstance(artist, Text)
            artist.set_verticalalignment("bottom")
            artist.set_horizontalalignment("left")
            box = artist.get_window_extent(renderer=renderer)
            box_xs.append(-box.x0)
            pt = np.array((box.x0, box.y0))[:, np.newaxis]
            ptrot = np.dot(cw90_mat, pt)
            bounding_boxes.append(
                np.array(
                    (
                        ptrot[0, 0],
                        ptrot[1, 0],
                        ptrot[0, 0] + box.width,
                        ptrot[1, 0] + box.height,
                    )
                )
            )
            idx += 1
        collected_bboxes = True

    collected_spikes = False
    spikes = []

    def collect_spikes(plot, element):
        nonlocal collected_spikes, spikes
        if collected_spikes:
            return
        segs = plot.handles["artist"].get_segments()
        transform = plot.handles["artist"].get_transform()
        for seg in segs:
            if (seg[0] == seg[1]).all():
                continue
            start = transform.transform(seg[0])
            end = transform.transform(seg[1])
            segrot1 = np.dot(cw90_mat, start)
            segrot2 = np.dot(cw90_mat, end)
            spikes.append(np.concatenate((segrot1, segrot2)))
        collected_spikes = True

    spectrogram_draft = spectrogram.clone()
    spectrogram_draft.opts(hooks=[collect_spikes])
    spectrogram_draft *= labels.opts(
        holoviews.opts.Labels(color="black", size=10, hooks=[collect_normal_bboxes])
    )
    spectrogram_draft = spectrogram_draft.opts(fig_inches=fig_inches, aspect=aspect)
    holoviews.render(spectrogram_draft)

    bounding_boxes_np = np.vstack(bounding_boxes)

    assert isinstance(spikes, list)

    spikes_np = np.vstack(spikes)
    # spikes_np = np.dot(cw90_mat, spikes_np)

    # draw_boxes(bounding_boxes_np, spikes_np, "orig_boxes.png")

    from textalloc.overlap_functions import (
        non_overlapping_with_boxes,
        non_overlapping_with_lines,
    )

    done = []
    for idx in np.argsort(box_xs):
        new_box = bounding_boxes_np[idx, :]
        incrs = np.concatenate((np.array([0, 1]), np.arange(5, 1000, 5)))[:, np.newaxis]
        new_box_cands = new_box[np.newaxis, :] + np.hstack([incrs, incrs, incrs, incrs])
        prev_bboxes = bounding_boxes_np[done]
        done.append(idx)
        acceptable = np.bitwise_and.reduce(
            (
                non_overlapping_with_boxes(prev_bboxes, new_box_cands, 0.01, 0.01),
                non_overlapping_with_lines(spikes_np, new_box_cands, 0.01, 0.01),
            )
        )
        nz = np.nonzero(acceptable)[0]
        if len(nz) == 0:
            continue  # Just accept the original position
        bounding_boxes_np[idx] = new_box_cands[nz[0]]

    # draw_boxes(bounding_boxes_np, spikes_np, "new_boxes.png")

    def apply_moved_labels(plot, element):
        from matplotlib.collections import LineCollection
        from matplotlib.text import Text

        idx = 0
        artists = plot.handles["artist"]
        fig = plot.state
        lines = []
        for artist in artists:
            if isinstance(artist, Text):
                artist.set_verticalalignment("bottom")
                artist.set_horizontalalignment("left")
                data_x, data_y = artist.get_position()
                x, y, _, _ = bounding_boxes_np[idx]
                ptrot = np.array((x, y))[:, np.newaxis]
                pt = np.dot(ccw90_mat, ptrot)
                x, y = pt[0, 0], pt[1, 0]
                x, y = artist.get_transform().inverted().transform((x, y))
                artist.set_position((x, y))
                bump_x, bump_y = artist.get_transform().inverted().transform(
                    (5, 5)
                ) - artist.get_transform().inverted().transform((0, 0))
                lines.append(
                    np.array(((data_x, data_y), (data_x, y), (x + bump_x, y + bump_y)))
                )
                idx += 1
        line_collection = LineCollection(
            lines, linewidths=0.5, colors="gray", linestyles="dashed", zorder=0
        )
        line_collection.set_clip_on(False)
        fig.gca().add_collection(line_collection)

    return apply_moved_labels


def plot_spectrogram(
    df, *, scale=None, max_x=None, label=False, label_threshold=0.1, fig_inches, aspect
):
    try:
        import holoviews  # pyright: ignore[reportMissingImports]

        holoviews.extension("matplotlib")
    except ImportError:
        raise ImportError("Plotting requires holoviews and matplotlib; please install")

    if scale == "max":
        df["intensity"] /= df["intensity"].max()
    elif scale == "sum":
        df["intensity"] /= df["intensity"].sum()
    elif scale is not None:
        raise ValueError(
            f"Unsupported scale {scale}; expected one of 'max', 'sum', or None"
        )
    if max_x is not None:
        x_dim = holoviews.Dimension("m/z", soft_range=(0, max_x))
    else:
        x_dim = holoviews.Dimension(
            "m/z", soft_range=(0, df["atomic_mass"].max() * 1.1)
        )
    y_dim = holoviews.Dimension("Intensity", soft_range=(0, 1.05))
    spectrogram = holoviews.Spikes(
        (df["atomic_mass"], df["intensity"]),
        x_dim,
        y_dim,
    )

    def render():
        return holoviews.render(spectrogram.opts(fig_inches=fig_inches, aspect=aspect))

    if label == "none":
        return render()
    elif label == "threshold":
        labels_df = df[df["intensity"] >= label_threshold]
    elif label == "all":
        labels_df = df
    elif label == "nonzero":
        labels_df = df[df["intensity"] > 0]
    else:
        raise ValueError(
            "Unknown value for `label`: expected one of 'none', 'threshold', 'all', or 'nonzero'"
        )
    labels = holoviews.Labels(
        {
            "x": labels_df["atomic_mass"].to_numpy(copy=True),
            "y": labels_df["intensity"].to_numpy(copy=True),
            "text": labels_df["product_name"].to_list(),
        },
        ["x", "y"],
        "text",
    )

    spectrogram *= labels.opts(
        holoviews.opts.Labels(
            color="black",
            size=10,
            rotation=45,
            hooks=[relayout_labels(spectrogram, labels, fig_inches, aspect)],
        )
    )
    return render()


def plot_spectrogram_to_file(outf, df, *args, **kwargs):
    try:
        import holoviews  # pyright: ignore[reportMissingImports]
    except ImportError:
        raise ImportError("Plotting requires holoviews and matplotlib; please install")

    holoviews.extension("matplotlib")  # type: ignore
    spectrogram = plot_spectrogram(df, *args, **kwargs, fig_inches=(6, 3), aspect=2)
    spectrogram.savefig(outf, dpi=300, bbox_inches="tight")


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
