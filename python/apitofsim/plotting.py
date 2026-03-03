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
        return db.db.execute(
            sql
            + f"""
        where
            experiment_run_id = ?
        order by
            {qual}parent_id
        """,
            (er_id,),
        ).fetchdf()
    else:
        return db.db.execute(
            sql
            + f"""
        where
            experiment_run_id = ? and
            {qual}parent_id = ?
        """,
            (er_id, cluster_id),
        ).fetchdf()


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


def plot_spectrogram(df, scale=None, max_x=None):
    try:
        import holoviews  # pyright: ignore[reportMissingImports]
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
    print(df)
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
    return spectrogram


def plot_spectrogram_to_file(outf, df, *args, **kwargs):
    try:
        import holoviews  # pyright: ignore[reportMissingImports]
    except ImportError:
        raise ImportError("Plotting requires holoviews and matplotlib; please install")

    holoviews.extension("matplotlib")  # type: ignore
    spectrogram = plot_spectrogram(outf, df, *args, **kwargs)
    spectrogram = spectrogram.opts(fig_inches=(6, 3), aspect=2)
    matplotlib_fig = holoviews.render(spectrogram)
    matplotlib_fig.savefig(outf, dpi=300)
