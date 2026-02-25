def get_joint_survivals(db, er_id):
    from functools import reduce
    from operator import mul

    import duckdb

    joint_survivals = {}

    for cluster in db.clusters_df(parents_only=True).itertuples():
        print("#", cluster.common_name)
        df = (
            db.db.table("experiment_report")
            .filter(
                (
                    duckdb.ColumnExpression("experiment_run_id")
                    == duckdb.ConstantExpression(er_id)
                )
                & (
                    duckdb.ColumnExpression("cluster_id")
                    == duckdb.ConstantExpression(cluster.id)
                )
            )
            .select(
                duckdb.SQLExpression(
                    "format('{} -> {} + {}', cluster_common_name, product1_common_name, product2_common_name)"
                ).alias("pathway_name"),
                *(
                    duckdb.ColumnExpression(col)
                    for col in [
                        "outcome_type",
                        "failure_msg",
                        "nwarnings",
                        "n_fragmented_total",
                        "n_escaped_total",
                        "ncoll_total",
                        "counter_collision_rejections",
                    ]
                ),
                duckdb.SQLExpression(
                    "n_escaped_total / (n_escaped_total + n_fragmented_total)"
                ).alias("survival_rate"),
            )
        ).fetchdf()
        print(df)
        print()
        if len(df) == 0:
            print("No results for", cluster.common_name)
            continue
        # This seems a bit naughty
        survival_rate = df["survival_rate"][df["survival_rate"] > 0]
        joint_survivals[cluster.common_name] = reduce(mul, survival_rate, 1.0)
    return joint_survivals


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


def get_intensities_multipathway(db, er_id, cluster_id=None):
    intensities_sql = """
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
    """
    if cluster_id is None:
        return db.db.execute(
            intensities_sql
            + """
        where
            cluster_counts.experiment_run_id = ?
        order by
            cluster_counts.parent_id
        """,
            (er_id,),
        ).fetchdf()
    else:
        return db.db.execute(
            intensities_sql
            + """
        where
            cluster_counts.experiment_run_id = ? and
            cluster_counts.parent_id = ?
        """,
            (er_id, cluster_id),
        ).fetchdf()


def plot_spectrogram(outf, df):
    try:
        import holoviews  # pyright: ignore[reportMissingImports]
    except ImportError:
        raise ImportError("Plotting requires holoviews and matplotlib; please install")

    holoviews.extension("matplotlib")  # type: ignore

    spectrogram = holoviews.Spikes(
        (df["atomic_mass"], df["intensity"]),
        holoviews.Dimension("m/z", soft_range=(0, 1)),
        "Intensity",
    ).opts(fig_inches=(6, 3), aspect=2)
    matplotlib_fig = holoviews.render(spectrogram)
    matplotlib_fig.savefig(outf, dpi=300)
