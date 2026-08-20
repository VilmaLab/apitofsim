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


def make_survival_plot(cluster_names, values):
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
    fig.tight_layout()
    return fig
