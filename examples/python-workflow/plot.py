import sys
from functools import reduce
from operator import mul
from pprint import pprint

import duckdb
import matplotlib.pyplot as plt
from apitofsim.workflow import ExperimentDatabase

db = ExperimentDatabase(sys.argv[1])
print(db.experiment_summary_df().to_string(index=False))
er_id = None
while 1:
    er_id = input("Choose an experiment to output > ")
    try:
        er_id = int(er_id)
    except ValueError:
        continue
    else:
        break
print()

name_map = {
    "1B1A": "1S1B",
    "1B2A": "2S1B",
    "1B2A1D": "1D2S1B",
    "1B3A1D": "1D3S1B",
    "1B1A2D": "2D1S1B",
    "1B2A2D": "2D2S1B",
    "1B3A2D": "2D3S1B",
    "1B3A3D": "3D3S1B",
    "1B4A4D": "4D4S1B",
}

joint_survivals = {}

for cluster in db.clusters_df(parents_only=True).itertuples():
    print("#", name_map[cluster.common_name])
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
    joint_survivals[name_map[cluster.common_name]] = reduce(mul, survival_rate, 1.0)

pprint(joint_survivals)


def make_plot(outf, new_model):
    import numpy as np

    # Sample data (replace with your own)
    clusters = list(name_map.values())
    experiment = [0.965, 0.437, 0.316, 0.185, 0.366, 0.313, 0.206, 0.696, 0.675]
    old_model = [0.746, 0.419, 0.229, 0.182, 0.267, 0.211, 0.857, 0.666, 0.996]

    # Set up the figure with a light gray background
    fig, ax = plt.subplots(figsize=(10, 6))
    # fig.patch.set_facecolor("#e8e8e8")
    # ax.set_facecolor("#e8e8e8")

    # Bar positions
    x = np.arange(len(clusters))
    width = 0.2

    # Colors matching the image
    purple = "#6B2D7B"  # Dark purple/plum
    lime_green = "#9ACD32"  # Yellow-green/lime
    cyan = "#4BACC6"
    # orange = "#F79646"

    # Create bars
    ax.bar(
        x - width, experiment, width, label="Experiment", color=purple, edgecolor="none"
    )
    ax.bar(
        x, old_model, width, label="Original code", color=lime_green, edgecolor="none"
    )
    ax.bar(
        x + width,
        new_model,
        width,
        label="New code",
        color=cyan,
        edgecolor="none",
    )

    # Customize axes
    ax.set_xlabel("Cluster", fontsize=12)
    ax.set_ylabel("Survival Probability", fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(clusters)
    # ax.set_ylim(0, 1.05)
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

    # Add annotation box in lower right (optional - like the S, B, D legend)
    textstr = "S: Sulfuric Acid\nB: Bisulfate Ion\nD: Dimethylamine"
    props = dict(boxstyle="square", facecolor="white", edgecolor="gray", alpha=0.9)
    ax.text(
        1.02,
        0.02,
        textstr,
        transform=ax.transAxes,
        fontsize=9,
        verticalalignment="bottom",
        bbox=props,
    )

    plt.tight_layout()
    plt.savefig(outf, dpi=150, facecolor=fig.get_facecolor(), bbox_inches="tight")


new_model = [joint_survivals[n] for n in name_map.values()]
make_plot(sys.argv[2], new_model)
