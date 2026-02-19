from apitofsim.db import ExperimentDatabase

import duckdb
import holoviews
import sys
import pandas as pd

holoviews.extension("matplotlib")  # type: ignore

db = ExperimentDatabase(sys.argv[1])
df = db.db.execute(
    """
    select
        multi_pathway_experiment_result.id,
        experiment_config.name as config_name,
        experiment_run.start_time as start_time,
        cluster.common_name as cluster_name,
        multi_pathway_experiment_result.n_escaped_total as escaped,
    from multi_pathway_experiment_result
    inner join
        cluster on multi_pathway_experiment_result.cluster_id = cluster.id
    inner join
        experiment_run on multi_pathway_experiment_result.experiment_run_id = experiment_run.id
    inner join
        experiment_config on experiment_run.experiment_config_id = experiment_config.id
    order by
        multi_pathway_experiment_result.id
    """
).fetchdf()
print(df.to_string(index=False))
er_id = None
while 1:
    er_id = input("Choose an result to output > ")
    try:
        er_id = int(er_id)
    except ValueError:
        continue
    else:
        break
print()

df = db.db.execute(
    """
    with
        pathway_products as (
            select id as pathway_id, product1_id as product_id from pathway
            union
            select id as pathway_id, product2_id as product_id from pathway
        ),
        cluster_counts as (
            select
                multi_pathway_experiment_result.id as experiment_result_id,
                pathway_products.product_id as cluster_id,
                pathway_fragmentation.count as count
            from 
                multi_pathway_experiment_result
            inner join
                pathway_fragmentation on multi_pathway_experiment_result.id = pathway_fragmentation.experiment_result_id
            inner join
                pathway_products on pathway_products.pathway_id = pathway_fragmentation.pathway_id 
            union
            select
                multi_pathway_experiment_result.id as experiment_result_id,
                multi_pathway_experiment_result.cluster_id as cluster_id,
                multi_pathway_experiment_result.n_escaped_total as count
            from
                multi_pathway_experiment_result
        ),
        experiment_counts as (
            select
                pathway_fragmentation.experiment_result_id as experiment_result_id,
                sum(pathway_fragmentation.count) as count
            from
                pathway_fragmentation
            group by 
                pathway_fragmentation.experiment_result_id 
        )
    select
        cluster.common_name as cluster_name,
        cluster.atomic_mass,
        cluster_counts.count / experiment_counts.count as intensity
    from 
        cluster_counts
    inner join
        cluster on cluster.id = cluster_counts.cluster_id
    inner join
        experiment_counts on experiment_counts.experiment_result_id = cluster_counts.experiment_result_id
    where
        cluster_counts.experiment_result_id = ?
    """,
    (er_id,),
).fetchdf()

spectrogram = holoviews.Spikes(
    (df["atomic_mass"], df["intensity"]),
    holoviews.Dimension("m/z", soft_range=(0, 1)),
    "Intensity",
).opts(fig_inches=(6, 3), aspect=2)
matplotlib_fig = holoviews.render(spectrogram)
matplotlib_fig.savefig("spectrogram.png", dpi=300)
