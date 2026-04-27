create or replace view event_report as
with
    event as (
        select realization_id, 'collision' as event_type, postime from collision_event
        union
        select realization_id, 'fragmentation' as event_type, postime from fragmentation_event
        union
        select realization_id, 'escape' as event_type, postime from escape_event
    ),
    pathway_experiment_result as (
        select * from single_pathway_experiment_result
        union by name
        select * from multi_pathway_experiment_result
    )
select
    pathway_experiment_result.id as experiment_result_id,
    cluster.common_name as parent_name,
    event_type,
    realization_id,
    unnest(postime)
from
    event
inner join
    realization on event.realization_id = realization.id
inner join
    pathway_experiment_result on pathway_experiment_result.id = realization.experiment_result_id
inner join
    cluster on cluster.id = pathway_experiment_result.cluster_id
order by
    parent_name;
