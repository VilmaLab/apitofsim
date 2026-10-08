create or replace view event_report as
with
    pathway_experiment_result as (
        select * from single_pathway_experiment_result
        union by name
        select * from multi_pathway_experiment_result
    )
select
    pathway_experiment_result.id as experiment_result_id,
    cluster.id as cluster_id,
    cluster.common_name as parent_name,
    event_info.event_type,
    event_info.realization_id,
    unnest(event_info.postime),
    event_info.id as event_id,
    event_info.velocity,
    event_info.omega,
    event_info.rot_energy,
    event_info.vibrational_energy,
    event_info.particle_index,
    collision_event.theta,
    collision_event.u_norm,
    collision_event.accepted,
    fragmentation_event.pathway_id
from
    event_info
inner join
    realization on event_info.realization_id = realization.id
inner join
    pathway_experiment_result on pathway_experiment_result.id = realization.experiment_result_id
inner join
    cluster on cluster.id = pathway_experiment_result.cluster_id
left join
    collision_event on collision_event.id = event_info.id
left join
    fragmentation_event on fragmentation_event.id = event_info.id
order by
    parent_name;
