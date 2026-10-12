create or replace view event_details as
select
    event_info.*,
    unnest(event_info.postime),
    collision_event.id as collision_event_id,
    collision_event.theta,
    collision_event.u_norm,
    collision_event.accepted,
    fragmentation_event.id as fragmentation_event_id,
    fragmentation_event.pathway_id
from event_info
left join collision_event on collision_event.id = event_info.id
left join fragmentation_event on fragmentation_event.id = event_info.id;

create or replace view event_report as
select
    experiment_result.id as experiment_result_id,
    cluster.id as cluster_id,
    cluster.common_name as parent_name,
    event_details.event_type,
    event_details.realization_id,
    event_details.x,
    event_details.y,
    event_details.z,
    event_details.t,
    event_details.id as event_id,
    event_details.velocity,
    event_details.omega,
    event_details.rot_energy,
    event_details.vibrational_energy,
    event_details.particle_index,
    event_details.theta,
    event_details.u_norm,
    event_details.accepted,
    event_details.pathway_id
from
    event_details
inner join
    realization on event_details.realization_id = realization.id
inner join
    experiment_result on experiment_result.id = realization.experiment_result_id
inner join
    cluster on cluster.id = experiment_result.cluster_id
order by
    parent_name;
