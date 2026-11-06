create or replace temp view event_details as
select
    {event_columns}
from _event_archive e
join event_execution execution on execution.result_id=e.result_id
where execution.recorded;

create or replace temp view event_info as select * from event_details;

create or replace temp view realization as
select distinct realization_id as id, result_id as experiment_result_id from event_details;

create or replace temp view event_report as
select execution.result_id as experiment_result_id,execution.run_id as experiment_run_id,
    execution.cluster_id,cluster.common_name as parent_name,execution.status,
    e.event_type,e.realization_id,e.realization_index,e.event_index,
    e.x,e.y,e.z,e.t,e.id as event_id,e.velocity,e.omega,
    e.rot_energy,e.vibrational_energy,e.particle_index,
    e.theta,e.u_norm,e.accepted,e.pathway_id
from event_details e join event_execution execution on execution.result_id=e.result_id
left join cluster on cluster.id=execution.cluster_id;
