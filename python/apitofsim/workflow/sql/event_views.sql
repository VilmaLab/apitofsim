create or replace temp view event_details as
select
    concat_ws(':', e.result_id, e.realization_index, e.event_index) as id,
    concat_ws(':', e.result_id, e.realization_index) as realization_id,
    e.result_id, e.realization_index, e.event_index,
    ['init','collision','postcollision','fragmentation','postfragmentation','escape'][e.event_kind+1] as event_type,
    struct_pack(x:=x,y:=y,z:=z,t:=t) as postime,
    struct_pack(x:=vx,y:=vy,z:=vz) as velocity,
    struct_pack(x:=ox,y:=oy,z:=oz) as omega,
    e.x,e.y,e.z,e.t,e.rot_energy,e.vibrational_energy,e.particle_index,
    case when event_kind=1 then id end as collision_event_id,
    case when event_kind=1 then e.theta end as theta,
    case when event_kind=1 then e.u_norm end as u_norm,
    case when event_kind=1 then e.accepted::boolean end as accepted,
    case when event_kind=3 then id end as fragmentation_event_id,
    case when event_kind=3 then e.transition_index end as transition_index,
    case when event_kind=3 then e.next_particle_index end as next_particle_index,
    case when event_kind=3 then execution.pathway_ids[e.transition_index+1] end as pathway_id
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
