create or replace view experiment_pathway_report as
select
    -- Experiment run info
    er.id as experiment_run_id,
    er.start_time,

    -- Config info
    conf.name as config_name,
    conf.config as config,

    -- Result/Failure info
    res.id as result_id,
    case when res.msg is not null then 'failure' else 'result' end as outcome_type,

    -- Pathway info
    p.id as pathway_id,

    -- Cluster info (the main cluster)
    c.id as cluster_id,
    c.common_name as cluster_common_name,
    c.atomic_mass as cluster_atomic_mass,
    c.electronic_energy as cluster_electronic_energy,
    c.rotational_temperatures as cluster_rotational_temperatures,
    c.vibrational_temperatures as cluster_vibrational_temperatures,

    -- Product 1 info
    p1.id as product1_id,
    p1.common_name as product1_common_name,
    p1.atomic_mass as product1_atomic_mass,
    p1.electronic_energy as product1_electronic_energy,
    p1.rotational_temperatures as product1_rotational_temperatures,
    p1.vibrational_temperatures as product1_vibrational_temperatures,

    -- Product 2 info
    p2.id as product2_id,
    p2.common_name as product2_common_name,
    p2.atomic_mass as product2_atomic_mass,
    p2.electronic_energy as product2_electronic_energy,
    p2.rotational_temperatures as product2_rotational_temperatures,
    p2.vibrational_temperatures as product2_vibrational_temperatures,

    -- Result/failure fields
    res.msg as failure_msg,
    res.loop_us,
    res.total_us,
    res.nwarnings,
    res.n_fragmented_total,
    res.n_escaped_total,
    res.ncoll_total,
    res.counter_collision_rejections

from experiment_run as er
left join (
    select * from single_pathway_experiment_result
    union by name
    (
        select
            multi_pathway_experiment_result.id as id,
            count as n_fragmented_total,
            * exclude (id, count)
        from multi_pathway_experiment_result
        inner join pathway_fragmentation
        on pathway_fragmentation.experiment_result_id = multi_pathway_experiment_result.id
    )
    union by name
    select * from experiment_failure
) as res on res.experiment_run_id = er.id
inner join experiment_config as conf on conf.id = er.experiment_config_id
inner join pathway p on p.id = res.pathway_id
inner join cluster c on c.id = p.cluster_id
inner join cluster p1 on p1.id = p.product1_id
inner join cluster p2 on p2.id = p.product2_id;


create or replace view experiment_cluster_report as
select
    -- Experiment run info
    er.id as experiment_run_id,
    er.start_time,

    -- Config info
    conf.name as config_name,
    conf.config as config,

    -- Cluster info (the main cluster)
    c.id as cluster_id,
    c.common_name as cluster_common_name,
    c.atomic_mass as cluster_atomic_mass,
    c.electronic_energy as cluster_electronic_energy,
    c.rotational_temperatures as cluster_rotational_temperatures,
    c.vibrational_temperatures as cluster_vibrational_temperatures,

    (
        select count(*) > 0
        from single_pathway_experiment_result
        where single_pathway_experiment_result.experiment_run_id = er.id
    ) as is_single_pathway

from experiment_run as er
left join (
    select experiment_run_id, cluster_id from multi_pathway_experiment_result
    union by name
    (
        select distinct experiment_run_id, pathway.cluster_id
        from single_pathway_experiment_result
        join pathway
        on single_pathway_experiment_result.pathway_id = pathway.id
    )
) as cluster_res on cluster_res.experiment_run_id = er.id
inner join experiment_config as conf on conf.id = er.experiment_config_id
inner join cluster c on c.id = cluster_res.cluster_id;


create or replace view experiment_summary as
select
    er.id as experiment_run_id,
    conf.name as config_name,
    er.start_time,
    (
        select count()
        from (
            select * from single_pathway_experiment_result
            union by name
            select * from multi_pathway_experiment_result
        ) as experiment_result
        where experiment_result.experiment_run_id = er.id
    ) as successes,
    (
        select count()
        from experiment_failure
        where experiment_failure.experiment_run_id = er.id
    ) as failures,
    (
        select count(*) > 0
        from single_pathway_experiment_result
        where single_pathway_experiment_result.experiment_run_id = er.id
    ) as is_single_pathway
from experiment_run er
join experiment_config conf on conf.id = er.experiment_config_id
