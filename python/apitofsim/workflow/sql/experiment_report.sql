-- Common aggregate shape for successful single-cluster and tree executions.
create or replace view multi_cluster_result as
select * from multi_pathway_experiment_result
union all
select * exclude (failure_id) from cluster_tree_experiment_result
where total_us is not null;

create or replace view experiment_result as
select
    res.id,
    res.experiment_run_id,
    pathway.cluster_id,
    res.pathway_id,
    true as is_single_pathway
from single_pathway_experiment_result as res
inner join pathway on pathway.id = res.pathway_id
union all
select
    id,
    experiment_run_id,
    cluster_id,
    null as pathway_id,
    false as is_single_pathway
from multi_pathway_experiment_result
union all
select id, experiment_run_id, cluster_id, null as pathway_id, false as is_single_pathway
from cluster_tree_experiment_result;

create or replace view experiment_pathway_report as
select
    -- Experiment run info
    er.id as experiment_run_id,
    er.start_time,
    er.simulation_mode,

    -- Config info
    conf.name as config_name,
    conf.config as config,

    -- Result/Failure info
    res.id as result_id,
    case when res.msg is not null then 'failure' else 'result' end as outcome_type,

    -- Pathway info
    p.id as pathway_id,
    p.cluster_id as pathway_cluster_id,
    pc.common_name as pathway_cluster_common_name,

    -- Cluster info (the main cluster)
    c.id as cluster_id,
    c.common_name as cluster_common_name,
    c.atomic_mass as cluster_atomic_mass,
    c.charge as cluster_charge,
    c.electronic_energy as cluster_electronic_energy,
    c.rotational_temperatures as cluster_rotational_temperatures,
    c.vibrational_temperatures as cluster_vibrational_temperatures,

    -- Product 1 info
    p1.id as product1_id,
    p1.common_name as product1_common_name,
    p1.atomic_mass as product1_atomic_mass,
    p1.charge as product1_charge,
    p1.electronic_energy as product1_electronic_energy,
    p1.rotational_temperatures as product1_rotational_temperatures,
    p1.vibrational_temperatures as product1_vibrational_temperatures,

    -- Product 2 info
    p2.id as product2_id,
    p2.common_name as product2_common_name,
    p2.atomic_mass as product2_atomic_mass,
    p2.charge as product2_charge,
    p2.electronic_energy as product2_electronic_energy,
    p2.rotational_temperatures as product2_rotational_temperatures,
    p2.vibrational_temperatures as product2_vibrational_temperatures,

    -- Result/failure fields
    res.msg as failure_msg,
    res.loop_us,
    res.total_us,
    res.nwarnings,
    res.n_escaped_pathway,
    -- First fragmentation for SINGLE_CLUSTER; last fragmentation for CLUSTER_TREE.
    res.n_fragmented_pathway,
    res.n_escaped_cluster,
    res.ncoll_total,
    res.counter_collision_rejections

from experiment_run as er
left join (
    (
        select
            n_escaped_total as n_escaped_pathway,
            n_fragmented_total as n_fragmented_pathway,
            * exclude (n_escaped_total, n_fragmented_total)
        from single_pathway_experiment_result
    )
    union by name
    (
        select
            multi_cluster_result.id as id,
            count as n_fragmented_pathway,
            n_escaped_total as n_escaped_cluster,
            * exclude (id, count, n_escaped_total)
        from multi_cluster_result
        left join pathway_fragmentation
        on pathway_fragmentation.experiment_result_id = multi_cluster_result.id
    )
    union by name
    select * from experiment_failure
) as res on res.experiment_run_id = er.id
inner join experiment_config as conf on conf.id = er.experiment_config_id
left join pathway p on p.id = res.pathway_id
-- Serial pathways may belong to a descendant of the original result cluster.
inner join cluster c on c.id = coalesce(res.cluster_id, p.cluster_id)
left join cluster pc on pc.id = p.cluster_id
left join cluster p1 on p1.id = p.product1_id
left join cluster p2 on p2.id = p.product2_id

order by
    er.start_time,
    er.id,
    conf.name,
    conf.id,
    c.common_name,
    c.id,
    p1.common_name,
    p1.id,
    p2.common_name,
    p2.id;


create or replace view experiment_cluster_report as
with result_counts as (
    select
        res.experiment_run_id,
        p.cluster_id,
        res.n_escaped_total as escaped,
        res.n_fragmented_total as fragmented
    from single_pathway_experiment_result res
    join pathway p on p.id = res.pathway_id
    union all
    -- Each fragmented trajectory contributes once, also in CLUSTER_TREE mode.
    select
        res.experiment_run_id,
        res.cluster_id,
        res.n_escaped_total as escaped,
        coalesce(sum(pf.count), 0) as fragmented
    from multi_cluster_result res
    left join pathway_fragmentation pf on pf.experiment_result_id = res.id
    group by res.id, res.experiment_run_id, res.cluster_id, res.n_escaped_total
    union all
    select
        failure.experiment_run_id,
        coalesce(failure.cluster_id, p.cluster_id) as cluster_id,
        null as escaped,
        null as fragmented
    from experiment_failure failure
    left join pathway p on p.id = failure.pathway_id
)
select
    er.id as experiment_run_id,
    er.start_time,
    er.simulation_mode,
    conf.name as config_name,
    conf.config,
    c.id as cluster_id,
    c.common_name as cluster_common_name,
    c.atomic_mass as cluster_atomic_mass,
    c.charge as cluster_charge,
    c.electronic_energy as cluster_electronic_energy,
    c.rotational_temperatures as cluster_rotational_temperatures,
    c.vibrational_temperatures as cluster_vibrational_temperatures,
    sum(escaped + fragmented) as realizations,
    case when any_value(er.simulation_mode) = 'PATHWAY_AT_A_TIME'
        then product(escaped / nullif(escaped + fragmented, 0))
        else sum(escaped) / nullif(sum(escaped + fragmented), 0)
    end as survival_rate,
    er.simulation_mode = 'PATHWAY_AT_A_TIME' as is_single_pathway
from result_counts
join experiment_run er on er.id = result_counts.experiment_run_id
join experiment_config conf on conf.id = er.experiment_config_id
join cluster c on c.id = result_counts.cluster_id
group by all
order by
    start_time,
    experiment_run_id,
    config_name,
    cluster_common_name;


create or replace view experiment_summary as
select
    er.id as experiment_run_id,
    conf.name as config_name,
    er.start_time,
    er.simulation_mode,
    (
        select count()
        from (
            select * from single_pathway_experiment_result
            union by name
            select * from multi_cluster_result
        ) as experiment_result
        where experiment_result.experiment_run_id = er.id
    ) as successes,
    (
        select count()
        from experiment_failure
        where experiment_failure.experiment_run_id = er.id
    ) as failures,
    er.simulation_mode = 'PATHWAY_AT_A_TIME' as is_single_pathway
from experiment_run as er
join experiment_config conf on conf.id = er.experiment_config_id;
