create sequence experiment_config_sequence start 1;
create sequence experiment_run_sequence start 1;
create sequence experiment_result_sequence start 1;
create sequence pathway_fragmentation_sequence start 1;
create sequence fragmentation_product_sequence start 1;

create table experiment_config (
    id integer default nextval('experiment_config_sequence') primary key,
    name varchar,
    config json
);

create table experiment_run (
    id integer default nextval('experiment_run_sequence') primary key,
    experiment_config_id integer not null,
    run_config json,
    foreign key (experiment_config_id) references experiment_config (id),
    start_time timestamp
);

create table single_pathway_experiment_result (
    id integer default nextval('experiment_result_sequence') primary key,
    experiment_run_id integer not null,
    pathway_id integer not null,
    foreign key (experiment_run_id) references experiment_run (id),
    foreign key (pathway_id) references pathway (id),
    loop_us uint64,
    total_us uint64,
    nwarnings uint64,
    n_fragmented_total uint64,
    n_escaped_total uint64,
    ncoll_total uint64,
    counter_collision_rejections uint64
);

create table multi_pathway_experiment_result (
    id integer default nextval('experiment_result_sequence') primary key,
    experiment_run_id integer not null,
    foreign key (experiment_run_id) references experiment_run (id),
    cluster_id integer not null,
    foreign key (cluster_id) references cluster (id),
    loop_us uint64,
    total_us uint64,
    nwarnings uint64,
    n_escaped_total uint64,
    ncoll_total uint64,
    counter_collision_rejections uint64
);

create table pathway_fragmentation (
    id integer default nextval('pathway_fragmentation_sequence') primary key,
    experiment_result_id integer not null,
    foreign key (experiment_result_id) references multi_pathway_experiment_result (id),
    pathway_id integer not null,
    foreign key (pathway_id) references pathway (id),
    count uint64
);

create table fragmentation_product (
    id integer default nextval('fragmentation_product_sequence') primary key,
    experiment_result_id integer not null,
    foreign key (experiment_result_id) references multi_pathway_experiment_result (id),
    cluster_id integer not null,
    foreign key (cluster_id) references cluster (id),
    count uint64
);

create table experiment_failure (
    id integer default nextval('experiment_result_sequence') primary key,
    experiment_run_id integer not null,
    foreign key (experiment_run_id) references experiment_run (id),
    pathway_id integer,
    foreign key (pathway_id) references pathway (id),
    cluster_id integer,
    foreign key (cluster_id) references cluster (id),
    exc_name varchar,
    msg varchar,
    overflow_requested double
);
