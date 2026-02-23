create sequence cluster_id_sequence start 1;
create sequence pathway_id_sequence start 1;

create table cluster (
    id integer default nextval('cluster_id_sequence') primary key,
    common_name varchar,
    atomic_mass double,
    charge int,
    electronic_energy double,
    rotational_temperatures double[3],
    vibrational_temperatures double[],
    import_info json
);

create table pathway (
    id integer default nextval('pathway_id_sequence') primary key,
    cluster_id integer not null,
    product1_id integer not null,
    product2_id integer not null,
    foreign key (cluster_id) references cluster (id),
    foreign key (product1_id) references cluster (id),
    foreign key (product2_id) references cluster (id)
);
