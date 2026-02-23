create sequence histogram_params_sequence start 1;
create sequence dos_sequence start 1;
create sequence products_dos_sequence start 1;
create sequence k_rate_mesh_sequence start 1;
create sequence k_rate_sequence start 1;

create table histogram_params (
    id integer default nextval('histogram_params_sequence') primary key,
    bin_width double,
    max double
);

create table cluster_dos (
    id integer default nextval('dos_sequence') primary key,
    histogram_params_id integer not null,
    foreign key (histogram_params_id) references histogram_params (id),
    cluster_id integer not null,
    foreign key (cluster_id) references cluster (id),
    data double[]
);

create table products_dos (
    id integer default nextval('products_dos_sequence') primary key,
    histogram_params_id integer not null,
    foreign key (histogram_params_id) references histogram_params (id),
    cluster1_id integer not null,
    cluster2_id integer not null,
    foreign key (cluster1_id) references cluster (id),
    foreign key (cluster2_id) references cluster (id),
    data double[]
);

create table k_rate_mesh (
    id integer default nextval('k_rate_mesh_sequence') primary key,
    histogram_params_id integer not null,
    foreign key (histogram_params_id) references histogram_params (id),
    data double[]
);

create table k_rate (
    id integer default nextval('k_rate_sequence') primary key,
    pathway_id integer not null,
    foreign key (pathway_id) references pathway (id),
    histogram_params_id integer not null,
    foreign key (histogram_params_id) references histogram_params (id),
    data double[]
);
