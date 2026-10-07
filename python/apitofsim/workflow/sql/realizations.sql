create sequence realization_sequence start 1;
create sequence realization_event_sequence start 1;

create table realization (
    id integer default nextval('realization_sequence') primary key,
    experiment_result_id integer null
);

create type position_type as struct(x float, y float, z float, t float);

create table collision_event (
    id integer default nextval('realization_event_sequence') primary key,
    realization_id integer not null,
    foreign key (realization_id) references realization (id),
    postime position_type not null
);

create table fragmentation_event (
    id integer default nextval('realization_event_sequence') primary key,
    realization_id integer not null,
    foreign key (realization_id) references realization (id),
    postime position_type not null,
    pathway_id integer null,
);

create table escape_event (
    id integer default nextval('realization_event_sequence') primary key,
    realization_id integer not null,
    foreign key (realization_id) references realization (id),
    postime position_type not null,
);

create table init_event (
    id integer default nextval('realization_event_sequence') primary key,
    realization_id integer not null references realization (id),
    postime struct(x double, y double, z double, t double) not null,
    velocity struct(x double, y double, z double) not null,
    omega struct(x double, y double, z double) not null,
    rot_energy double not null,
    internal_energy double not null,
    particle_index integer not null
);
