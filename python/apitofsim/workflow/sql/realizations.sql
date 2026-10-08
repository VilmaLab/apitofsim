create sequence realization_sequence start 1;
create sequence realization_event_sequence start 1;

create table realization (
    id integer default nextval('realization_sequence') primary key,
    experiment_result_id integer null
);

create type event_type as enum('init', 'collision', 'fragmentation', 'escape');
create type position_type as struct(x double, y double, z double, t double);
create type vector_type as struct(x double, y double, z double);

create table event_info (
    id integer default nextval('realization_event_sequence') primary key,
    realization_id integer not null references realization (id),
    event_type event_type not null,
    postime position_type not null,
    velocity vector_type not null,
    omega vector_type not null,
    rot_energy double not null,
    vibrational_energy double not null,
    particle_index integer not null
);

create table collision_event (
    id integer primary key references event_info (id),
    theta double not null,
    u_norm double not null,
    accepted boolean not null
);

create table fragmentation_event (
    id integer primary key references event_info (id),
    pathway_id integer null references pathway (id)
);
