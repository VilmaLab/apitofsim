create table if not exists vib_energy_index (
    dos_id integer not null,
    foreign key (dos_id) references cluster_dos (id),
    mode varchar not null,
    data double[] not null,
    primary key (dos_id, mode)
);
