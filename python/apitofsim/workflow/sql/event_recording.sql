-- Schema version 1 selects the Arrow IPC event sidecar and version 2 selects
-- Parquet. Batch ranges apply only to IPC.
create table event_recording_state (
    schema_version integer primary key,
    archive_id varchar not null,
    generation bigint not null,
    file_size bigint not null
);

insert into event_recording_state values (1, 'database:' || uuid()::varchar, 0, 0);

create table event_execution (
    result_id bigint primary key,
    run_id integer not null,
    cluster_id integer,
    pathway_id integer,
    pathway_ids integer[] not null,
    recorded boolean not null,
    status varchar not null,
    failure_id integer,
    event_count bigint not null default 0,
    batch_start bigint not null default 0,
    batch_count bigint not null default 0
);
