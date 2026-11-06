-- Separate from rate DOS: ground-inclusive log counts, keyed by ordered
-- product frequencies, actual SI grid, payload kind and algorithm version.
-- Version 2 stores shared marginal and combined tables separately. Version 1
-- pair snapshots may coexist but are not reused by the version 2 preparation.
create table fragmentation_dos (
    key varchar primary key,
    data double[] not null
);
