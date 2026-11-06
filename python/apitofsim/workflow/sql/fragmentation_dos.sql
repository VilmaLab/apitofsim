-- Separate from rate DOS: ground-inclusive log counts, keyed by ordered
-- product frequencies, actual SI grid, atom/molecule kind and algorithm version.
create table fragmentation_dos (
    key varchar primary key,
    data double[] not null
);
