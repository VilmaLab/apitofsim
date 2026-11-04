-- Format version includes the numerical construction and interpolation contract.
-- No DOS or cluster foreign key: different physical instruments/species can
-- share the same dimensionless transverse equation.
create table if not exists rf_basis (
    ac_scaled double not null,
    dc_scaled double not null,
    cells integer not null,
    version integer not null,
    data double[] not null,
    primary key (ac_scaled, dc_scaled, cells, version)
);
