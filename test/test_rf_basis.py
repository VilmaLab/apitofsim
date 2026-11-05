"""Persisted RF preparation, input ownership, and invalidation contracts."""

from dataclasses import replace

import numpy as np
import pytest
from apitofsim import api
from apitofsim.workflow import DerivedDataPreparer, SuperClusterDatabase


def rf_config(config):
    config["quadrupole"] = api.Quadrupole(
        api.Q_(2, "V"), api.Q_(100, "V"), api.Q_(1.3, "MHz"), api.Q_(6, "mm")
    )


def prepare(case):
    _, runner, config, _ = case
    return runner.prepare_from_config(config, cached_densityandrate=(1, 2))


def test_rf_persistence_and_shared_inputs(prepared_case, monkeypatch, tmp_path):
    db, runner, config, _ = prepared_case
    rf_config(config)
    prepared = prepare(prepared_case)
    # Two distinct clusters with equal masses share the dimensionless equation.
    first, second = (g.substance.rf_basis for g in prepared.groups)
    assert first is second
    row = db.db.execute("select * from rf_basis").fetchone()
    assert row is not None
    assert len(row[-1]) == 16 * (api.RFBasis.default_cells + 1)
    assert db.db.execute("select count(*) from rf_basis").fetchone() == (1,)
    cluster_indexed, names, pathways = db.get_all_lookups()
    assert api.prepare_rf_basis(config["quadrupole"], cluster_indexed[1]) is not None

    def unexpected(*args):
        pytest.fail("unexpected RF table reconstruction")

    monkeypatch.setattr(DerivedDataPreparer, "_build_rf_basis", unexpected)
    # A new on-disk database/preparer proves reuse is not a process-global cache.
    filename = str(tmp_path / "cache.duckdb")
    disk = SuperClusterDatabase(filename)
    disk.create_tables(foreign_keys=False)
    disk.db.execute("insert into rf_basis values (?, ?, ?, ?, ?)", row)
    disk.close()
    disk = SuperClusterDatabase(filename, readonly=True)
    try:
        loader = DerivedDataPreparer(disk)
        loader._prepare_rf_bases(config, cluster_indexed, pathways)
        np.testing.assert_array_equal(loader.rf_bases[1].data, first.data)
    finally:
        disk.close()

    config["T"] *= 1.1
    config["pressures"] *= 1.2
    config["vib_energy_sampler"] = "dos-only"
    config["hazard_rtol"] = 1e-8
    prepared = prepare(prepared_case)
    # Worker inputs retain ownership even after the preparer's map is replaced.
    runner.preparer.rf_bases = {}
    with runner.iter_mass_spec(prepared, prepared.groups[0], 2) as stream:
        for _ in stream:
            pass
        assert stream.result.result.counters.n_realizations == 2


@pytest.mark.parametrize(
    "field",
    [
        "ac_field",
        "dc_field",
        "radiofrequency",
        "r_quadrupole",
        "mass",
        "cells",
        "version",
    ],
)
def test_rf_cache_keys(prepared_case, field):
    db, _, config, _ = prepared_case
    rf_config(config)
    prepare(prepared_case)
    if field == "mass":
        db.db.execute("update cluster set atomic_mass = atomic_mass * 1.1 where id = 2")
    elif field in {"cells", "version"}:
        db.db.execute(f"update rf_basis set {field} = {field} + 1")
    else:
        q = config["quadrupole"]
        config["quadrupole"] = replace(q, **{field: getattr(q, field) * 1.1})
    prepared = prepare(prepared_case)
    assert db.db.execute("select count(*) from rf_basis").fetchone() == (2,)
    if field == "mass":
        assert (
            prepared.groups[0].substance.rf_basis
            is not prepared.groups[1].substance.rf_basis
        )


@pytest.mark.parametrize(
    "inactive", ["legacy", "no-quadrupole", "ac_field", "radiofrequency"]
)
def test_rf_not_prepared_when_unused(prepared_case, inactive, monkeypatch):
    db, runner, config, _ = prepared_case
    rf_config(config)
    if inactive == "legacy":
        config["collision_time_method"] = "legacy"
    elif inactive == "no-quadrupole":
        del config["quadrupole"]
    else:
        q = config["quadrupole"]
        config["quadrupole"] = replace(q, **{inactive: getattr(q, inactive) * 0})
    monkeypatch.setattr(
        runner.preparer, "_build_rf_basis", lambda *a: pytest.fail("RF preparation")
    )
    assert all(g.substance.rf_basis is None for g in prepare(prepared_case).groups)
    assert db.db.execute("select count(*) from rf_basis").fetchone() == (0,)


def test_rf_corrupt_cache_is_error(prepared_case):
    db, _, config, _ = prepared_case
    rf_config(config)
    prepare(prepared_case)
    db.db.execute("update rf_basis set data = [1.0]")
    with pytest.raises(ValueError, match="persisted RF basis"):
        prepare(prepared_case)


def test_rf_validation():
    raw = api.apitofsimraw
    for ac, dc, cells in [
        (np.nan, 0, 4096),
        (0, np.inf, 4096),
        (1, 0, 16),
        (0, -4, 4096),
        (0, 0, -1),
        (0, 0, 0),
    ]:
        with pytest.raises((ValueError, RuntimeError)):
            raw.RFBasis(ac, dc, cells)
    basis = raw.RFBasis(0.1, 0.03)
    data = basis.data
    data[32] += 0.01
    with pytest.raises(ValueError, match="validation|tolerance"):
        raw.RFBasis(0.1, 0.03, data)
