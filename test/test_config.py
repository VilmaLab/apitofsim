from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
from apitofsim import (
    CONFIG_FIELDS,
    Gas,
    MassSpectrometer,
    Quadrupole,
    build_config,
    skimmer,
)
from apitofsim.config import ConfigFile
from apitofsim.workflow import ExperimentDatabase, ExperimentRunner
from pint import UnitRegistry

# A consumer-owned registry works too; the builder returns application quantities.
Q_ = UnitRegistry().Quantity


@pytest.fixture
def inputs():
    return dict(
        lengths=dict(
            Lsk=Q_(0.5, "mm"),
            L3=Q_(4.48, "mm"),
            L2=Q_(10.1, "cm"),
            L1=Q_(2.44, "mm"),
            L0=Q_(1, "mm"),
        ),
        voltages={
            f"V{i}": Q_(v * 1000, "mV") for i, v in enumerate([-19, -9, -7, -6, 11])
        },
        pressures=dict(second=Q_(3.53, "Pa"), first=Q_(1.82, "mbar")),
        gas=dict(
            radius=Q_(2.46, "angstrom"), mass=Q_(4.8506e-23, "g"), adiabatic_index=1.4
        ),
        quadrupole=dict(
            dc_field=Q_(0, "mV"),
            ac_field=Q_(0.2, "kV"),
            radiofrequency=Q_(1.3, "MHz"),
            r_quadrupole=Q_(6, "mm"),
        ),
        T=Q_(26.85, "degC"),
        N=2,
        N_iter=1000,
        M_iter=1000,
        resolution=1000,
        tolerance=1e-8,
        dc=Q_(0.5, "mm"),
        alpha_factor=Q_(45, "degree"),
        bin_width=Q_(8, "K"),
        energy_max=Q_(200, "kK"),
        energy_max_rate=Q_(100, "kK"),
        bonding_energy=Q_(0, "K"),
    )


def assert_same(actual, expected):
    assert actual.keys() == expected.keys()
    for key, value in actual.items():
        other = expected[key]
        if isinstance(value, Gas | Quadrupole):
            assert_same(vars(value), vars(other))
        elif hasattr(value, "units"):
            np.testing.assert_allclose(value.magnitude, other.to(value.units).magnitude)
        else:
            assert value == other


def test_named_mixed_units_match_legacy_and_runner(inputs):
    config = build_config(**inputs)
    legacy = ConfigFile(filename=Path(__file__).parent / "data/raw/config.in")
    expected = legacy.into_json_config()
    # The legacy file has no optional pinhole radius.
    expected.pop("radius_pinhole")
    assert_same(config, expected)
    for name, values in dict(
        lengths=[0.001, 0.00244, 0.101, 0.00448, 0.0005],
        voltages=[-19, -9, -7, -6, 11],
        pressures=[182, 3.53],
    ).items():
        np.testing.assert_allclose(config[name].magnitude, values)
        np.testing.assert_allclose(legacy.get(name, asa="value"), values)
    assert CONFIG_FIELDS["lengths"].components == ("L0", "L1", "L2", "L3", "Lsk")
    assert CONFIG_FIELDS["pressures"].components == ("first", "second")
    assert CONFIG_FIELDS["gas.mass"].unit == "kilogram"
    assert config["quadrupole"].ac_field.magnitude == 200
    np.testing.assert_allclose(skimmer(config), skimmer(expected))
    instrument = MassSpectrometer(
        skimmer(config),
        config["lengths"],
        config["voltages"],
        config["T"],
        config["pressures"],
        quadrupole=config["quadrupole"],
    )
    assert instrument.into_cpp() is not None
    # The builder never rewrites the consumer's mappings or units.
    assert str(inputs["lengths"]["L0"].units) == "millimeter"


def test_objects_arrays_and_database_roundtrip(inputs):
    inputs["quadrupole"]["dc_field"] = Q_(10, "V")
    config = build_config(
        **inputs, radius_pinhole=Q_(2, "mm"), fragmentation_energy=Q_(150, "K")
    )
    again = build_config(**config)
    assert_same(config, again)
    assert isinstance(again["gas"], Gas)
    assert isinstance(again["quadrupole"], Quadrupole)
    db = ExperimentDatabase(":memory:")
    try:
        db.create_tables()
        db.insert_config("custom", config)
        (row,) = db.iter_configs("custom")
        assert_same(row.config, config)
    finally:
        db.close()


def test_optional_fields_and_quadrupole_defaults(inputs):
    inputs.pop("quadrupole")
    inputs.pop("bonding_energy")
    config = build_config(**inputs)
    assert "quadrupole" not in config
    assert "radius_pinhole" not in config
    assert "bonding_energy" not in config
    inputs["quadrupole"] = dict(radiofrequency=Q_(1, "MHz"), r_quadrupole=Q_(6, "mm"))
    config = build_config(**inputs)
    assert config["quadrupole"].dc_field.magnitude == 0
    assert config["quadrupole"].ac_field.magnitude == 0


@pytest.mark.parametrize(
    "field,value,match",
    [
        ("lengths", dict(L0=Q_(1, "m")), "lengths.*missing.*L1"),
        ("voltages", Q_([1, 2], "V"), "voltages.*shape"),
        ("pressures", Q_([[1, 2]], "Pa"), "pressures.*shape"),
        ("lengths", Q_([1] * 5, "s"), "lengths.*meter"),
        ("gas", dict(radius=Q_(1, "m")), "gas.*missing.*mass"),
        ("quadrupole", dict(radiofrequency=Q_(1, "Hz")), "quadrupole.*r_quadrupole"),
        ("T", Q_(1, "m"), "T.*kelvin"),
        ("T", 300, "T.*Pint quantity"),
        ("N_iter", 2.5, "N_iter.*integer"),
        ("resolution", True, "resolution.*numeric scalar"),
        ("dc", Q_([1, 2], "m"), "dc.*scalar"),
        ("tolerance", np.nan, "tolerance.*finite"),
        ("gas", None, "gas"),
    ],
)
def test_invalid_fields(inputs, field, value, match):
    inputs[field] = value
    with pytest.raises(ValueError, match=match):
        build_config(**inputs)


def test_invalid_named_component(inputs):
    inputs["lengths"]["L1"] = Q_(3, "second")
    with pytest.raises(ValueError, match=r"lengths.L1.*meter"):
        build_config(**inputs)
    inputs["lengths"]["L1"] = Q_(1, "m")
    inputs["pressures"]["third"] = Q_(1, "Pa")
    with pytest.raises(ValueError, match="pressures.*unknown.*third"):
        build_config(**inputs)
    inputs = deepcopy(inputs)
    del inputs["pressures"]
    with pytest.raises(TypeError, match="pressures"):
        build_config(**inputs)


def test_legacy_optional_quadrupole():
    legacy = ConfigFile(filename=Path(__file__).parent / "data/raw/config.in")
    for name in (
        "DC_quadrupole",
        "AC_quadrupole",
        "Radiofrequency_quadrupole",
        "Half-distance_between_quadrupole_rods",
    ):
        del legacy.config[name]
    assert legacy.get("quadrupole") is None
    assert "quadrupole" not in legacy.into_json_config()


def test_public_runner_preparation(inputs):
    inputs.update(
        bin_width=Q_(1000, "K"),
        energy_max=Q_(100_000, "K"),
        energy_max_rate=Q_(50_000, "K"),
    )
    config = build_config(**inputs)
    db = ExperimentDatabase(":memory:")
    try:
        db.create_tables()
        for name, mass, charge, energy in (
            ("parent", 200, -1, -1),
            ("ion", 100, -1, -0.5),
            ("neutral", 100, 0, -0.49),
        ):
            db.insert_cluster(name, mass, charge, energy, [1, 2, 3], [10, 20, 30])
        db.insert_pathway(1, 2, 3)
        config_id = db.insert_config("built", config)
        runner = ExperimentRunner(db)
        runner.start_run(config_id)
        prepared = runner.prepare_from_config(config)
        assert_same(vars(prepared.mass_spec.quadrupole), vars(config["quadrupole"]))
        with runner.iter_mass_spec(prepared, prepared.groups[0], N=0) as stream:
            list(stream)
        assert db.db.execute("select count(*) from experiment_result").fetchone() == (
            1,
        )
    finally:
        db.close()


def test_partial_stored_config_decoding():
    from apitofsim.config import import_raw_config

    raw = dict(
        name="partial",
        bin_width="8 kelvin",
        lengths=[[1, 2], "mm"],  # Stored partial configs are not shape-validated.
        N_iter=1.5,  # Integer validation belongs to the builder, too.
        gas=dict(radius=2.46e-10, mass=[4.8506e-26, "kg"], adiabatic_index=1.4),
        quadrupole=dict(radiofrequency="1.3 MHz", r_quadrupole=[6, "mm"]),
    )
    quadrupole_components = raw["quadrupole"]
    decoded = import_raw_config(raw)
    assert decoded["name"] == "partial"
    assert decoded["bin_width"].to("K").magnitude == 8
    np.testing.assert_array_equal(decoded["lengths"].to("mm").magnitude, [1, 2])
    assert decoded["N_iter"] == 1.5
    assert decoded["gas"].radius.to("m").magnitude == 2.46e-10
    assert decoded["gas"].into_cpp() is not None
    assert decoded["quadrupole"].ac_field.to("V").magnitude == 0
    assert decoded["quadrupole"].into_cpp() is not None
    # Existing objects are passed through by the stored-config reader.
    gas = decoded["gas"]
    assert import_raw_config(decoded)["gas"] is gas
    assert quadrupole_components.keys() == {"radiofrequency", "r_quadrupole"}
