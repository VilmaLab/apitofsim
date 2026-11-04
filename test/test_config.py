from contextlib import closing
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


def test_nested_simulation_defaults_and_database_roundtrip():
    import tomllib

    from apitofsim.config import import_raw_config, merge_config, simulation_option
    from apitofsim.workflow.base import SimulationMode

    source = tomllib.loads("""
[default_config.simulation]
N = 10
hazard_atol = 1e-8
hazard_rtol = 1e-6
energy_max = "200 kK"
energy_max_rate = "100 kK"

[[configs]]
name = "custom"
[configs.simulation]
collision_time_method = "legacy"
hazard_rtol = 1e-4
N = 20
simulation_mode = "cluster-tree"
sample_mode = "dss_normalized"
use_old_dos_impl = true
k_total_mesh_mode = "no_mesh"
k_rate_cancellation = false
""")
    config = merge_config(source["default_config"], source["configs"][0])
    assert config["simulation"]["N"] == 20
    assert source["default_config"]["simulation"]["N"] == 10
    with closing(ExperimentDatabase(":memory:")) as db:
        db.create_tables()
        db.insert_config("custom", config)
        (row,) = db.iter_configs()
    assert "simulation" not in row.config
    assert row.config["energy_max"].to("K").magnitude == 200_000
    assert row.config["N"] == 20
    assert row.config["use_old_dos_impl"] is True
    assert row.config["k_rate_cancellation"] is False
    assert row.config["collision_time_method"] == "legacy"
    assert row.config["hazard_atol"] == 1e-8
    assert row.config["hazard_rtol"] == 1e-4
    assert (
        simulation_option(row.config, "simulation_mode") == SimulationMode.CLUSTER_TREE
    )
    assert simulation_option(row.config, "sample_mode").name == "dss_normalized"
    assert simulation_option(row.config, "k_total_mesh_mode").name == "no_mesh"
    assert import_raw_config(deepcopy(row.config)) == row.config


@pytest.mark.parametrize(
    "raw,match",
    [
        ({"simulation": {"tolerence": 1e-8}}, "unknown.*tolerence"),
        ({"N": 1, "simulation": {"N": 2}}, "also present.*N"),
        ({"simulation": []}, "expected a table"),
        ({"simulation": {"simulation_mode": "invalid"}}, "simulation_mode.*expected"),
        ({"simulation": {"sample_mode": "invalid"}}, "sample_mode.*expected"),
        (
            {"simulation": {"vib_energy_sampler": "invalid"}},
            "vib_energy_sampler.*expected",
        ),
        (
            {"simulation": {"k_total_mesh_mode": "invalid"}},
            "k_total_mesh_mode.*expected",
        ),
        (
            {"simulation": {"collision_time_method": "invalid"}},
            "collision_time_method.*expected",
        ),
        ({"simulation": {"hazard_atol": 0}}, "hazard_atol.*positive"),
        ({"simulation": {"hazard_rtol": float("nan")}}, "hazard_rtol.*finite"),
        ({"simulation": {"use_old_dos_impl": "false"}}, "expected a boolean"),
        ({"simulation": {"k_rate_cancellation": "false"}}, "expected a boolean"),
        ({"simulation": {"k_rate_cancellation": 0}}, "expected a boolean"),
    ],
)
def test_invalid_simulation_settings(raw, match):
    from apitofsim.config import import_raw_config

    with pytest.raises(ValueError, match=match):
        import_raw_config(raw)


@pytest.mark.parametrize("kwargs", [{"hazard_atol": 0}, {"hazard_rtol": float("inf")}])
def test_native_collision_time_rejects_invalid_tolerances(kwargs):
    from apitofsim import CollisionTimeMethod, CollisionTimeOptions

    with pytest.raises(ValueError, match=next(iter(kwargs))):
        CollisionTimeOptions(CollisionTimeMethod.legacy, **kwargs)


def test_builder_simulation_options(inputs):
    config = build_config(
        **inputs,
        simulation_mode="pathway-at-a-time",
        sample_mode="dss_unnormalized",
        collision_time_method="integrated-hazard",
        hazard_atol=2e-10,
        hazard_rtol=3e-5,
        use_old_dos_impl=True,
        k_total_mesh_mode="compute_mesh_single_threaded",
        k_rate_cancellation=False,
    )
    assert_same(build_config(**config), config)


@pytest.mark.parametrize(
    "mesh_mode", ["compute_mesh_single_threaded", "compute_mesh_linear", "no_mesh"]
)
@pytest.mark.parametrize("atom_like", [False, True])
@pytest.mark.parametrize("cancellation", [None, True, False])
def test_preparation_uses_simulation_options(
    inputs, monkeypatch, mesh_mode, atom_like, cancellation
):
    import apitofsim.api as api
    from apitofsim.api import MeshMode
    from apitofsim.workflow import DerivedDataPreparer

    calls = {}
    real_dos = api.compute_density_of_states_batch
    real_rates = api.compute_k_total_batch

    def dos(*args, **kwargs):
        calls["dos"] = kwargs["use_old_impl"]
        return real_dos(*args, **kwargs)

    def rates(*args, **kwargs):
        calls["mesh"] = kwargs["mesh"]
        calls["cancellation"] = kwargs["k_rate_cancellation"]
        return real_rates(*args, **kwargs)

    monkeypatch.setattr(api, "compute_density_of_states_batch", dos)
    monkeypatch.setattr("apitofsim.compute_k_total_batch", rates)
    config = build_config(
        **{
            **inputs,
            "bin_width": Q_(1, "K"),
            "energy_max": Q_(1999, "K"),
            "energy_max_rate": Q_(1000, "K"),
        },
        use_old_dos_impl=True,
        k_total_mesh_mode=mesh_mode,
        k_rate_cancellation=cancellation,
    )
    with closing(ExperimentDatabase(":memory:")) as db:
        db.create_tables()
        db.insert_cluster("parent", 200, -1, -1, [1, 2, 3], [10, 20, 30])
        db.insert_cluster("ion", 100, -1, -0.5, [1, 2, 3], [10, 20, 30])
        db.insert_cluster(
            "neutral",
            100,
            0,
            -0.499,
            None if atom_like else [1, 2, 3],
            None if atom_like else [10, 20, 30],
        )
        db.insert_pathway(1, 2, 3)
        clusters, names, pathways = db.get_all_lookups()
        preparer = DerivedDataPreparer(db)
        first, _ = preparer.run_densityandrate(
            config, clusters, pathways, name_lookup=names, show_progress=False
        )
        assert calls == {
            "dos": True,
            "mesh": MeshMode[mesh_mode],
            "cancellation": True if cancellation is None else cancellation,
        }
        calls.clear()
        cached, _ = preparer.run_densityandrate(
            config, clusters, pathways, name_lookup=names, show_progress=False
        )
        assert not calls
        np.testing.assert_array_equal(first[1], cached[1])


@pytest.mark.parametrize(
    "mode", ["single-cluster", "pathway-at-a-time", "cluster-tree"]
)
@pytest.mark.parametrize("vib_mode", ["direct", "dos-only", "group-dos"])
def test_runner_uses_configured_mode_and_sampling(inputs, monkeypatch, mode, vib_mode):
    from apitofsim.api import CollisionTimeMethod, SampleMode
    from apitofsim.workflow.base import SimulationMode

    config = build_config(
        **{**inputs, "N": 1, "bin_width": Q_(1000, "K")},
        simulation_mode=mode,
        sample_mode="dss_normalized",
        vib_energy_sampler=vib_mode,
        collision_time_method="legacy",
        hazard_atol=4e-8,
        hazard_rtol=2e-6,
    )
    monkeypatch.delenv("N_OVERRIDE", raising=False)
    with closing(ExperimentDatabase(":memory:")) as db:
        db.create_tables()
        for name, mass, charge, energy in (
            ("parent", 200, -1, -1),
            ("ion", 100, -1, -0.5),
            ("neutral", 100, 0, -0.49),
        ):
            db.insert_cluster(name, mass, charge, energy, [1, 2, 3], [10, 20, 30])
        db.insert_pathway(1, 2, 3)
        config_id = db.insert_config("configured", config)
        runner = ExperimentRunner(db)
        calls = []
        method = "iter_mass_spec" if mode == "single-cluster" else "run_mass_spec"
        original = getattr(runner, method)

        def run(*args, **kwargs):
            options = (
                args[0].collision_time
                if method == "iter_mass_spec"
                else kwargs["collision_time"]
            )
            calls.append(
                (
                    kwargs["sample_mode"],
                    options.method,
                    options.hazard_atol,
                    options.hazard_rtol,
                )
            )
            return original(*args, **kwargs)

        monkeypatch.setattr(runner, method, run)
        runner.run_prepared_config(name="configured")
        assert calls == [
            (SampleMode.dss_normalized, CollisionTimeMethod.legacy, 4e-8, 2e-6)
        ]
        assert db.db.execute(
            "select experiment_config_id, simulation_mode from experiment_run"
        ).fetchall() == [(config_id, mode.replace("-", "_").upper())]
        # An explicit run argument overrides the stored mode.
        runner.run_from_config(config, mode=SimulationMode.SINGLE_CLUSTER)
        assert runner._simulation_mode == SimulationMode.SINGLE_CLUSTER


@pytest.mark.parametrize(
    "value",
    [
        "single-cluster",
        "pathway-at-a-time",
        "cluster-tree",
        "SINGLE_CLUSTER",
        "Cluster_Tree",
    ],
)
def test_simulation_mode_matches_cli(inputs, value):
    import click
    from apitofsim.cli import run
    from apitofsim.config import import_raw_config, simulation_option

    option = next(param for param in run.params if param.name == "simulation_mode")
    with click.Context(run, **run.context_settings) as context:
        expected = option.type.convert(value, option, context)
    config = build_config(**inputs, simulation_mode=value)
    assert config["simulation_mode"] == expected.name.replace("_", "-").lower()
    assert simulation_option(config, "simulation_mode") == expected
    assert import_raw_config({"simulation": {"simulation_mode": value}}) == {
        "simulation_mode": config["simulation_mode"]
    }


def test_vibrational_index_dynamic_range():
    from apitofsim.api import VibEnergyMode
    from apitofsim.workflow import DerivedDataPreparer

    density = np.array([1e308, 0.0, 1e308])
    with pytest.raises(ValueError, match="overflow for cluster 17"):
        DerivedDataPreparer._build_vib_index(VibEnergyMode.dos_only, density, 17)
    tree = DerivedDataPreparer._build_vib_index(VibEnergyMode.group_dos, density, 17)
    assert tree[1] == pytest.approx(np.log(1e308) + np.log(2))
    assert tree[5] == tree[7] == -np.inf


def test_sampler_preparation_with_partly_warmed_dos(inputs):
    from apitofsim.workflow import DerivedDataPreparer

    config = build_config(
        **{
            **inputs,
            "bin_width": Q_(1, "K"),
            "energy_max": Q_(1999, "K"),
            "energy_max_rate": Q_(1000, "K"),
        },
        vib_energy_sampler="group-dos",
    )
    with closing(ExperimentDatabase(":memory:")) as db:
        db.create_tables()
        for name, energy, charge in (
            ("parent", -1, -1),
            ("ion", -0.5, -1),
            ("neutral", -0.499, 0),
            ("neutral2", -0.498, 0),
        ):
            db.insert_cluster(name, 100, charge, energy, [1, 2, 3], [10, 20, 30])
        db.insert_pathway(1, 2, 3)
        preparer = DerivedDataPreparer(db)
        for i in (1, 2):
            clusters, names, paths = db.get_all_lookups()
            preparer.run_densityandrate(
                config, clusters, paths, name_lookup=names, show_progress=False
            )
            assert db.db.execute("select count(*) from products_dos").fetchone() == (i,)
            assert db.db.execute(
                "select count(*) from vib_energy_index"
            ).fetchone() == (1,)
            if i == 1:
                db.insert_pathway(1, 2, 4)
