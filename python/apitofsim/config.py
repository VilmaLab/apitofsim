import json
import os
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, NotRequired, Tuple, TypedDict

import numpy
from pint import Quantity, get_application_registry
from pint.errors import PintError

from .api import (
    ClusterData,
    Gas,
    Histogram,
    MassSpectrometer,
    Quadrupole,
    _AtomLikeCluster,
    _Gas,
    _MassSpectrometer,
    _MolecularCluster,
    _Quadrupole,
    ureg,
)

Q_ = ureg.Quantity


class LengthComponents(TypedDict):
    L0: Quantity
    L1: Quantity
    L2: Quantity
    L3: Quantity
    Lsk: Quantity


class VoltageComponents(TypedDict):
    V0: Quantity
    V1: Quantity
    V2: Quantity
    V3: Quantity
    V4: Quantity


class PressureComponents(TypedDict):
    first: Quantity
    second: Quantity


class GasComponents(TypedDict):
    radius: Quantity
    mass: Quantity
    adiabatic_index: float


class QuadrupoleComponents(TypedDict):
    radiofrequency: Quantity
    r_quadrupole: Quantity
    dc_field: NotRequired[Quantity]
    ac_field: NotRequired[Quantity]


class SimulationConfig(TypedDict):
    """Quantity-based dictionary accepted by ExperimentRunner and ExperimentDatabase.

    Arrays are in CONFIG_FIELDS component order. Optional keys retain the
    runner's defaults when absent; a missing quadrupole means no mass filter.
    """

    lengths: Quantity
    voltages: Quantity
    pressures: Quantity
    gas: Gas
    T: Quantity
    N: int
    N_iter: int
    M_iter: int
    resolution: int
    tolerance: float
    dc: Quantity
    alpha_factor: Quantity
    bin_width: Quantity
    energy_max: Quantity
    energy_max_rate: Quantity
    quadrupole: NotRequired[Quadrupole]
    radius_pinhole: NotRequired[Quantity]
    bonding_energy: NotRequired[Quantity]
    fragmentation_energy: NotRequired[Quantity]


@dataclass(frozen=True)
class ConfigField:
    """Canonical unit and named array order (empty for scalars).

    ``dimensionless`` fields accept plain numbers. Dimensional fields require
    Pint quantities. Object members use dotted names, e.g. ``gas.mass``.
    """

    unit: str
    components: tuple[str, ...] = ()
    default: int | None = None


CONFIG_FIELDS = MappingProxyType(
    {
        **{
            key: ConfigField(unit)
            for key, unit in {
                "T": "kelvin",
                "N": "dimensionless",
                "N_iter": "dimensionless",
                "M_iter": "dimensionless",
                "resolution": "dimensionless",
                "tolerance": "dimensionless",
                "dc": "meter",
                "alpha_factor": "halfturn",
                "bin_width": "kelvin",
                "energy_max": "kelvin",
                "energy_max_rate": "kelvin",
                "radius_pinhole": "meter",
                "bonding_energy": "kelvin",
                "fragmentation_energy": "kelvin",
                "gas.radius": "meter",
                "gas.mass": "kilogram",
                "gas.adiabatic_index": "dimensionless",
                "quadrupole.radiofrequency": "hertz",
                "quadrupole.r_quadrupole": "meter",
            }.items()
        },
        "quadrupole.dc_field": ConfigField("volt", default=0),
        "quadrupole.ac_field": ConfigField("volt", default=0),
        "lengths": ConfigField("meter", ("L0", "L1", "L2", "L3", "Lsk")),
        "voltages": ConfigField("volt", ("V0", "V1", "V2", "V3", "V4")),
        "pressures": ConfigField("pascal", ("first", "second")),
    }
)


_OBJECT_TYPES = {"gas": Gas, "quadrupole": Quadrupole}


def _field_spec(field):
    if field in CONFIG_FIELDS:
        return CONFIG_FIELDS[field]
    group, _, component = field.partition(".")
    spec = CONFIG_FIELDS[group]
    if component not in spec.components:
        raise KeyError(field)
    return ConfigField(spec.unit)


def _check_components(values, names, field):
    missing = set(names) - values.keys()
    extra = values.keys() - set(names)
    if missing or extra:
        raise ValueError(
            f"{field}: expected components {', '.join(names)}; "
            f"missing {sorted(missing)}, unknown {sorted(extra)}"
        )


def _config_value(value, field, *, strict=False):
    """Decode stored values, or normalize and validate a builder input."""
    if field in _OBJECT_TYPES:
        cls = _OBJECT_TYPES[field]
        if not strict and (value is None or isinstance(value, cls)):
            return value
        if isinstance(value, cls):
            value = vars(value)
        if not isinstance(value, Mapping):
            raise ValueError(f"{field}: expected named components or {cls.__name__}")
        names = cls.__dataclass_fields__
        defaults = {
            name: Q_(spec.default, spec.unit)
            for name in names
            if (spec := CONFIG_FIELDS[f"{field}.{name}"]).default is not None
        }
        value = {**defaults, **value}
        if strict:
            _check_components(value, names, field)
        return cls(
            **{
                name: _config_value(component, f"{field}.{name}", strict=strict)
                for name, component in value.items()
            }
        )
    try:
        spec = _field_spec(field)
    except KeyError:
        return value  # Preserve unrelated keys in partial stored configurations.
    if spec.components and isinstance(value, Mapping):
        _check_components(value, spec.components, field)
        value = Q_(
            [
                _config_value(value[name], f"{field}.{name}", strict=strict)
                .to(spec.unit)
                .magnitude
                for name in spec.components
            ],
            spec.unit,
        )
    if not strict:
        return into_quantity(value, None if spec.unit == "dimensionless" else spec.unit)
    shape = (len(spec.components),) if spec.components else ()
    try:
        if isinstance(value, Quantity):
            # halfturn is absent from a consumer's unmodified Pint registry.
            value = (
                value.to("radian").magnitude / numpy.pi
                if spec.unit == "halfturn"
                else value.to(spec.unit).magnitude
            )
        elif spec.unit != "dimensionless":
            raise ValueError("a Pint quantity is required")
        arr = numpy.asarray(value)
        if (
            arr.shape != shape
            or arr.dtype.kind not in "iuf"
            or not numpy.isfinite(arr).all()
        ):
            raise ValueError(
                f"expected finite numeric {'array' if shape else 'scalar'} with shape {shape}"
            )
        if SimulationConfig.__annotations__.get(field) is int:
            if arr.item() != int(arr.item()):
                raise ValueError("expected an integer")
            if field == "N" and arr.item() < 0:
                raise ValueError("expected a non-negative integer")
            return int(arr.item())
        value = arr.astype(float).copy() if shape else arr.item()
        return value if spec.unit == "dimensionless" else Q_(value, spec.unit)
    except (TypeError, ValueError, PintError) as exc:
        raise ValueError(
            f"{field}: expected shape {shape} in {spec.unit}: {exc}"
        ) from exc


def validate_realizations(value) -> int:
    """Normalize a non-negative realization count using the builder's rules."""
    return _config_value(value, "N", strict=True)


def build_config(
    *,
    lengths: LengthComponents | Quantity,
    voltages: VoltageComponents | Quantity,
    pressures: PressureComponents | Quantity,
    gas: GasComponents | Gas,
    T: Quantity,
    N: int,
    N_iter: int,
    M_iter: int,
    resolution: int,
    tolerance: float,
    dc: Quantity,
    alpha_factor: Quantity,
    bin_width: Quantity,
    energy_max: Quantity,
    energy_max_rate: Quantity,
    quadrupole: QuadrupoleComponents | Quadrupole | None = None,
    radius_pinhole: Quantity | None = None,
    bonding_energy: Quantity | None = None,
    fragmentation_energy: Quantity | None = None,
) -> SimulationConfig:
    """Build a runner configuration from named components and Pint quantities.

    Normalize to CONFIG_FIELDS units without mutating inputs. All non-optional
    arguments are required; no numerical/physical tuning defaults are supplied.
    Quadrupole AC/DC voltages default to zero when a quadrupole is supplied.
    None omits optional scalar fields, retaining the runner's defaults (including
    the 1 mm pinhole). Energy histogram quantities use temperature units; callers
    supplying energies must explicitly convert with a Pint Boltzmann context.
    Form controls and submission-specific physical bounds belong to consumers.
    """
    return SimulationConfig(
        **{
            field: _config_value(value, field, strict=True)
            for field, value in locals().items()
            if value is not None or field in SimulationConfig.__required_keys__
        }
    )


def read_dat(fn):
    if os.stat(fn).st_size == 0:
        return None
    return numpy.asfortranarray(numpy.loadtxt(fn, dtype=numpy.float64))


def read_histogram(fn):
    arr = read_dat(fn)
    if arr is None:
        return None
    return Histogram(Q_(arr[:, 0], "kelvin"), arr[:, 1])


def read_skimmer(fn):
    arr = read_dat(fn)
    if arr is None:
        return None
    mesh_skimmer = arr[1, 0] - arr[0, 0]
    return arr[:, 1:4], mesh_skimmer


NAME_MAP = {
    "Cluster_charge_sign": "cluster_charge_sign",
    "Atomic_mass_cluster": "amu_0",
    "Atomic_mass_first_product": "amu_1",
    "Atomic_mass_second_product": "amu_2",
    "Temperature_(K)": "T",
    "Pressure_first_chamber(Pa)": "pressure_first",
    "Pressure_second_chamber(Pa)": "pressure_second",
    "Length_of_1st_chamber_(meters)": "L0",
    "Length_of_skimmer_(meters)": "Lsk",
    "Length_between_skimmer_and_front_quadrupole_(meters)": "L1",
    "Length_between_front_quadrupole_and_back_quadrupole_(meters)": "L2",
    "Length_between_back_quadrupole_and_2nd_skimmer_(meters)": "L3",
    "Voltage0_(Volt)": "V0",
    "Voltage1_(Volt)": "V1",
    "Voltage2_(Volt)": "V2",
    "Voltage3_(Volt)": "V3",
    "Voltage4_(Volt)": "V4",
    "Number_of_realizations": "N",
    "Radius_at_smallest_cross_section_skimmer_(m)": "dc",
    "Angle_of_skimmer_(multiple_of_PI)": "alpha_factor",
    "Fragmentation_energy_(Kelvin)": "bonding_energy",
    "Energy_max_for_density_of_states_(Kelvin)": "energy_max",
    "Energy_max_for_rate_constant_(Kelvin)": "energy_max_rate",
    "Energy_resolution_(Kelvin)": "bin_width",
    "Gas_molecule_radius_(meters)": "R_gas",
    "Gas_molecule_mass_(kg)": "m_gas",
    "Adiabatic_index": "ga",
    "DC_quadrupole": "dc_field",
    "AC_quadrupole": "ac_field",
    "Radiofrequency_quadrupole": "radiofrequency",
    "Half-distance_between_quadrupole_rods": "r_quadrupole",
    "Number_of_iterations_in_solving_equation": "N_iter",
    "Number_of_iterations_in_solving_equation2": "M_iter",
    "Number_of_solved_points": "resolution",
    "Tolerance_in_solving_equation": "tolerance",
    "Radius_at_pinhole": "radius_pinhole",
}

# Legacy spellings are adapters to the canonical fields, not another unit schema.
_LEGACY_NAMES = {
    **{
        name: f"{group}.{name}"
        for group in ("lengths", "voltages")
        for name in CONFIG_FIELDS[group].components
    },
    **{
        f"pressure_{name}": f"pressures.{name}"
        for name in CONFIG_FIELDS["pressures"].components
    },
    **{name: f"quadrupole.{name}" for name in Quadrupole.__dataclass_fields__},
    "R_gas": "gas.radius",
    "m_gas": "gas.mass",
    "ga": "gas.adiabatic_index",
}
_LEGACY_SHORTNAMES = {canonical: short for short, canonical in _LEGACY_NAMES.items()}
UNITS = {}
DEFAULTS = {}
for _short in NAME_MAP.values():
    try:
        _spec = _field_spec(_LEGACY_NAMES.get(_short, _short))
    except KeyError:
        continue
    UNITS[_short] = None if _spec.unit == "dimensionless" else _spec.unit
    if _spec.default is not None:
        DEFAULTS[_short] = _spec.default
TOPLEVEL = [
    name
    for name in SimulationConfig.__annotations__
    if name in NAME_MAP.values() and name in CONFIG_FIELDS
]
INT_PARAMS = ["cluster_charge_sign", "amu_0", "amu_1", "amu_2"] + [
    name for name, kind in SimulationConfig.__annotations__.items() if kind is int
]


class SettingsMetadata:
    def __init__(self):
        self.table = []
        self.lookups = {}
        for idx, (long_name, short_name) in enumerate(NAME_MAP.items()):
            long_name_no_units = long_name.split("_(")[0].split("(")[0]
            human_name_units = " ".join((word for word in long_name.split("_")))
            human_name_no_units = " ".join(
                (word for word in long_name_no_units.split("_"))
            )
            indices = {
                "long_name": long_name,
                "long_name_no_units": long_name_no_units,
                "human_name_units": human_name_units,
                "human_name_no_units": human_name_no_units,
                "short_name": short_name,
            }
            self.table.append(
                {
                    **indices,
                    "unit": UNITS.get(short_name),
                    "toplevel": short_name in TOPLEVEL,
                }
            )
            for index, value in indices.items():
                self.lookups.setdefault(index, {})[value] = idx

    def get(self, key, *, by="any"):
        if by == "any":
            for index in self.lookups.values():
                if key in index:
                    return self.table[index[key]]
        else:
            return self.table[self.lookups[by][key]]
        raise KeyError("Not found")


METADATA = SettingsMetadata()


def parse_config_to_pairs(fn):
    with open(fn) as f:
        for line in f:
            line = line.strip()
            if " " not in line:
                continue
            value, name = line.strip().split(maxsplit=1)
            yield name, value


def parse_config(fn):
    config = {}
    for name, value in parse_config_to_pairs(fn):
        shortname = NAME_MAP.get(name)
        if shortname is not None:
            if shortname in INT_PARAMS:
                value = int(value)
            else:
                value = float(value)
        config[name] = value
    check_for_deprecated_keys(config)
    return config


class ConfigFile:
    config: dict[str, Any]

    def __init__(self, *, filename=None, config=None, cwd=None):
        self.cwd = cwd
        if filename:
            config = parse_config(filename)
        elif config is None:
            raise ValueError("Either filename or config must be provided")
        self.config = config

    def get(self, quantity, *, by="any", asa="measurement"):
        if quantity in ("lengths", "voltages", "pressures"):
            spec = CONFIG_FIELDS[quantity]
            names = tuple(
                _LEGACY_SHORTNAMES[f"{quantity}.{name}"] for name in spec.components
            )
            value = numpy.array(
                [self.get(k, by="short_name", asa="value") for k in names]
            )
            if asa == "measurement":
                return Q_(value, spec.unit)
            return value
        if quantity in _OBJECT_TYPES:
            components = {
                name: self.get(
                    _LEGACY_SHORTNAMES[f"{quantity}.{name}"], by="short_name", asa=asa
                )
                for name in _OBJECT_TYPES[quantity].__dataclass_fields__
            }
            if (
                quantity == "quadrupole"
                and components["radiofrequency"] is None
                and components["r_quadrupole"] is None
            ):
                return None
            if asa == "measurement":
                return _config_value(components, quantity)
            cls = _Gas if quantity == "gas" else _Quadrupole
            return cls(**components)
        if quantity == "mass_spec":
            cls = MassSpectrometer if asa == "measurement" else _MassSpectrometer
            return lambda skimmer, mesh_skimmer: cls(
                skimmer,
                lengths=self.get("lengths", by="short_name", asa=asa),
                voltages=self.get("voltages", by="short_name", asa=asa),
                T=self.get("T", by="short_name", asa=asa),
                pressures=self.get("pressures", by="short_name", asa=asa),
                quadrupole=self.get("quadrupole", by="short_name", asa=asa),
                mesh_skimmer=mesh_skimmer,
            )
        if quantity in ("cluster", "first_product", "second_product"):
            from contextlib import chdir, nullcontext

            with chdir(self.cwd) if self.cwd else nullcontext():
                return cluster_from_particle_config(
                    get_particle(self.config, quantity), asa=asa
                )
        if quantity == "clusters":
            return (
                self.get("cluster", by=by, asa=asa),
                self.get("first_product", by=by, asa=asa),
                self.get("second_product", by=by, asa=asa),
            )
        entry = METADATA.get(quantity, by=by)
        value = self.config.get(entry["long_name"], DEFAULTS.get(entry["short_name"]))
        if value is None:
            return None
        unit = entry["unit"]
        if asa == "measurement" and unit is not None:
            return Q_(value, unit)
        return value

    def into_json_config(self):
        config = {
            k: self.get(k, by="short_name")
            for k in (["lengths", "voltages", "gas", "pressures"] + TOPLEVEL)
        }
        quadrupole = self.get("quadrupole")
        if quadrupole is not None:
            config["quadrupole"] = quadrupole
        return config


def get_particle(config, particle):
    particle_data = {}
    for quantity in [
        "vibrational_temperatures",
        "rotational_temperatures",
        "electronic_energy",
    ]:
        config_key = f"file_{quantity}_{particle}"
        particle_data[quantity] = read_dat(config[config_key])
        name = config[config_key].rsplit(".", 1)[0].rsplit("/", 1)[-1]
        if "_" in name:
            name = name.rsplit("_", 1)[0]
        particle_data["name"] = name
    particle_data["electronic_energy"] = particle_data["electronic_energy"][0]
    particle_data["atomic_mass"] = config[f"Atomic_mass_{particle}"]
    return particle_data


def parse_config_with_particles(fn):
    config = parse_config(fn)
    result = {"config": config}
    for particle in ["cluster", "first_product", "second_product"]:
        result[particle] = get_particle(config, particle)
    return result


def cluster_from_particle_config(
    particle_config, asa="measurement", ureg=get_application_registry(), charge=0
):
    vibrational_temperatures = particle_config["vibrational_temperatures"]
    if vibrational_temperatures is None:
        vibrational_temperatures = numpy.empty(0)
    if asa == "measurement":
        return ClusterData(
            ureg.Quantity(particle_config["atomic_mass"], "amu"),
            ureg.Quantity(particle_config["electronic_energy"], "hartree"),
            particle_config["rotational_temperatures"],
            vibrational_temperatures,
            charge,
        )
    if numpy.size(vibrational_temperatures) == 0:
        return _AtomLikeCluster(
            particle_config["atomic_mass"],
            particle_config["electronic_energy"],
            charge,
        )
    return _MolecularCluster(
        particle_config["atomic_mass"],
        particle_config["electronic_energy"],
        particle_config["rotational_temperatures"],
        vibrational_temperatures,
        charge,
    )


def get_clusters(
    full_config, ureg=get_application_registry()
) -> Tuple[ClusterData, ClusterData, ClusterData]:
    clusters = []
    for particle in ["cluster", "first_product", "second_product"]:
        particle_config = full_config[particle]
        cluster = cluster_from_particle_config(particle_config, ureg=ureg)
        clusters.append(cluster)
    return tuple(clusters)


def get_gas(config, ureg=get_application_registry()):
    return Gas(
        radius=ureg.Quantity(config["Gas_molecule_radius_(meters)"], "m"),
        mass=ureg.Quantity(config["Gas_molecule_mass_(kg)"], "kg"),
        adiabatic_index=config["Adiabatic_index"],
    )


def config_to_shortnames(config):
    return {NAME_MAP.get(k, k): v for k, v in config.items()}


def parse_config_list(fn):
    from contextlib import chdir

    with open(fn) as f:
        config_dict = json.load(f)
        for k, conf_info in config_dict.items():
            with chdir(conf_info["cwd"]):
                config_dict[k] = parse_config(conf_info["config"])
    return config_dict


def dump_to_raw(obj):
    import orjson
    import pint

    def default(obj):
        from ase import Atoms

        if isinstance(obj, Exception):
            return str(obj)
        elif isinstance(obj, Atoms):
            return str(obj.symbols)
        elif isinstance(obj, pint.Quantity):
            return [obj.magnitude, str(obj.units)]
        elif isinstance(obj, Gas | Quadrupole):
            return vars(obj)
        raise TypeError

    return orjson.dumps(obj, default=default, option=orjson.OPT_SERIALIZE_NUMPY)


def into_quantity(obj, default_unit=None):
    if isinstance(obj, list) and len(obj) == 2:
        return Q_(obj[0], obj[1])
    elif isinstance(obj, str):
        mag, unit = obj.split(" ", 1)
        return Q_(float(mag), unit)
    elif isinstance(obj, (int, float)) and default_unit is not None:
        return Q_(obj, default_unit)
    else:
        return obj


def into_quantity_obj(obj, key):
    return into_quantity(obj[key], UNITS.get(key))


def import_raw_config(config):
    for key, value in config.items():
        config[key] = _config_value(value, key)
    check_for_deprecated_keys(config)
    return config


def check_for_deprecated_keys(config):
    import warnings

    for k in config:
        if k == "cluster_charge_sign":
            warnings.warn(
                "'cluster_charge_sign' is deprecated, add cluster charges during data import instead",
                DeprecationWarning,
            )
