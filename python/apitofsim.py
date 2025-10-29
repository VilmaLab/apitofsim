# This is imported just to get its bundled version of OpenMP on Windows
import sklearn  # noqa: F401

import os
import numpy
import json
from typing import Any, Callable
from dataclasses import dataclass
from enum import Enum
from pandas import DataFrame
from pint import get_application_registry, Quantity

from _apitofsim import (
    skimmer as _skimmer,
    ClusterData as _ClusterData,
    Gas as _Gas,
    densityandrate as _densityandrate,
    Histogram as _Histogram,
    Quadrupole as _Quadrupole,
    pinhole as _pinhole,
)


ureg = get_application_registry()
ureg.define(
    "halfturn = 2 * π * radian = _ = halfrevolution = halfcycle = halfcircle = multiple_of_PI"
)
Q_ = ureg.Quantity


__all__ = [
    # Imports
    "ClusterData",
    "Gas",
    "Quadrupole",
    "densityandrate",
    "pinhole",
    "skimmer",
    "parse_config_with_particles",
    "config_to_shortnames",
    "read_dat",
    "read_histogram",
    "read_skimmer",
    "get_clusters",
    "get_gas",
]


@dataclass
class ClusterData:
    mass: Quantity
    electronic_energy: Quantity
    rotations: Any
    frequencies: Any

    def into_cpp(self):
        return _ClusterData(
            self.mass.to("amu").magnitude,
            self.electronic_energy.to("hartree").magnitude,
            self.rotations,
            numpy.asfortranarray(self.frequencies, dtype=numpy.float64),
        )


@dataclass
class Gas:
    radius: Quantity
    mass: Quantity
    adiabatic_index: float

    def into_cpp(self):
        return _Gas(
            self.radius.to("m").magnitude,
            self.mass.to("kg").magnitude,
            self.adiabatic_index,
        )


@dataclass
class Histogram:
    x: Quantity
    y: numpy.ndarray

    @classmethod
    def from_cpp(cls, histogram: _Histogram):
        return cls(Q_(histogram.x, "kelvin"), histogram.y)

    def into_cpp(self):
        return _Histogram(self.x.to("kelvin").magnitude, self.y)


@dataclass
class Quadrupole:
    dc_field: Quantity
    ac_field: Quantity
    radiofrequency: Quantity
    r_quadrupole: Quantity

    def into_cpp(self):
        return _Quadrupole(
            self.dc_field.to("volts").magnitude,
            self.ac_field.to("volts").magnitude,
            self.radiofrequency.to("hertz").magnitude,
            self.r_quadrupole.to("m").magnitude,
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


INT_PARAMS = [
    "cluster_charge_sign",
    "amu_0",
    "amu_1",
    "amu_2",
    "N",
    "N_iter",
    "M_iter",
    "resolution",
]


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
}

UNITS = {
    "dc_field": "volts",
    "ac_field": "volts",
    "radiofrequency": "Hz",
    "r_quadrupole": "meters",
}

TOPLEVEL = [
    "T",
    "pressure_first",
    "pressure_second",
    "N",
    "N_iter",
    "M_iter",
    "dc",
    "alpha_factor",
    "bonding_energy",
    "energy_max",
    "energy_max_rate",
    "bin_width",
    "resolution",
    "tolerance",
]

DEFAULTS = {
    "dc_field": 0,
    "ac_field": 0,
}


class SettingsMetadata:
    def __init__(self):
        self.table = []
        self.lookups = {}
        for idx, (long_name, short_name) in enumerate(NAME_MAP.items()):
            long_name_unit_bits = long_name.split("_(")
            long_name_unit_bits = [
                *long_name_unit_bits[0].split("("),
                *long_name_unit_bits[1:],
            ]
            unit = None
            if len(long_name_unit_bits) > 1:
                unit = long_name_unit_bits[-1][:-1]
                if unit == "Kelvin":
                    unit = "K"
            if unit is None:
                unit = UNITS.get(short_name)
            long_name_no_units = long_name_unit_bits[0]
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
                {**indices, "unit": unit, "toplevel": short_name in TOPLEVEL}
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
    return config


class ConfigFile:
    config: dict[str, Any]

    def __init__(self, *, filename=None, config=None):
        if filename:
            config = parse_config(filename)
        elif config is None:
            raise ValueError("Either filename or config must be provided")
        self.config = config

    def get(self, quantity, *, by="any", asa="measurement"):
        if quantity == "lengths":
            value = numpy.array(
                [
                    self.get(k, by="short_name", asa="value")
                    for k in ["L0", "L1", "L2", "L3", "Lsk"]
                ]
            )
            if asa == "measurement":
                value = Q_(value, "meter")
            return value
        if quantity == "voltages":
            value = numpy.array(
                [self.get(f"V{k}", by="short_name", asa="value") for k in range(5)]
            )
            if asa == "measurement":
                value = Q_(value, "volts")
            return value
        if quantity == "gas":
            cls = Gas if asa == "measurement" else _Gas
            return cls(
                adiabatic_index=self.get("ga", by="short_name", asa=asa),
                radius=self.get("R_gas", by="short_name", asa=asa),
                mass=self.get("m_gas", by="short_name", asa=asa),
            )
        if quantity == "quadrupole":
            cls = Quadrupole if asa == "measurement" else _Quadrupole
            return cls(
                ac_field=self.get("ac_field", by="short_name", asa=asa),
                dc_field=self.get("dc_field", by="short_name", asa=asa),
                radiofrequency=self.get("radiofrequency", by="short_name", asa=asa),
                r_quadrupole=self.get("r_quadrupole", by="short_name", asa=asa),
            )
        entry = METADATA.get(quantity, by=by)
        value = self.config.get(entry["long_name"], DEFAULTS.get(entry["short_name"]))
        if value is None:
            return None
        unit = entry["unit"]
        if asa == "measurement" and unit is not None:
            return Q_(value, unit)
        return value


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


def get_clusters(full_config, ureg=get_application_registry()):
    clusters = []
    for particle in ["cluster", "first_product", "second_product"]:
        particle_config = full_config[particle]
        vibrational_temperatures = particle_config["vibrational_temperatures"]
        if vibrational_temperatures is None:
            vibrational_temperatures = numpy.empty(0)
        cluster = ClusterData(
            ureg.Quantity(particle_config["atomic_mass"], "amu"),
            ureg.Quantity(particle_config["electronic_energy"], "hartree"),
            particle_config["rotational_temperatures"],
            vibrational_temperatures,
        )
        clusters.append(cluster)
    return clusters


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


SKIMMER_COLUMNS = [
    "r",
    "vel",
    "T",
    "P",
    "rho",
    "speed_of_sound",
]


MaybeQuantity = Quantity | float
MaybeQuantityArray = Quantity | numpy.ndarray


class QuantityProcessor:
    def __init__(self, locals, quantities_strict=True):
        self.locals = locals
        self.quantities_strict = quantities_strict

    def __call__(self, name, unit):
        arg = self.locals[name]
        if isinstance(arg, Quantity):
            return arg.to(unit).magnitude
        elif not self.quantities_strict:
            return arg
        else:
            raise ValueError(
                f"Argument {name} (Value: {arg}) must be a pint.Quantity when `quantities_strict` is True"
            )


def skimmer(
    T0: MaybeQuantity,
    P0: MaybeQuantity,
    rmax: MaybeQuantity,
    dc: MaybeQuantity,
    alpha_factor: MaybeQuantity,
    gas: Gas | _Gas,
    N: int,
    M: int,
    resolution: int,
    tolerance: float,
    *,
    output_pandas=False,
    quantities_strict=True,
):
    process_arg = QuantityProcessor(locals(), quantities_strict)
    T0 = process_arg("T0", "kelvin")
    P0 = process_arg("P0", "pascal")
    rmax = process_arg("rmax", "meters")
    dc = process_arg("dc", "meters")
    alpha_factor = process_arg("alpha_factor", "halfturn")
    if isinstance(gas, Gas):
        gas = gas.into_cpp()
    out = _skimmer(T0, P0, rmax, dc, alpha_factor, gas, N, M, resolution, tolerance)
    if output_pandas:
        return DataFrame(out, columns=SKIMMER_COLUMNS)
    else:
        return out


def densityandrate(
    cluster_0: ClusterData,
    cluster_1: ClusterData,
    cluster_2: ClusterData,
    energy_max: MaybeQuantity,
    energy_max_rate: MaybeQuantity,
    bin_width: MaybeQuantity,
    fragmentation_energy: MaybeQuantity | None = None,
    *,
    quantities_strict=True,
):
    process_arg = QuantityProcessor(locals(), quantities_strict)
    energy_max = process_arg("energy_max", "kelvin")
    energy_max_rate = process_arg("energy_max_rate", "kelvin")
    bin_width = process_arg("bin_width", "kelvin")
    if fragmentation_energy is None:
        fragmentation_energy = 0
    else:
        fragmentation_energy = process_arg("fragmentation_energy", "kelvin")
    density_cluster, rate_const = _densityandrate(
        cluster_0.into_cpp(),
        cluster_1.into_cpp(),
        cluster_2.into_cpp(),
        energy_max,
        energy_max_rate,
        bin_width,
        fragmentation_energy,
    )
    return Histogram.from_cpp(density_cluster), Histogram.from_cpp(rate_const)


def pinhole(
    cluster_0: ClusterData,
    cluster_1: ClusterData,
    cluster_2: ClusterData,
    gas: Gas,
    density_cluster: Histogram,
    rate_const: Histogram,
    skimmer: numpy.ndarray,
    lengths: MaybeQuantityArray,
    voltages: MaybeQuantityArray,
    T: MaybeQuantity,
    pressure_first: MaybeQuantity,
    pressure_second: MaybeQuantity,
    N: int,
    *,
    mesh_skimmer: float | None = None,
    quadrupole: Quadrupole | None = None,
    cluster_charge_sign: int = 1,
    fragmentation_energy: MaybeQuantity | None = None,
    seed: int = 42,
    log_callback: Callable[[int, str], None] | None = None,
    result_callback: Callable[[numpy.ndarray], None] | None = None,
    quantities_strict=True,
):
    process_arg = QuantityProcessor(locals(), quantities_strict)
    lengths = process_arg("lengths", "meters")
    voltages = process_arg("voltages", "volts")
    T = process_arg("T", "kelvin")
    pressure_first = process_arg("pressure_first", "pascals")
    pressure_second = process_arg("pressure_second", "pascals")
    if skimmer.shape[1] == 3:
        if mesh_skimmer is None:
            raise ValueError("mesh_skimmer must be supplied when 3 column array is given for skimmer")
    elif skimmer.shape[1] == 6:
        if mesh_skimmer is not None:
            raise ValueError("mesh_skimmer should not be supplied when 6 column array is given for skimmer")
        mesh_skimmer = float(skimmer[1, 0] - skimmer[0, 0])
        skimmer = skimmer[:, 1:4]
    else:
        raise ValueError("skimmer must have 3 or 6 columns")
    return _pinhole(
        cluster_0.into_cpp(),
        cluster_1.into_cpp(),
        cluster_2.into_cpp(),
        gas.into_cpp(),
        density_cluster.into_cpp(),
        rate_const.into_cpp(),
        skimmer,
        mesh_skimmer,
        lengths,
        voltages,
        T,
        pressure_first,
        pressure_second,
        N,
        fragmentation_energy=fragmentation_energy,
        quadrupole=quadrupole and quadrupole.into_cpp(),
        cluster_charge_sign=cluster_charge_sign,
        seed=seed,
        log_callback=log_callback,
        result_callback=result_callback,
    )
