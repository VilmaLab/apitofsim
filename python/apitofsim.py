import os
import numpy
import json
from pandas import DataFrame

from _apitofsim import (
    skimmer as _skimmer,
    ClusterData,
    Gas,
    densityandrate,
    Histogram,
    pinhole,
)


__all__ = [
    # Imports
    "ClusterData",
    "Gas",
    "densityandrate",
    "pinhole",
    "skimmer_numpy",
    "skimmer_pandas",
    "parse_config_with_particles",
    "config_to_shortnames",
    "read_dat",
    "read_histogram",
    "read_skimmer",
    "get_clusters",
]


def read_dat(fn):
    if os.stat(fn).st_size == 0:
        return None
    return numpy.asfortranarray(numpy.loadtxt(fn, dtype=numpy.float64))


def read_histogram(fn):
    arr = read_dat(fn)
    if arr is None:
        return None
    return Histogram(arr[:, 0], arr[:, 1])


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


def get_particle(config, particle):
    particle_data = {}
    for quantity in [
        "vibrational_temperatures",
        "rotational_temperatures",
        "electronic_energy",
    ]:
        config_key = f"file_{quantity}_{particle}"
        particle_data[quantity] = read_dat(config[config_key])
        particle_data["name"] = config[config_key].rsplit(".", 1)[0].rsplit("/", 1)[-1]
    particle_data["atomic_mass"] = config[f"Atomic_mass_{particle}"]
    return particle_data


def parse_config_with_particles(fn):
    config = parse_config(fn)
    result = {"config": config}
    for particle in ["cluster", "first_product", "second_product"]:
        result[particle] = get_particle(config, particle)
    return result


def get_clusters(full_config):
    clusters = []
    for particle in ["cluster", "first_product", "second_product"]:
        particle_config = full_config[particle]
        vibrational_temperatures = particle_config["vibrational_temperatures"]
        if vibrational_temperatures is None:
            vibrational_temperatures = numpy.empty(0)
        cluster = ClusterData(
            particle_config["atomic_mass"],
            float(particle_config["electronic_energy"]),
            particle_config["rotational_temperatures"],
            vibrational_temperatures,
        )
        clusters.append(cluster)
    return clusters


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


def skimmer_numpy(*args):
    return _skimmer(*args)


def skimmer_pandas(*args):
    return DataFrame(skimmer_numpy(*args), columns=SKIMMER_COLUMNS)
