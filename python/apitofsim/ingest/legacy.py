from glob import glob
from os.path import dirname, isfile, basename, expanduser
import os
import numpy
from pint import get_application_registry


ureg = get_application_registry()


def read_dat(fn):
    if os.stat(fn).st_size == 0:
        return None
    return numpy.asfortranarray(numpy.loadtxt(fn, dtype=numpy.float64))


def get_common_prefix(config, particle):
    paths = [
        config[f"file_{quantity}_{particle}"]
        for quantity in [
            "vibrational_temperatures",
            "rotational_temperatures",
            "electronic_energy",
        ]
    ]
    return os.path.commonprefix(paths).rstrip("_/.")


def get_particle(config, particle):
    particle_data = {}
    for quantity in [
        "vibrational_temperatures",
        "rotational_temperatures",
        "electronic_energy",
    ]:
        config_key = f"file_{quantity}_{particle}"
        particle_data[quantity] = read_dat(config[config_key])
    if particle_data["vibrational_temperatures"] is not None:
        particle_data["vibrational_temperatures"] = particle_data["vibrational_temperatures"] * ureg.kelvin
    if particle_data["rotational_temperatures"] is not None:
        particle_data["rotational_temperatures"] = particle_data["rotational_temperatures"] * ureg.kelvin
    particle_data["electronic_energy"] = particle_data["electronic_energy"][0] * ureg.hartree
    particle_data["atomic_mass"] = config[f"Atomic_mass_{particle}"] * ureg.amu
    return particle_data


class DotAccessDict(dict):
    __getattr__ = dict.get


def parse_legacy_one(filename, clusters):
    from contextlib import chdir
    from apitofsim.config import parse_config

    pathway = []
    with chdir(dirname(filename)):
        config = parse_config(filename)
        for particle in ["cluster", "first_product", "second_product"]:
            prefix = get_common_prefix(config, particle)
            name = prefix.split("/")[-1].rstrip("_").rstrip(".")
            sources = {}
            for source_name, source in clusters["sources"].items():
                method = source.get("type", source_name)
                if method == "dat":
                    sources[source_name] = get_particle(config, particle)
                elif method == "orca":
                    from apitofsim.ingest.orca import parse_orca
                    extension = source["append_to_common_prefix"]
                    path = prefix + extension
                    with open(path) as f:
                        orca_result = parse_orca(f)
                        if len(orca_result) != 1:
                            raise ValueError(f"Expected one structure in ORCA output {path}, got {len(orca_result)}")
                        sources[source_name] = orca_result[0]
                elif method == "gaussian":
                    from apitofsim.ingest.gaussian import parse_gaussian
                    extension = source["append_to_common_prefix"]
                    path = prefix + extension
                    with open(path) as f:
                        sources[source_name] = parse_gaussian(f)
                elif method == "map":
                    sources[source_name] = source.get(name, {})
                else:
                    raise ValueError(f"Unknown method: {method}")
            provenance = {}
            combined = {}
            for quantity in [
                "vibrational_temperatures",
                "rotational_temperatures",
                "electronic_energy",
                "atomic_mass",
                "charge",
            ]:
                use_eval = False
                if quantity in clusters:
                    source_name = clusters[quantity]
                    if "." in source_name:
                        use_eval = True
                else:
                    source_name = clusters["default_source"]
                if use_eval:
                    eval_ctx = {source_name: DotAccessDict(sources[source_name]) for source_name in sources}
                    try:
                        combined[quantity] = eval(source_name, eval_ctx)
                    except Exception as e:
                        raise ValueError(
                            f"Error evaluating expression for particle {filename} {particle} {name} ; source: {source_name} ; quantity: {quantity}"
                        ) from e
                else:
                    try:
                        combined[quantity] = sources[source_name][quantity]
                    except Exception as e:
                        raise ValueError(
                            f"Error getting quantity for particle: {filename} {particle} {name} ; source: {source_name} ; quantity: {quantity}"
                        ) from e
                provenance[quantity] = source_name
            pathway.append({
                "name": name,
                "sources": sources,
                "provenance": provenance,
                "particle": combined
            })
    return pathway


def parse_legacy_tree(path, clusters):
    filenames = glob(expanduser(path), recursive=True)
    for filename in filenames:
        yield parse_legacy_one(filename, clusters)
