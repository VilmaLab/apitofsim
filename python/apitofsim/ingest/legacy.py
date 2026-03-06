import os
from collections import namedtuple
from glob import glob
from os.path import expanduser

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
        particle_data["vibrational_temperatures"] = (
            particle_data["vibrational_temperatures"] * ureg.kelvin
        )
    if particle_data["rotational_temperatures"] is not None:
        particle_data["rotational_temperatures"] = (
            particle_data["rotational_temperatures"] * ureg.kelvin
        )
    particle_data["electronic_energy"] = (
        particle_data["electronic_energy"][0] * ureg.hartree
    )
    particle_data["atomic_mass"] = config[f"Atomic_mass_{particle}"] * ureg.amu
    return particle_data


ingest_cluster_file_info = namedtuple("ingest_cluster_file_info", "prefix")


def parse_legacy_one(filename, clusters, path_base=None):
    import pathlib
    from contextlib import chdir

    from apitofsim.config import parse_config
    from apitofsim.ingest.common import combine_sources, import_source

    pathway = []
    filename_path = pathlib.Path(filename)
    if path_base is None:
        working_dir = filename_path.parent
    else:
        working_dir = (path_base / filename_path).parent
    with chdir(working_dir):
        config = parse_config(filename)
        for particle in ["cluster", "first_product", "second_product"]:
            prefix = get_common_prefix(config, particle)
            particle_name = prefix.split("/")[-1].rstrip("_").rstrip(".")
            sources = {}
            for source_name, source in clusters["sources"].items():
                method = source.get("type", source_name)
                if method == "dat":
                    sources[source_name] = get_particle(config, particle)
                else:
                    sources[source_name] = import_source(
                        source,
                        method,
                        particle_name,
                        source_name,
                        ingest_cluster_file_info(prefix),
                        path_base,
                    )
            combined, provenance = combine_sources(sources, clusters)
            pathway.append(
                {
                    "name": particle_name,
                    "sources": sources,
                    "provenance": provenance,
                    "particle": combined,
                }
            )
    return pathway


def parse_legacy_tree(path, clusters, path_base):
    filenames = glob(expanduser(path), recursive=True)
    for filename in filenames:
        yield parse_legacy_one(filename, clusters, path_base)
