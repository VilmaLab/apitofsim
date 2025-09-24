import argparse
import sys
import os
import numpy
from apitofsim import (
    skimmer_pandas,
    parse_config_with_particles,
    config_to_shortnames,
    ClusterData,
    Gas,
    densityandrate,
)


def main():
    parser = argparse.ArgumentParser(description="APITOF Simulation CLI")
    parser.add_argument(
        "command",
        nargs="?",
        choices=["skimmer", "densityandrate", "apitof_pinhole"],
        help="Command to execute",
    )
    parser.add_argument("config", help="Path to the configuration file")
    parser.add_argument(
        "-C",
        "--chdir",
        help="Change the working directory before executing",
        default=None,
    )
    args = parser.parse_args()

    if args.chdir:
        try:
            os.chdir(args.chdir)
        except FileNotFoundError:
            print(f"Error: The directory {args.chdir} does not exist.", file=sys.stderr)
            sys.exit(1)
        except PermissionError:
            print(
                f"Error: Permission denied to change to directory {args.chdir}.",
                file=sys.stderr,
            )
            sys.exit(1)

    full_config = parse_config_with_particles(args.config)
    config = config_to_shortnames(full_config["config"])
    from pprint import pprint

    pprint(full_config)

    if args.command == "skimmer" or args.command is None:
        skimmer_df = skimmer_pandas(
            *(
                config[c]
                for c in (
                    "T",
                    "pressure_first",
                    "Lsk",
                    "dc",
                    "alpha_factor",
                    "m_gas",
                    "ga",
                    "N_iter",
                    "M_iter",
                    "resolution",
                    "tolerance",
                )
            )
        )
        print(skimmer_df)
    if args.command == "densityandrate" or args.command is None:
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
        gas = Gas(config["R_gas"], config["m_gas"])
        rhos, k_rate = densityandrate(
            *clusters,
            *(
                config[setting]
                for setting in (
                    "energy_max",
                    "energy_max_rate",
                    "bin_width",
                    "bonding_energy",
                )
            ),
        )
        numpy.set_printoptions(threshold=sys.maxsize)
        print("densities")
        print(rhos)
        print("k_rate")
        print(k_rate)


if __name__ == "__main__":
    main()
