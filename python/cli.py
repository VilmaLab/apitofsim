import argparse
import sys
import os
import numpy
from apitofsim import (
    read_histogram,
    read_skimmer,
    skimmer_pandas,
    parse_config_with_particles,
    config_to_shortnames,
    get_clusters,
    Gas,
    densityandrate,
    pinhole,
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
        clusters = get_clusters(full_config)
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
    if args.command == "apitof_pinhole" or args.command is None:
        clusters = get_clusters(full_config)
        gas = Gas(radius=config["R_gas"], mass=config["m_gas"])
        density_cluster = read_histogram(config["output_file_density_cluster"])
        rate_constant = read_histogram(config["output_file_rate_constant"])
        skimmer, mesh_skimmer = read_skimmer(config["Output_file_skimmer"])

        def log_callback(type, message):
            # print(type, message, end="")
            pass

        def result_callback(counters):
            print(counters)

        print("pinhole")

        counters = pinhole(
            *clusters,
            gas,
            config["bonding_energy"],
            density_cluster,
            rate_constant,
            skimmer,
            mesh_skimmer,
            *(
                config[setting]
                for setting in (
                    "cluster_charge_sign",
                    "L0",
                    "Lsk",
                    "L1",
                    "L2",
                    "L3",
                    "V0",
                    "V1",
                    "V2",
                    "V3",
                    "V4",
                    "T",
                    "pressure_first",
                    "pressure_second",
                    "r_quadrupole",
                    "radiofrequency",
                    "dc_field",
                    "ac_field",
                    "N",
                )
            ),
            42,
            None,
            result_callback,
        )
        print("Final counters:", counters)


if __name__ == "__main__":
    main()
