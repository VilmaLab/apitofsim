import argparse
import sys
import os
import numpy
from apitofsim.api import (
    skimmer,
    densityandrate,
    pinhole,
    Gas,
)
from apitofsim.config import (
    read_histogram,
    read_skimmer,
    parse_config_with_particles,
    config_to_shortnames,
    ConfigFile,
    get_clusters,
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

    config = ConfigFile(filename=args.config)

    if args.command == "skimmer" or args.command is None:
        skimmer_df = skimmer(
            *(
                config.get(c, by="short_name")
                for c in (
                    "T",
                    "pressure_first",
                    "Lsk",
                    "dc",
                    "alpha_factor",
                    "gas",
                    "N_iter",
                    "M_iter",
                    "resolution",
                    "tolerance",
                )
            ),
            output_pandas=True,
        )
        print(skimmer_df)
    if args.command == "densityandrate" or args.command is None:
        clusters = get_clusters(parse_config_with_particles(args.config))
        rhos, k_rate = densityandrate(
            *clusters,
            *(
                config.get(setting, by="short_name")
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
        config_dict = parse_config_with_particles(args.config)
        clusters = get_clusters(config_dict)
        density_cluster = read_histogram(
            config_dict["config"]["output_file_density_cluster"]
        )
        rate_constant = read_histogram(
            config_dict["config"]["output_file_rate_constant"]
        )
        skimmer_data, mesh_skimmer = read_skimmer(
            config_dict["config"]["Output_file_skimmer"]
        )

        def log_callback(type, message):
            # print(type, message, end="")
            pass

        def result_callback(counters):
            print(counters)

        counters = pinhole(
            *clusters,
            config.get("gas"),
            density_cluster,
            rate_constant,
            skimmer_data,
            config.get("lengths"),
            config.get("voltages"),
            config.get("T"),
            config.get("pressure_first"),
            config.get("pressure_second"),
            config.get("N"),
            mesh_skimmer=mesh_skimmer,
            quadrupole=config.get("quadrupole"),
            fragmentation_energy=config.get("bonding_energy") or None,
            cluster_charge_sign=config.get("cluster_charge_sign") or 1,
            seed=42,
            log_callback=None,
            result_callback=result_callback,
        )
        print("Final counters:", counters)


if __name__ == "__main__":
    main()
