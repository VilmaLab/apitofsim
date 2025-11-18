import argparse
import sys
import os
import numpy
from apitofsim.api import (
    skimmer,
    densityandrate,
    pinhole,
)
from apitofsim.config import (
    read_histogram,
    read_skimmer,
    parse_config_with_particles,
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
            config.get("T", by="short_name"),  # pyright: ignore [reportArgumentType]
            config.get("pressure_first", by="short_name"),  # pyright: ignore [reportArgumentType]
            config.get("Lsk", by="short_name"),  # pyright: ignore [reportArgumentType]
            config.get("dc", by="short_name"),  # pyright: ignore [reportArgumentType]
            config.get("alpha_factor", by="short_name"),  # pyright: ignore [reportArgumentType]
            config.get("gas", by="short_name"),  # pyright: ignore [reportArgumentType]
            config.get("N_iter", by="short_name"),  # pyright: ignore [reportArgumentType]
            config.get("M_iter", by="short_name"),  # pyright: ignore [reportArgumentType]
            config.get("resolution", by="short_name"),  # pyright: ignore [reportArgumentType]
            config.get("tolerance", by="short_name"),  # pyright: ignore [reportArgumentType]
            output_pandas=True,
        )
        print(skimmer_df)
    if args.command == "densityandrate" or args.command is None:
        clusters = get_clusters(parse_config_with_particles(args.config))
        rhos, k_rate = densityandrate(
            *clusters,
            config.get("energy_max", by="short_name"),  # pyright: ignore [reportArgumentType]
            config.get("energy_max_rate", by="short_name"),  # pyright: ignore [reportArgumentType]
            config.get("bin_width", by="short_name"),  # pyright: ignore [reportArgumentType]
            config.get("bonding_energy", by="short_name"),  # pyright: ignore [reportArgumentType]
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
        skimmer_data = read_skimmer(config_dict["config"]["Output_file_skimmer"])
        if skimmer_data is None:
            raise ValueError("Skimmer file is empty")
        skimmer_data, mesh_skimmer = skimmer_data

        def log_callback(type, message):
            # print(type, message, end="")
            pass

        def result_callback(counters):
            print(counters)

        counters = pinhole(
            *clusters,
            config.get("gas"),  # pyright: ignore [reportArgumentType]
            density_cluster,  # pyright: ignore [reportArgumentType]
            rate_constant,  # pyright: ignore [reportArgumentType]
            skimmer_data,
            config.get("lengths"),  # pyright: ignore [reportArgumentType]
            config.get("voltages"),  # pyright: ignore [reportArgumentType]
            config.get("T"),  # pyright: ignore [reportArgumentType]
            config.get("pressure_first"),  # pyright: ignore [reportArgumentType]
            config.get("pressure_second"),  # pyright: ignore [reportArgumentType]
            config.get("N"),  # pyright: ignore [reportArgumentType]
            mesh_skimmer=mesh_skimmer,
            quadrupole=config.get("quadrupole"),  # pyright: ignore [reportArgumentType]
            fragmentation_energy=config.get("bonding_energy") or None,  # pyright: ignore [reportArgumentType]
            cluster_charge_sign=config.get("cluster_charge_sign") or 1,  # pyright: ignore [reportArgumentType]
            seed=42,
            log_callback=None,
            result_callback=result_callback,
        )
        print("Final counters:", counters)


if __name__ == "__main__":
    main()
