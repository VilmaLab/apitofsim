from os import environ
import sys
import pickle
import duckdb
from apitofsim import (
    skimmer,
    densityandrate,
    pinhole,
    ClusterData,
    ProductsCluster,
    KTotalInput,
    compute_density_of_states_batch,
    compute_k_total_batch,
    FragmentationPathway,
)
from apitofsim.db import ClusterDatabase
from timeit import default_timer as timer
import pint
from apitofsim.api import Histogram


ureg = pint.UnitRegistry()
Q_ = ureg.Quantity

db = ClusterDatabase(sys.argv[1] + ".duckdb")
with open(sys.argv[1] + ".pkl", "rb") as inf:
    config = pickle.load(inf)

cluster_indexed, name_lookup = db.clusters_objects_indexed(include_name_lookup=True)
print()

print("Running skimmer")
skimmer_np = skimmer(
    T0=config["T"],
    P0=config["pressure_first"],
    rmax=config["lengths"][-1],
    dc=config["dc"],
    alpha_factor=config["alpha_factor"],
    gas=config["gas"],
    N=config["N_iter"],
    M=config["M_iter"],
    resolution=config["resolution"],
    tolerance=config["tolerance"],
)
print("Done")

num_pathways = 0
density_of_states_inputs = []
for cluster, _, _ in db.pathways_objs(indexed=cluster_indexed):
    density_of_states_inputs.append(cluster)
    num_pathways += 1

for _, product1, product2 in db.pathways_objs(indexed=cluster_indexed):
    density_of_states_inputs.append(ProductsCluster(product1, product2))

print("Computing density of states")
start = timer()

density_of_states = compute_density_of_states_batch(
    density_of_states_inputs,
    energy_max=config["energy_max"],
    bin_width=config["bin_width"],
)
print(f"Done in {timer() - start}s")

print("Got")
print(density_of_states)
cluster_dos = density_of_states[:, :num_pathways]
product_dos = density_of_states[:, num_pathways:]

k_total_inputs = []
for idx, (cluster, product1, product2) in enumerate(
    db.pathways_objs(indexed=cluster_indexed)
):
    k_total_inputs.append(
        KTotalInput(
            product1.into_cpp(),
            product2.into_cpp(),
            FragmentationPathway(
                cluster.into_cpp(), product1.into_cpp(), product2.into_cpp()
            ).fragmentation_energy_kelvin(),
            cluster_dos[:, idx],
            product_dos[:, idx],
        )
    )

print(f"Compute k total on {len(k_total_inputs)} inputs with mesh_mode=1")
start = timer()

k_rates = compute_k_total_batch(
    k_total_inputs,
    energy_max_rate=config["energy_max_rate"],
    bin_width=config["bin_width"],
    mesh_mode=1,
)

print(f"Done in {timer() - start}")
print("Got")
print(k_rates)

failures = 0
for (cluster_id, product1_id, product2_id), rate_const, density_cluster in zip(
    db.pathways_ids(),
    k_rates.T,
    cluster_dos.T,
):
    cluster = cluster_indexed[cluster_id]
    product1 = cluster_indexed[product1_id]
    product2 = cluster_indexed[product2_id]
    print(
        f"{name_lookup[cluster_id]} -> {name_lookup[product1_id]} + {name_lookup[product2_id]}"
    )
    density_hist = Histogram.from_mesh(
        config["bin_width"],
        config["energy_max"],
        density_cluster,
    )
    rate_hist = Histogram.from_mesh(
        config["bin_width"],
        config["energy_max_rate"],
        rate_const,
    )
    try:
        result = pinhole(
            cluster,
            product1,
            product2,
            config["gas"],
            density_hist,
            rate_hist,
            skimmer_np,
            config["lengths"],
            config["voltages"],
            config["T"],
            config["pressure_first"],
            config["pressure_second"],
            int(environ["N_OVERRIDE"]) if "N_OVERRIDE" in environ else config["N"],
            quadrupole=config.get("quadrupole"),
            fragmentation_energy=config.get("fragmentation_energy"),
            cluster_charge_sign=config.get("cluster_charge_sign", -1),
            sample_mode=2,
            loglevel=0,
        )
    except Exception as e:
        failures += 1
        print(f"Error in simulation: {e}")
        continue
    print(f"Fragmented: {result[1]}, Escaped: {result[2]}")

print(f"Total failures: {failures} out of {num_pathways} pathways")
