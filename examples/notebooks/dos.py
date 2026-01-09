import marimo

__generated_with = "0.18.4"
app = marimo.App()


@app.cell
def _():
    import matplotlib.pyplot as plt
    import sys
    import pickle
    import duckdb
    from apitofsim import (
        skimmer,
        densityandrate,
        mass_spec,
        ClusterData,
        ProductsCluster,
        compute_density_of_states_batch,
        compute_k_total_batch,
    )
    from timeit import default_timer as timer
    import pint
    from os import makedirs, environ
    import numpy as np
    from apitofsim.db import ClusterDatabase
    from pint import get_application_registry
    import marimo as mo
    return (
        ClusterDatabase,
        compute_density_of_states_batch,
        environ,
        get_application_registry,
        mo,
        np,
        plt,
        timer,
    )


@app.cell
def _(ClusterDatabase, environ, get_application_registry, np):
    ureg = get_application_registry()
    db = ClusterDatabase(environ["DATABASE"])

    clusters, name_lookup = db.clusters_objects_indexed(include_name_lookup=True)
    id_to_index = {id: idx for idx, id in enumerate(clusters.keys())}

    energy_max = 1e5 * ureg.K
    bin_width = 1 * ureg.K
    n_bins = int(energy_max / bin_width)
    x = np.arange(0, n_bins) * bin_width.magnitude + (bin_width.magnitude / 2)
    boltzmann = 1.38064852e-23
    kT = 300 * boltzmann
    x_scaled = x * boltzmann
    return (
        bin_width,
        boltzmann,
        clusters,
        energy_max,
        id_to_index,
        kT,
        name_lookup,
        x,
        x_scaled,
    )


@app.cell
def _(bin_width, clusters, compute_density_of_states_batch, energy_max, timer):
    def compute_dos(use_old_impl):
        start = timer()
        result = compute_density_of_states_batch(
            clusters.values(), energy_max=energy_max, bin_width=bin_width, use_old_impl=use_old_impl
        )
        interval = timer() - start
        print(
            f"Took {interval:.2f}s for {len(clusters)} clusters; {(interval / len(clusters)):.2f}s per cluster"
        )
        return result
    return (compute_dos,)


@app.cell
def _(compute_dos):
    all_old_dos = compute_dos(use_old_impl=True);
    return (all_old_dos,)


@app.cell
def _(compute_dos):
    all_new_dos = compute_dos(use_old_impl=False);
    return (all_new_dos,)


@app.cell
def _(mo, name_lookup):
    cluster_dropdown = mo.ui.dropdown(
        options={v: k for k, v in name_lookup.items()},
        label="Pick a cluster",
        searchable=True,
    )
    cluster_dropdown
    return (cluster_dropdown,)


@app.cell
def _(
    all_new_dos,
    all_old_dos,
    boltzmann,
    cluster_dropdown,
    id_to_index,
    kT,
    mo,
    name_lookup,
    np,
    plt,
    x,
    x_scaled,
):
    cur_cluster_id = cluster_dropdown.value
    mo.stop(not cur_cluster_id, "Select a cluster above to see the plot here")
    cur_cluster_index = id_to_index[cur_cluster_id]
    cur_cluster_name = name_lookup[cur_cluster_id]

    fig, ax = plt.subplots(2, 2)
    fig.suptitle(cur_cluster_name)
    old_dos = all_old_dos[:, cur_cluster_index]
    new_dos = all_new_dos[:, cur_cluster_index]

    def plot_dos(ax, title, x, dos):
        ax.set_title(title)
        ax.plot(x, dos)

    def plot_prob(ax, title, x, dos):
        dos_scaled = dos / boltzmann
        prob = dos_scaled * np.exp((-x_scaled / kT))
        prob = prob / prob.sum()
        ax.plot(x_scaled[:5000], prob[:5000])

    plot_dos(ax[0, 0], "Old DOS", x, old_dos)
    plot_dos(ax[0, 1], "New DOS", x, new_dos)
    plot_prob(ax[1, 0], "Old prob", x, old_dos)
    plot_prob(ax[1, 1], "New prob", x, new_dos)
    fig
    return


if __name__ == "__main__":
    app.run()
