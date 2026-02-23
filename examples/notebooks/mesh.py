import marimo

__generated_with = "0.19.4"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _():
    from apitofsim.apitofsimraw import precompute_mesh, MeshMode
    from numpy import arange
    import matplotlib.pyplot as plt
    import time
    import marimo as mo

    return MeshMode, arange, mo, plt, precompute_mesh, time


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Mesh mode

    This script benchmarks the different mesh modes for precompute_mesh
    """)
    return


@app.cell
def _(MeshMode, arange, precompute_mesh, time):
    modes = [mode for mode in MeshMode if mode != MeshMode.no_mesh]
    runtime_data = {mode: [] for mode in modes}
    energy_range = arange(1, 5.25, 0.25)

    def run_bench():
        for e in energy_range:
            for mesh_mode in modes:
                start_time = time.time()
                precompute_mesh(10**e, 1, mesh_mode=mesh_mode)
                elapsed_time = time.time() - start_time
                runtime_data[mesh_mode].append(elapsed_time)
                print(mesh_mode, e, elapsed_time)

    run_bench()
    return energy_range, runtime_data


@app.cell
def _(energy_range, plt, runtime_data):
    # Plot the runtime comparison
    plt.figure()
    for mesh_mode, runtimes in runtime_data.items():
        plt.plot(energy_range, runtimes, label=f"Mesh mode {mesh_mode}")

    plt.xlabel("Energy (10^e)")
    plt.ylabel("Runtime (seconds)")
    plt.title("Runtime Comparison for Mesh Modes")
    plt.legend()
    plt.grid(True)
    plt.show()
    return


@app.cell
def _():
    import marimo as mo

    return (mo,)


if __name__ == "__main__":
    app.run()
