These notebooks explore some of the internals of apitofsim.

* `gas-cluster-collision-sampler.py`: This notebook plots the distribution of collision angle and normal velocity of the cluster and the gas, shows how the rejection sampler works, and compares it to other sampling methods.
* `dos.py`: This compares different algorithms for computing the density of states (DOS) of a cluster in apitofsim.

## Running the notebooks with Mamba

The recommended way to run the notebooks is using Mamba.
This will use a version of apitofsim from a package.

First [install miniforge according to the instructions](https://github.com/conda-forge/miniforge?tab=readme-ov-file#install).
Then create the environment:

```bash
 $ mamba create -p ./cenv -f env.yaml
```

Then you can run a notebook using:

```bash
 $ mamba run -p ./cenv marimo run gas-cluster-collision-sampler.py
```

## Running the notebooks with UV

You can also install the requirements for the notebooks with uv.
This will compile and use the version of apitofsim you have checked out.

First [install uv according to the instructions](https://docs.astral.sh/uv/getting-started/installation/).

Assuming you have clone `apitofsim` and are in the root directory:

```bash
 $ uv sync --all-groups
```

You can then run a notebook using:

```bash
 $ uv run marimo edit gas-cluster-collision-sampler.py
```
