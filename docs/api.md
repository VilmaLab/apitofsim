# Using the Python API

This page documents the Python API. There is

# Main API

The main API is in the `apitofsim.api` module, re-exported from `apitofsim`.

## Data classes

These classes hold data to be

## Simulation functions

::: apitofsim.pinhole

::: apitofsim.densityandrate

::: apitofsim.skimmer

# Database

The `apitofsim.db` module, contains functions to keep cluster data in a database, convenient for running scaled-up simulations.

::: apitofsim.db.ClusterDatabase
    members: true

# Workflow example with Python

The following example shows how to run a full simulation workflow using Python scripts.

--8<-- "examples/python-workflow/README.md"

``` title="prepare.py"
--8<-- "examples/python-workflow/prepare.py"
```

``` title="run.py"
--8<-- "examples/python-workflow/run.py"
```
