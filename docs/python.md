# Using the Python API

This page documents the Python API.
The main API is in the `apitofsim.api` module, most of which is re-exported from `apitofsim`.

## Data classes

These classes input data to the simulation, as well as output data from intermediate steps in the simulation.

::: apitofsim.api.ClusterLike
    options:
      members: true
      summary: false
      heading_level: 3

::: apitofsim.ClusterData
    options:
      members: true
      summary: false
      heading_level: 3

::: apitofsim.ProductsCluster
    options:
      members: true
      summary: false
      heading_level: 3
      inherited_members: true

::: apitofsim.Gas
    options:
      members: true
      summary: false
      heading_level: 3

::: apitofsim.Histogram
    options:
      members: true
      summary: false
      heading_level: 3

::: apitofsim.Quadrupole
    options:
      members: true
      summary: false
      heading_level: 3

::: apitofsim.MassSpecInputFragmentationPathway
    options:
      heading_level: 3

::: apitofsim.MassSpecSubstanceSingleInput
    options:
      heading_level: 3

::: apitofsim.MassSpectrometer
    options:
      members: true
      summary: false
      heading_level: 3

## Workflow interface

The `apitofsim.workflow` module, contains functions to keep cluster data in a database, convenient for running scaled-up simulations.

Typically an `ExperimentDatabase` is created and its tables created with `db.create_tables(...)` and then clusters are ingested e.g. with `ingest_tree(...)`.

After this, multiple simulation can be run using the `ExperimentRunner` class.

::: apitofsim.workflow.ClusterDatabase
    options:
      members: true
      heading_level: 3
      filters: ["!^_", "^__init__$"]

::: apitofsim.workflow.SuperClusterDatabase
    options:
      members: true
      heading_level: 3
      filters: public

::: apitofsim.workflow.ExperimentDatabase
    options:
      members: true
      heading_level: 3
      filters: public

::: apitofsim.workflow.ingest_tree
    options:
      heading_level: 3

::: apitofsim.workflow.ExperimentRunner
    options:
      members: true
      heading_level: 3
      filters: ["!^_", "^__init__$"]

## Individual simulation functions

The individual simulation functions are the low level interface to the simulation.

::: apitofsim.mass_spec
    options:
      heading_level: 3

::: apitofsim.compute_density_of_states_batch
    options:
      heading_level: 3

::: apitofsim.precompute_mesh
    options:
      heading_level: 3

::: apitofsim.compute_k_total_batch
    options:
      heading_level: 3

::: apitofsim.densityandrate
    options:
      heading_level: 3

::: apitofsim.skimmer
    options:
      heading_level: 3
