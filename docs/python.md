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

### Database read queries

`SuperClusterDatabase.available_dos_histograms_query()` returns available DOS
configuration IDs (`histogram_params_id`) and `bin_width` in kelvin, including
both cluster and products DOS. `available_rate_histograms_query()` returns the
same columns for configurations used by `k_rate`. Each configuration appears
once, even when several configurations share a bin width. These are independent
sets with no implied pairing or ordering; callers choose compatible configurations.

`ExperimentDatabase.experiment_overview_query(experiment_run_id=None)` returns
the `experiment_summary` columns plus raw `config` and `run_config` JSON. Omit
the run ID to retrieve all runs. Timestamps remain Python datetimes when fetched.
`experiment_clusters_query(experiment_run_id, cluster_id=None)` returns rows from
`experiment_cluster_report`, including cluster facts and `is_single_pathway`,
optionally filtered to one cluster. Mode flags retain the report views' semantics:
they indicate whether the run has single-pathway results, so a run without
results has a false flag; its requested simulation mode remains in `run_config`.

All four methods return DuckDB relations that callers can select, filter, sort
or fetch. Missing run or cluster IDs return empty relations. Display labels and
date formatting belong to callers.

### Result and event views

`experiment_result` contains one row per successful result, with `id`,
`experiment_run_id`, resolved parent `cluster_id`, optional `pathway_id`, and
`is_single_pathway`. Multi-pathway results have a null `pathway_id`; failures
are excluded.

Realization databases also provide `event_details`, which retains every
`event_info` row and left-joins optional collision and fragmentation fields.
It includes the raw `postime` struct and its `x`, `y`, `z`, `t` fields, with
coordinates and physical quantities in the recorder's original units. It
retains initial events and events whose realization has no successful result.
`event_report` joins these events to successful results and parent clusters,
preserving its existing columns, including the fragmentation `pathway_id`.
Neither view clamps coordinates or assigns display labels or regions;
`event_details` has no display ordering.

New databases receive these views through `create_tables()`. For existing
writable files, use `refresh_views()` or `apitofsim refresh-views DATABASE`.
The SQL continues to resolve tables through the connection's search path,
without fixed attachment aliases.

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
