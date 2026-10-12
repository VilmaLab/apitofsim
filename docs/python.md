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

### Recorded realizations and events

`RealizationDatabase` provides DuckDB relations for querying recorded output.
`results_query()`, `realizations_query()`, `events_query()` and
`recorded_output_query()` accept optional `run_id`, `parent` (cluster ID) and
`result_id` filters. Supplied filters are intersected; omitting all filters selects
the current database. Both single- and multi-pathway results are supported.
Database scope includes realizations that have not yet been linked to a result;
run, parent and result scopes include only linked realizations.
The queries use the `experiment_result` and `event_details` views. For existing
files, refresh the views as described above before using these APIs.

```python
from apitofsim.workflow.db import RealizationDatabase, connection_scope

with connection_scope(RealizationDatabase, "output.duckdb", readonly=True) as db:
    events = db.events_query(run_id=1, parent=3, sort=True).fetchdf()
    diagnostics = db.recorded_output_query(run_id=1, parent=3).fetchone()
```

Relations can also be consumed with `.pl()` (with Polars installed) or
`.fetchall()`, or further filtered and projected before fetching. Fetch results
before closing the database. `events_query(sort=True)` orders by realization ID,
simulation time and event ID, including deterministic ordering for time ties.
Events retain raw state structs, event and cohort IDs, and optional subtype fields.
The flattened `x`, `y`, `z` and `t` columns from `event_details` are also available.

Diagnostics return one row, including for empty cohorts (all counts are zero):

| Column | Recorded quantity |
| --- | --- |
| `realization_count` | Realization rows, including those with no events |
| `event_count` | Event-info rows |
| `escape_count` | Escape event-info rows |
| `collision_event_count` | Collision event-info rows |
| `collision_subtype_count` | Joined collision subtype rows |
| `fragmentation_event_count` | Fragmentation event-info rows |
| `fragmentation_subtype_count` | Joined fragmentation subtype rows |
| `terminal_realization_count` | Distinct realizations containing escape or fragmentation event-info rows |

These counts describe recorded output rather than simulation counters. Missing
subtypes remain visible through separate counts and nullable `collision_event_id`
and `fragmentation_event_id` columns in the event query. A present fragmentation
subtype may have a null `pathway_id`, so that field alone cannot detect missing
subtype records. Terminal realization counts do not require subtype records and
count a realization with multiple terminal events only once.

::: apitofsim.workflow.RealizationDatabase
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
