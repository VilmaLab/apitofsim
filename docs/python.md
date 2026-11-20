# 🤖 Using the Python API

This page documents the Python API.
The main API is in the `apitofsim.api` module, most of which is re-exported from `apitofsim`.

## Building simulation configurations

Use `build_config()` for a GUI or custom runner. It returns `SimulationConfig`, a
public `TypedDict` containing Pint quantities, ordered quantity arrays, `Gas`,
and (optionally) `Quadrupole`. Pass it directly to `ExperimentRunner` or
`ExperimentDatabase.insert_config()`; database reads reconstruct the objects
and quantities. Input mappings and objects are not mutated.

`CONFIG_FIELDS` is a read-only mapping from canonical field names to `ConfigField`
records. Each record exposes `unit`, `components` (an empty tuple for scalars),
and `default` (`None` when no default is supplied).
Object members use dotted names such as `gas.radius` and
`quadrupole.radiofrequency`. Array components are:

| Field | Canonical unit | Component order |
| --- | --- | --- |
| lengths | meter | L0, L1, L2, L3, Lsk |
| voltages | volt | V0, V1, V2, V3, V4 |
| pressures | pascal | first, second |

The named input types are `LengthComponents`, `VoltageComponents`,
`PressureComponents`, `GasComponents`, and `QuadrupoleComponents`. Input rules:

* Arrays may also be supplied as a Pint quantity with the exact shape `(5,)`, `(5,)`, or `(2,)`, respectively.
* Named mappings require every listed component and reject unknown keys; gas and quadrupole inputs also accept existing public objects.
* Dimensional scalars require Pint quantities; dimensionless scalars accept plain numbers or dimensionless quantities.
* Values must be finite; iteration, resolution and realization counts must be integers.
* Errors name the invalid field/component and expected unit or shape; physical bounds and form-specific validation remain the consumer's responsibility.

All arguments without defaults in `build_config()` are required. The builder
supplies no numerical tuning defaults. Omitted/`None` quadrupoles mean no mass
filter. When a quadrupole is supplied, its frequency and rod radius are required;
AC and DC voltages default to zero. Optional `bonding_energy`,
`fragmentation_energy`, and `radius_pinhole` are omitted when `None`, otherwise
retained in kelvin, kelvin, and meters. Histogram energies use temperature units;
convert energy quantities explicitly with a Pint Boltzmann context if needed.
`fragmentation_energy` is a pathway-at-a-time runner override; grouped preparation
uses electronic-energy differences. `radius_pinhole` is preserved for consumers
constructing `MassSpectrometer` directly; `ExperimentRunner` uses its default
pinhole radius of 1 mm.

Geometry follows the simulation's definitions: `L0` is the first chamber length,
`L1` is the gap after the skimmer and before the quadrupole, `L2` is the quadrupole
length, `L3` is the remaining second chamber length, and `Lsk` is the axial
skimmer region length. `dc` is the radius at the smallest skimmer cross section.
The flow calculation uses the expanding radius `dc + z * tan(alpha)`, where
`alpha = alpha_factor * pi`; `alpha_factor` is normalized to `halfturn`.
For example, 45 degrees becomes 0.25 halfturn. This angle is the slope angle
used by that formula. See [configuration geometry](configuration.md#top-level-arrays)
for voltage locations and diagrams.

This example uses only public APIs:

```python
from pint import get_application_registry
from apitofsim import CONFIG_FIELDS, SimulationConfig, build_config
from apitofsim.workflow import ExperimentRunner

Q_ = get_application_registry().Quantity
config: SimulationConfig = build_config(
    lengths=dict(L0=Q_(1, "mm"), L1=Q_(2.44, "mm"), L2=Q_(10.1, "cm"),
                 L3=Q_(4.48, "mm"), Lsk=Q_(0.5, "mm")),
    voltages=dict(V0=Q_(-19, "V"), V1=Q_(-9, "V"), V2=Q_(-7, "V"),
                  V3=Q_(-6, "V"), V4=Q_(11, "V")),
    pressures=dict(first=Q_(194, "Pa"), second=Q_(3.88, "Pa")),
    gas=dict(radius=Q_(1.84, "angstrom"), mass=Q_(4.65e-26, "kg"),
             adiabatic_index=1.4),
    quadrupole=dict(ac_field=Q_(200, "V"), radiofrequency=Q_(1.3, "MHz"),
                    r_quadrupole=Q_(6, "mm")),
    T=Q_(300, "K"), dc=Q_(0.5, "mm"), alpha_factor=Q_(45, "degree"),
    N=1000, N_iter=1000, M_iter=1000, resolution=1000, tolerance=1e-8,
    bin_width=Q_(1, "K"), energy_max=Q_(200_000, "K"),
    energy_max_rate=Q_(100_000, "K"),
)
assert CONFIG_FIELDS["lengths"].components == ("L0", "L1", "L2", "L3", "Lsk")
# db is an existing ExperimentDatabase with clusters and pathways already ingested.
config_id = db.insert_config("custom", config)
runner = ExperimentRunner(db)
runner.start_run(config_id)
prepared = runner.prepare_from_config(config)
# Use runner.iter_mass_spec() as shown in the recording example below.
```

Legacy `ConfigFile` readers remain available, and stored configurations may still
be partial. `ConfigFile.into_json_config()` preserves quadrupole settings.

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

The `apitofsim.workflow` module stores cluster data in a database and runs scaled-up simulations.

Typically an `ExperimentDatabase` is created with `db.create_tables(...)` and clusters are ingested e.g. with `ingest_tree(...)`. Simulations are then run with the `ExperimentRunner` class.

`db.create_tables()` omits foreign keys by default while preserving primary keys,
other constraints and report views. Use `db.create_tables(foreign_keys=True)` to
enable foreign keys when creating a database.
Before using reports, call `db.check_foreign_keys()` to get orphan counts for
violated declared relationships (an empty dictionary means none); it supports
nullable/composite keys and accessible attached cluster tables.

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
optionally filtered to one cluster. Mode flags indicate whether the run has
single-pathway results (false for a run without results); the requested simulation
mode is in `run_config`.

All four methods return DuckDB relations that callers can select, filter, sort
or fetch. Missing run or cluster IDs return empty relations. Display labels and
date formatting belong to callers.

### Event storage and read queries

`ExperimentDatabase` stores configuration, tree topology, summaries and failure
information. Native event payloads are stored in `results.events.arrow` or
`results.events.parquet` alongside `results.duckdb`; the optional ASE sidecar keeps
its `results.ase.sqlite.db` naming. Move the files together with matching
basenames; an archive UUID and committed length detect mismatched sidecars.

Event payloads are read through a PyArrow IPC dataset or DuckDB's Parquet scan
according to the archive format; there are no physical `event_info`,
`realization`, `collision_event` or `fragmentation_event` tables. `event_execution`
contains one metadata row per execution with its result ID, status, event count and
batch range. Status is `running`, `complete`, `cancelled` or `failed`;
whole-experiment status is also persisted in `experiment_run.execution_status`.
Incomplete executions are never inserted into successful-result aggregates.

`results_query()`, `realizations_query()`, `events_query()` and
`recorded_output_query()` return DuckDB relations. Filters `run_id`, `parent`
(root cluster ID) and `result_id` intersect. Scoped event and realization queries
read only the IPC batches for matching executions or push result ID filters
into the Parquet scan. Failed/cancelled executions retain
their recorded events. Realization counts count observed histories, including
histories interrupted before a completed-realization counter was emitted.

```python
from apitofsim.workflow import ExperimentDatabase

with ExperimentDatabase("results.duckdb", readonly=True) as db:
    events = db.events_query(run_id=3, parent=17, sort=True).fetchdf()
    histories = db.fragmentation_histories_query(run_id=3).fetchdf()
```

To record through the native carquet writer, use
`ExperimentDatabase("results.duckdb", recording_format="parquet",
recording_compression="zstd", parquet_column_workers=4)` and request
`record_events=True` when running. A Parquet sidecar starts in a database without
recorded events and is immutable after publication. Use
`db.recording_transaction()` to group multiple runs into one sidecar. Standalone
`record_to=` recording uses Arrow IPC. `read_events()` accepts either format.


Every event has numeric `result_id`, `realization_index` and `event_index`.
The latter is monotonically increasing within one realization, including tied
timestamps. These three fields are the identity; ordering between realizations
is not stable. SQL's convenience `event_id` and `realization_id` are opaque
colon-separated strings of their components. Sort by the numeric components,
not the textual IDs. `events_query(sort=True)` does this automatically.

`event_details`, `event_info`, `realization` and `event_report` are temporary
read views created by the database wrapper; `event_details` exposes position/time,
velocity, energies, node identity and any collision/fragmentation fields. No event
associations are inferred from timestamps. Event types and their fields are listed
under [Native recording and Python event delivery](#native-recording-and-python-event-delivery).

Sidecar access is coordinated with locks:

* Readers hold a shared lock; a recording transaction requires exclusive access, so close other readers before appending.
* Event queries on the writing connection are unavailable during an append; use live Python delivery instead. Finish existing scans first, and do not execute retained event relations inside the recording transaction.
* Locks coordinate library-managed readers and writers only; direct file access must follow the same rule.

Deleting execution metadata hides its events without rewriting the archive.

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

## Individual simulation functions

The individual simulation functions are the low level interface to the simulation.

Both `mass_spec` and `mass_spec_iter` accept the same optional `collision_time`
argument. For example, pass `CollisionTimeOptions(hazard_atol=1e-11, hazard_rtol=1e-8)`
or `CollisionTimeOptions(method=CollisionTimeMethod.legacy)` from `apitofsim`.
Prepared streaming groups inherit these options from their run configuration.

For RF simulations using `integrated_hazard`, call `prepare_rf_basis(quadrupole, cluster)` from `apitofsim` and pass the result as `rf_basis` to `MassSpecSubstanceSingleInput` or each `MSSubstanceTreeCluster`.
Workflow runners handle this preparation and caching automatically. RF collision times use bounded thinning with these tables; hazard tolerances apply to non-RF integration.

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

## Collision-frequency profiles

A collision profile evaluates instantaneous total collision frequency along the
instrument for a fixed particle velocity. Prepare it once and reuse it when
selecting different recorded velocities; evaluation runs no trajectories and
performs no database writes.

Given an `ExperimentDatabase` named `db` and a selected `result_id`:

```python
import numpy as np
from pint import get_application_registry
from apitofsim import collision_profile, skimmer

Q_ = get_application_registry().Quantity
config, cluster = db.collision_profile_inputs(result_id=result_id)
# Alternatively: db.collision_profile_inputs(run_id=run_id, cluster_id=cluster_id)
profile = collision_profile(config, cluster, skimmer=skimmer(config))
z = Q_(np.linspace(0, config["lengths"].to("m").magnitude.sum(), 500), "m")
nu = profile.frequency(z)  # stationary particle; output in 1/s
nu_selected = profile.frequency(z, velocity=Q_([0, 0, 200], "m/s"))
```

Replace the example velocity with the selected event's `(vx, vy, vz)` in the
instrument frame. Gas flow is included, so even a stationary particle has
relative motion inside the skimmer. Collision rows record velocity before the
collision update. The returned rate includes collisions that may be rejected;
it is not an accepted-collision rate. Input resolution also works with read-only
and attached databases, provided the run references its stored configuration.

::: apitofsim.collision_profile

::: apitofsim.CollisionProfile
    options:
      members: true
      heading_level: 3

::: apitofsim.workflow.ExperimentDatabase.collision_profile_inputs


## Cached pathway rate curves and energy markers

Use [pathway_rate_curves][apitofsim.workflow.ExperimentDatabase.pathway_rate_curves]
to load a selected result's cached curves, including from read-only or attached
databases. By default it returns enabled pathways; use `all_pathways=True` to
include other known pathways. Missing curves have `histogram=None`.

```python
from apitofsim import internal_energy, available_fragmentation_energy

curves = db.pathway_rate_curves(run_id=1, parent=3, result_id=7)
for curve in curves:
    if curve.histogram is None:
        continue  # Cached curve absent; display an unavailable facet.
    xs = curve.histogram.x.to("eV").magnitude
    ks = curve.rates.to("1/s").magnitude
    # Raw recorded event energies are per-particle joules.
    energy = internal_energy(event.rot_energy, event.vibrational_energy)
    marker = available_fragmentation_energy(energy, curve.bonding_energy)
    if marker is not None:
        marker_ev = marker.to("eV").magnitude
```

The x coordinate is excess energy: internal energy minus bonding energy.
Energy helpers accept Pint quantities or recorded joule values and preserve
missing energies. Collision rows contain energies before the collision update.
For serial fragmentation, supply the event's current parent ID; it may be
unavailable in some records. Runs without histogram provenance raise
`RateCurveProvenanceError`.

::: apitofsim.workflow.PathwayRateCurve
    options:
      members: true
      heading_level: 3

::: apitofsim.internal_energy

::: apitofsim.available_fragmentation_energy

::: apitofsim.energy_in_joules

## Native recording and Python event delivery

Recording and Python observation are independent. `mass_spec()` returns a
`MassSpecResult` with `counters`, `timings`, `cancelled`, `status` and optional
`recording` provenance.

```python
from apitofsim import mass_spec, IpcRecording

result = mass_spec(instrument, substance, N=100)
recorded = mass_spec(instrument, substance, N=100, record_to="events.arrow")
print(recorded.recording.event_count)

# Existing standalone files require an explicit choice.
more = mass_spec(instrument, substance, N=100,
                 record_to=IpcRecording("events.arrow", mode="append"))
replacement = mass_spec(instrument, substance, N=100,
                        record_to=IpcRecording("events.arrow", mode="overwrite"))
```

Successful return includes native writer completion, file sync and publication.

`mass_spec_iter()` yields only event data. Counters/logs are available through
`result_callback`/`log_callback`; the final result is `stream.result` after
exhaustion or context cleanup. Use `loglevel` for diagnostic verbosity, independently
of recording and Python delivery.

```python
from apitofsim import mass_spec_iter, CollisionEvent

with mass_spec_iter(instrument, substance, 100, events="batches",
                    record_to="observed.arrow") as stream:
    for batch in stream:              # pyarrow.RecordBatch
        process_batch(batch)
print(stream.result.counters)

with mass_spec_iter(instrument, substance, 100, events="individual") as stream:
    for event in stream:              # immutable typed Event subclass
        if isinstance(event, CollisionEvent):
            inspect(event.theta, event.accepted, event.vibrational_energy)
```

Events are delivered as `InitEvent`, `CollisionEvent`, `PostCollisionEvent`,
`FragmentationEvent`, `PostFragmentationEvent` or `EscapeEvent`. Every event
carries identity, `particle_index`, `x/y/z/t`, `vx/vy/vz`, `ox/oy/oz`,
`rot_energy` and `vibrational_energy` as double-precision SI values. Extra fields
by event type:

| Event | Extra fields |
| ----- | ------------ |
| `CollisionEvent` | `theta`, `u_norm`, `accepted` (before-state) |
| `PostCollisionEvent` | accepted after-state |
| `FragmentationEvent` | `transition_index`, `next_particle_index` |
| `PostFragmentationEvent` | assigned daughter state |
| `InitEvent`, `EscapeEvent` | — |

SQL read views expose fields outside an event's kind as NULL.

Use `ordered=True` to deliver events by realization index and event index within
an execution. Recording implies ordered delivery; passing `ordered=False` with
recording raises an error. Without recording, histories are emitted as they
complete, with live events available once a realization finishes or ends early
through cancellation or failure.

Use the iterator as a context manager. `request_cancel()` is thread-safe and
nonblocking. Calling `stream.close()` or exiting the context early stops
admitting realizations, discards queued Python delivery, and waits for native
recording to stop. Partial recordings retain an incomplete status. A write
failure rolls back the pending append.

`initial_batch_size` and `max_batch_size` control delivery batch sizes; batches
contain up to 65536 events. `stream.finish()` discards remaining Python events,
delivers requested callbacks on the caller's thread, and waits for completion.

```python
from apitofsim import read_events

for batch in read_events("events.arrow", result_id=1):
    process_batch(batch)
for event in read_events("events.arrow", events="individual"):
    inspect(event)
```

## Prepared experiment execution

Start a run, prepare its roots, then choose blocking execution or event delivery.
Public preparation supports `SINGLE_CLUSTER` and `CLUSTER_TREE`. Input groups,
cache provenance and native node/transition order remain bound to that runner.

```python
from apitofsim.workflow import ExperimentRunner, SimulationMode

runner = ExperimentRunner(db)
runner.start_run(config_id, simulation_mode=SimulationMode.CLUSTER_TREE)
prepared = runner.prepare_from_config(config)

# Every root; summaries only.
result = runner.run_experiment(prepared, N=100, record_events=False)

# Every root; one database-owned IPC writer and durable transaction.
result = runner.run_experiment(prepared, N=100, record_events=True)

# Every root; native recording plus optional Python observation.
with runner.iter_experiment(prepared, N=100, record_events=True,
                            events="batches") as stream:
    for batch in stream:
        process_batch(batch)
print(stream.result.roots)
```

`run_from_config(config, record_events=True)` prepares and executes through the
existing configuration workflow. `ExperimentDatabase(..., record_events=True)`
sets the default; individual calls can override it. Database recording always
appends to its managed sidecar. The metadata commit follows sidecar fsync;
one experiment shares a writer across all roots. Reopening the database permits
another experiment to append to the same file.

For a single prepared group, use
`runner.iter_mass_spec(prepared, group, N=100, events="individual", record_events=True)`.
Recording roots in one transaction is sequential; overlapping recorded root
streams are rejected. Always close each stream context before leaving the
recording transaction. An explicit `with db.recording_transaction():` can group several individual
calls into one durability boundary. Results inside such an outer scope are
provisional until that scope exits. Whole-experiment early exit retains the
current root's partial history and does not start remaining roots.

### Failure diagnostics

`recorded_failure(error)` returns the workflow's persisted failure identity,
including startup errors when failure insertion succeeds. Partial events carry
`failed`/`cancelled` execution status rather than being interpreted as successful
results. Consumer exceptions propagate after cancellation/cleanup. Native I/O
errors are reported as failures even when the simulation itself finished.

### Native tree inspection

`MassSpecSubstanceTreeInput.tree_nodes`, `tree_pathways`, `cluster_payloads`
and `pathway_payloads`, and each node's `pathway_indices`, are read-only views
of native vectors.

`substance.topology_columns()` returns owned NumPy `int64` snapshots:

| Column | Indexed by | Meaning |
| --- | --- | --- |
| `node_payload` | Node index | Cluster payload index |
| `transition_source` | Transition index | Source node index |
| `transition_payload` | Transition index | Pathway payload index |
| `transition_destination` | Transition index | Destination node, or `-1` for a terminal transition |


### Cluster-tree fragmentation histories

Tree executions persist `cluster_tree_experiment_result`, `cluster_tree_node`
and `cluster_tree_transition` before starting native execution. Nodes are keyed by
result ID and local node index; different routes to one species remain separate
nodes. Transitions describe the exact enabled pathways, including unobserved
branches. A NULL transition destination means an intentional stop.

`tree_events_query(run_id=None, parent=None, result_id=None)` returns ordered
recorded tree events. `fragmentation_histories_query(...)` returns their ordered
fragmentation rows. Both use the same intersected scopes as `events_query()`;
`parent` and `cluster_id` identify the execution root. `current_cluster_id`
identifies the event's current species, including the **before** state at a split.
`next_cluster_id` identifies a continuing destination. Recorded
`next_particle_index = -1` means an intentional stop; NULL means unavailable data.
An interrupted history can end at a continuing split, without implying a stop.

```python
history = db.fragmentation_histories_query(result_id=42).fetchdf()
states = db.tree_events_query(result_id=42).fetchdf()
```

| Column | Meaning |
| ------ | ------- |
| `provenance` | `persisted` for resolved records, otherwise a `missing_*` or `inconsistent_*` diagnostic |
| `continuity_valid` | successive node transitions consistent in event order; NULL when the preceding state/destination is unavailable |

Keep these diagnostic columns when consuming histories.

Tree realizations are linked as they are recorded, so scopes include retained
partial executions. `total_us` remains NULL until successful completion;
`failure_id` links a recorded failure. Aggregate reports include completed tree
results and existing failure records, without counting incomplete trees as
successes. Cancellation and failure status are persisted in `event_execution` and the IPC footer.
