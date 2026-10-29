# Using the Python API

This page documents the Python API.
The main API is in the `apitofsim.api` module, most of which is re-exported from `apitofsim`.

## Building simulation configurations

Use `build_config()` for a GUI or custom runner. It returns `SimulationConfig`, a
public `TypedDict` containing Pint quantities, ordered quantity arrays, `Gas`,
and (optionally) `Quadrupole`. Pass it directly to `ExperimentRunner` or
`ExperimentDatabase.insert_config()`; database reads reconstruct the objects
and quantities. Input mappings and objects are not mutated.

`CONFIG_FIELDS` is a read-only mapping from canonical field names to `ConfigField`
records. Each record exposes `unit` and `components` (an empty tuple for scalars).
Object members use dotted names such as `gas.radius` and
`quadrupole.radiofrequency`. Array components are:

| Field | Canonical unit | Component order |
| --- | --- | --- |
| lengths | meter | L0, L1, L2, L3, Lsk |
| voltages | volt | V0, V1, V2, V3, V4 |
| pressures | pascal | first, second |

The named input types are `LengthComponents`, `VoltageComponents`,
`PressureComponents`, `GasComponents`, and `QuadrupoleComponents`. Arrays may
also be supplied as a Pint quantity with the exact shape `(5,)`, `(5,)`, or
`(2,)`, respectively. Named mappings require every listed component and reject
unknown keys. Gas and quadrupole inputs also accept existing public objects.
Dimensional scalars require Pint quantities; dimensionless scalars accept plain
numbers or dimensionless quantities. Values must be finite, and iteration,
resolution and realization counts must be integers. Errors name the invalid
field/component and expected unit or shape. Physical bounds and form-specific
validation remain the consumer's responsibility.

All arguments without defaults in `build_config()` are required. The builder
supplies no numerical tuning defaults. Omitted/`None` quadrupoles mean no mass
filter. When a quadrupole is supplied, its frequency and rod radius are required;
AC and DC voltages default to zero. Optional `bonding_energy`,
`fragmentation_energy`, and `radius_pinhole` are omitted when `None`, otherwise
retained in kelvin, kelvin, and meters. Histogram energies use temperature units;
convert energy quantities explicitly with a Pint Boltzmann context if needed.
`fragmentation_energy` is a pathway-at-a-time runner override; grouped preparation
uses electronic-energy differences. `radius_pinhole` is preserved for consumers
constructing `MassSpectrometer` directly; `ExperimentRunner` currently retains
its existing default pinhole radius (1 mm).

Geometry follows the simulation's definitions: `L0` is the first chamber length,
`L1` is the gap after the skimmer and before the quadrupole, `L2` is the quadrupole
length, `L3` is the remaining second chamber length, and `Lsk` is the axial
skimmer region length. `dc` is the radius at the smallest skimmer cross section.
The flow calculation uses the expanding radius `dc + z * tan(alpha)`, where
`alpha = alpha_factor * pi`; `alpha_factor` is normalized to `halfturn`.
For example, 45 degrees becomes 0.25 halfturn. This angle is the slope angle
used by that formula. See [configuration geometry](configuration.md#top-level-arrays)
for voltage locations and diagrams.

This example uses only public APIs; a form can substitute its submitted Pint
quantities for the values below:

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

Legacy `ConfigFile` readers remain available. Their array ordering and gas/
quadrupole construction share the same normalization helpers, and
`ConfigFile.into_json_config()` preserves quadrupole settings.

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

`postcollision` rows contain the final particle state for accepted collisions,
after velocity updates and internal-energy redistribution. They correlate with
the preceding `collision` row within the realization by particle index and
position/time, and have null collision-detail fields. `collision` rows contain
the pre-collision state, including for rejected attempts.

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
| `event_count` | Event-info rows, including post-collision states |
| `escape_count` | Escape event-info rows |
| `collision_event_count` | Collision attempts, excluding post-collision states |
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
For serial fragmentation, supply the event's known current parent ID; older
records may not identify it. Runs without histogram provenance raise
`RateCurveProvenanceError`.

::: apitofsim.workflow.PathwayRateCurve
    options:
      members: true
      heading_level: 3

::: apitofsim.internal_energy

::: apitofsim.available_fragmentation_energy

::: apitofsim.energy_in_joules

## Cancelling a streaming simulation

`mass_spec_iter(...)` returns a `MassSpecIterator`. Call its thread-safe,
idempotent, non-blocking `request_cancel()` method to request cooperative
cancellation of that simulation. Another Python thread can call this while
iteration or `join_if_joinable()` is blocked. Continue iterating to drain queued
items and receive one `MassSpecFinalResult`, then normal iteration termination:

```python
from apitofsim.api import MassSpecFinalResult, MassSpecIntermediateCounter, mass_spec_iter

with mass_spec_iter(instrument, substance, N) as stream:
    for item in stream:
        if isinstance(item, MassSpecIntermediateCounter) and should_stop():
            stream.request_cancel()
        if isinstance(item, MassSpecFinalResult):
            completed = item.counters.n_realizations
            cancelled = item.cancelled  # Also available as stream.cancelled.
```

Final counters include only completed realizations. Active realizations may be
interrupted, and their streamed events may remain in the queue without a
corresponding completed realization. Cancellation does not send a signal, raise
`KeyboardInterrupt`, or cancel other iterators. Actual Ctrl-C behavior is
unchanged. Requests after completion have no effect; `cancelled` stays false for
full completion. Exiting the context manager waits for the worker; request
cancellation explicitly if needed before exiting.

## Prepared, database-recording simulations

For `SINGLE_CLUSTER` runs, preparation and execution can be consumed separately.
The database must already contain clusters, pathways and a configuration. Select
one parent by ID/name, or select pathways by IDs/name triples; the two selectors
are mutually exclusive. An empty or partly missing selection raises `ValueError`.
Selections spanning parents produce separate groups. The tuple `pathway_ids` in
each group matches its substance's native pathway indices and `product_ids`.

```python
from apitofsim.api import MassSpecIntermediateCounter, MassSpecLogItem
from apitofsim.workflow import ExperimentRunner, RecordedMassSpecResult, SimulationMode

# db is caller-owned: ExperimentDatabase or RealizationDatabase.
row = next(db.iter_configs())
runner = ExperimentRunner(db)
runner.start_run(row.id, simulation_mode=SimulationMode.SINGLE_CLUSTER)
print("Preparing")
prepared = runner.prepare_from_config(row.config, show_progress=False)
# To reuse explicitly selected caches, also pass cached_densityandrate=(dos_id, rate_id).
# Their independent meshes are used, and their actual IDs are saved as provenance.
print("Simulating")
for group in prepared.groups:
    with runner.iter_mass_spec(prepared, group, N=row.config["N"], seed=42, loglevel=1) as stream:
        for item in stream:
            if isinstance(item, MassSpecIntermediateCounter):
                print("Completed:", item.counters.n_realizations)
                if item.counters.n_realizations >= 100:
                    stream.request_cancel()  # Example stopping policy; keep draining.
            elif isinstance(item, MassSpecLogItem):
                print(item.type, item.name)
            elif isinstance(item, RecordedMassSpecResult):
                print("Saved:", item.run_id, item.cluster_id, item.result_id)
                print("Cancelled:", item.result.cancelled)
            # Native events are already recorded on this consuming thread.
```

`prepare_from_config()` requires `start_run()` and never inserts another run or
config. Preparation saves the effective configuration and actual histogram IDs
before any trajectories start. Prepared inputs retain their run identity even
if the runner later starts another run. Inputs belong to their preparing runner;
use a group from that prepared object's `groups`. Treat the prepared native inputs
as read-only. Grouped runs preserve existing electronic-energy bonding energies;
`fragmentation_energy` overrides currently apply to pathway-at-a-time runs only.

Use the recording stream as a context manager. Exiting early requests
cancellation, drains and records remaining events/results, and joins the worker.
The saved terminal wrapper is also available as `stream.result` after context
exit. This is stronger than the low-level iterator's join-only context exit.
`request_cancel()` is thread-safe, idempotent and non-blocking, including while
`next()` is blocked. Only that method may be called from another thread; iteration,
context exit and database access stay on the consuming thread.

A terminal wrapper is exposed only after counters, timings and realization
associations are saved. It contains the original `MassSpecFinalResult`, including
`cancelled`; cancellation status is not currently a database column. Cancelled
runs with zero completed realizations are saved too. Interrupted realizations may
have recorded events without contributing to final completed-realization counters.
Consumers choose how to display or delete cancelled output.

The stream records execution/persistence failures once when possible and re-raises
the original exception. Missing native completion is an error. Committed events
are retained and linked to the failure when possible. Do not insert another
execution-failure row in the consumer; consumer-specific diagnostics and errors
in consumer processing remain the consumer's responsibility. Secondary cleanup
errors are attached as exception notes. Final persistence uses the existing
non-atomic database methods: a persistence error may leave partial result rows
or associations; writes are not retried and no successful terminal wrapper is
exposed. The stream never closes or checkpoints the caller's database.

### Failure diagnostics

::: apitofsim.workflow.RecordedFailure

::: apitofsim.workflow.recorded_failure

See [add_failure_diagnostics][apitofsim.workflow.ExperimentDatabase.add_failure_diagnostics]
and [get_failure_diagnostics][apitofsim.workflow.ExperimentDatabase.get_failure_diagnostics]
for storing and reading diagnostics.

```python
from traceback import format_exception
from apitofsim.workflow import recorded_failure

try:
    with runner.iter_mass_spec(prepared, group, N=100) as stream:
        for item in stream:
            handle(item)
except Exception as exc:
    failure = recorded_failure(exc)
    diagnostics = {"stage": stage, "traceback": format_exception(exc)}
    if failure is not None:
        try:
            db.add_failure_diagnostics(failure.failure_id, diagnostics)
        except Exception as enrichment_error:
            exc.add_note(f"Saving diagnostics failed: {enrichment_error}")
            save_external_diagnostics(diagnostics)
    else:
        save_external_diagnostics(diagnostics)
    raise
```
