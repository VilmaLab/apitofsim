# Using the command line tools

If you have installed via Conda, and activated the relevant environment, the command line tool `apitofsim` should be installed into your path.
If you are using `uv` you may need to run `uv run apitofsim` instead.

::: mkdocs-click
    :module: apitofsim.cli
    :command: cli
    :prog_name: apitofsim
    :depth: 1
    :list_subcommands: True

## Native event recording

Use an experiment database for both summary-only and recorded runs:

```bash
apitofsim db prepare create config.toml database.duckdb
apitofsim db run database.duckdb --record-events
```

Recording appends natively to `database.events.arrow`. The optional ASE sidecar
is `database.ase.sqlite.db`. Move the files together with matching basenames.
`--no-record-events` is the default. The former `--db-type=realization` schema
and relational event inserts are removed; old event databases require migration.
Do not read the event sidecar while an append transaction is active.

## Convert a completed event recording

```bash
apitofsim convert-events database.duckdb --to parquet
# Opt in to Zstandard compression.
apitofsim convert-events database.duckdb --to parquet --compression zstd
# With DuckDB's vortex extension installed.
apitofsim convert-events database.duckdb --to vortex
```

Close recording readers and writers before converting. The command streams IPC
batches into `database.events.parquet` in their original order. The default
is uncompressed Parquet. It reports source size, available space and progress.
Budget roughly the IPC size for uncompressed output until publication.

Vortex writes `database.events.vortex` with its native encodings and verifies
event values and order against the IPC source before publication. Run
`INSTALL vortex` in the same DuckDB version used for conversion and reads.
Extension availability depends on platform. The extension tested with
DuckDB 1.5.6 requires Vortex paths without whitespace. The `--compression`
option controls Parquet pages. [DuckDB Vortex extension](https://duckdb.org/docs/current/core_extensions/vortex)

The database stays at the same path. After validation and publication, the
command removes the old IPC file. Re-run the same command after an interruption
to resume publication or finish cleanup using the output already written.
An unrelated file at the destination blocks conversion. If both files exist,
the database's committed format determines which file readers open. Move the database and its
active sidecar together. To record additional events, use a separate database.

## Event plots

`apitofsim db plot plot-events results.db plots/` writes one Matplotlib figure per recorded
parent cluster, containing the instrument schematic, stacked cumulative
fragmentation/escape fractions, and regional fragmentation/escape bars.
Repeat `--panel` to select 1–3 distinct panels in top-to-bottom order:

```bash
apitofsim db plot plot-events results.db plots/ --panel bars --panel schematic --panel cumulative
apitofsim db plot plot-events results.db plots/ --panel cumulative --rescale none
```

`--rescale schematic` (the default) uses the explorer's chamber widths;
`equal` gives each region equal width, and `none` uses physical distances in mm.
Fractions include all recorded realizations in the denominator. Incomplete or
ambiguous histories contribute to neither fragmentation nor escape counts.
The former `--plot-type` scatter, swarm, strip and violin options are replaced
by `--panel`.

In Python, the same ordered selection is available as
`plot_events_cluster(db, run_id, False, cluster_id, panels=("bars", "cumulative"))`
from `apitofsim.plotting.events`. It returns a Matplotlib figure;
`plot_events_experiment()` yields `(parent_name, figure)` pairs for a run.

## The legacy tools

These tools are not recommended for new users, and kept available to support existing workflows.

If you have installed via Conda, and activated the relevant environment, the legacy command line tools `apitofsim-skimmer`, `apitofsim-densityandrate` and `apitofsim-mass-spec` should be installed and in your path.
If you have compiled the sources yourself, you will need to add `build/src/cli` to your path for the following example to work.

An example is then available at `examples/legacy_cli`.
First `cd` to that directory, and then follow the instructions in the `README.md` file, included below:

```bash
$ git clone https://github.com/VilmaLab/apitofsim.git
$ cd apitofsim/examples/legacy_cli
```

--8<-- "examples/legacy_cli/README.md"
