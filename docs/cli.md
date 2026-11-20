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

Use `--record-events` to record individual particle events.

```bash
apitofsim db prepare create config.toml database.duckdb
apitofsim db run database.duckdb --record-events
```

Recording appends natively to `database.events.arrow`.
This must be kept in the same directory as the database.

Record to Parquet for smaller files with compression:

```bash
apitofsim db run database.duckdb --record-events --event-format parquet \
    --event-compression zstd --parquet-column-workers 4
```

Empirically, when using `zstd`, setting `--parquet-column-workers` to 25% of available CPU cores is close to optimal.

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

## The legacy tools

These legacy tools are kept for existing workflows and are not recommended for new ones.

If you have installed via Conda, and activated the relevant environment, the legacy command line tools `apitofsim-skimmer`, `apitofsim-densityandrate` and `apitofsim-mass-spec` should be installed and in your path.
If you have compiled the sources yourself, you will need to add `build/src/cli` to your path for the following example to work.

An example is then available at `examples/legacy_cli`.
First `cd` to that directory, and then follow the instructions in the `README.md` file, included below:

```bash
$ git clone https://github.com/VilmaLab/apitofsim.git
$ cd apitofsim/examples/legacy_cli
```

--8<-- "examples/legacy_cli/README.md"
