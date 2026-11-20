# 🤖 Benchmarks

Recording benchmarks compare writer implementations, encoding concurrency, and
Python delivery. See [Internals](internals.md) for the recording design and
[Development environment](development.md) for build instructions.

## Native benchmarks

The native benchmark executables are under `builds/test/test` when built with
the [test configuration](development.md#running-tests). Set
`APITOFSIM_BENCH_CONCURRENCY=1` to run them with a one-thread limit.

## Running the recording benchmarks

`test/bench/parquet_matrix.py` compares separately configured Release builds
using a cached private model database. For example:

```bash
python test/bench/parquet_matrix.py /path/to/model.duckdb "$HOME/.scratch/parquet-bench" \
  --python /path/to/environment/bin/python \
  --arrow-build "$HOME/.scratch/arrow-build" \
  --carquet-build "$HOME/.scratch/carquet-build" \
  --workers 1 2 4 8 --realizations 5000 --repeats 1
```

The Arrow comparison build must use an older revision with Arrow C++ recording
(the recorded baseline below is `287d81c`). Both builds need
`APITOFSIM_BUILD_PYTHON=ON` and `BUILD_TESTING=ON`, which mirrors
the package into `python/Release` next to its extension. The benchmark selects
cluster-tree roots 1 and 35 with seed 42. Each case runs in a fresh process;
preparation is excluded, and execution through durable database close is
included. Peak RSS includes preparation but excludes the subsequent full-file
verification. Every event column is hashed in physical order, with realization
ordering, continuous event indices, row-group limits and compression checked.
The matrix compares those hashes and simulation counters across all cases,
retains JSON reports and small logs, and removes its generated databases and
sidecars after successful verification.


## Original carquet prototype benchmark (2026-11-11)

This repeats the recording workload from [PR #173](https://github.com/VilmaLab/apitofsim/pull/173):
a private Besel2020 model, cluster-tree roots 1 and 35, 5,000 realizations each,
seed 42, and 45,762,568 recorded events. Each cell is one end-to-end measurement,
including simulation, encoding, event sync, database commit/close and database
sync. Preparation and verification are excluded. Peak RSS includes preparation
and is sampled before verification. The OS cache was left in place, with normal
desktop background load. Treat small differences between worker counts as noise.

Hardware: Intel Core Ultra 5 225U, 14 logical CPUs, NVMe/Btrfs, Linux x86-64.
Both simulation builds use Clang 22.1.8, Release (`-O3 -DNDEBUG -march=native`),
and LLD. Arrow C++/Parquet 26.0.1 comes from conda-forge; carquet 0.8.0 is built
from the pinned source with its upstream Release defaults. Both writers link the
same conda Zstandard library. Python is 3.12.16, PyArrow 26.0.1, DuckDB 1.5.6,
and oneTBB 2023.1. These fresh measurements use a different library build and
background load from the original PR, so compare backends within this table.

Both writers request dictionary encoding, statistics, a 65,536-row group limit
and Zstandard level 3 where enabled. The worker count controls encoding:
Arrow uses its private column pool, while carquet uses the writer thread's
OpenMP concurrency setting for its internal encoding workers. Carquet also has
its default asynchronous I/O thread. Simulation retains default TBB concurrency
in every case; the table does not count total application threads.

| Recording | Encoding workers | Wall time | Sidecar | Peak process RSS |
|---|---:|---:|---:|---:|
| Summary only | — | 32.4 s | — | 1.60 GiB |
| Arrow IPC | — | 36.0 s | 7.50 GiB | 1.69 GiB |
| Arrow C++ Parquet, none | 1 | 90.8 s | 4.01 GiB | 2.06 GiB |
| carquet Parquet, none | 1 | 42.5 s | 4.05 GiB | 1.97 GiB |
| Arrow C++ Parquet, none | 2 | 66.2 s | 4.01 GiB | 2.10 GiB |
| carquet Parquet, none | 2 | 40.5 s | 4.05 GiB | 2.10 GiB |
| Arrow C++ Parquet, none | 4 | 69.6 s | 4.01 GiB | 2.14 GiB |
| carquet Parquet, none | 4 | 39.2 s | 4.05 GiB | 2.09 GiB |
| Arrow C++ Parquet, none | 8 | 55.2 s | 4.01 GiB | 2.19 GiB |
| carquet Parquet, none | 8 | 43.2 s | 4.05 GiB | 2.06 GiB |
| Arrow C++ Parquet, Zstandard | 1 | 103.2 s | 3.74 GiB | 2.05 GiB |
| carquet Parquet, Zstandard | 1 | 61.4 s | 3.77 GiB | 1.98 GiB |
| Arrow C++ Parquet, Zstandard | 2 | 80.5 s | 3.74 GiB | 2.10 GiB |
| carquet Parquet, Zstandard | 2 | 42.0 s | 3.77 GiB | 2.16 GiB |
| Arrow C++ Parquet, Zstandard | 4 | 64.8 s | 3.74 GiB | 2.12 GiB |
| carquet Parquet, Zstandard | 4 | 46.7 s | 3.77 GiB | 2.18 GiB |
| Arrow C++ Parquet, Zstandard | 8 | 60.3 s | 3.74 GiB | 2.21 GiB |
| carquet Parquet, Zstandard | 8 | 40.4 s | 3.77 GiB | 2.12 GiB |

Comparing the fastest observed configurations, carquet is 1.41× faster
uncompressed (39.2 versus 55.2 seconds) and 1.49× faster with Zstandard (40.4
versus 60.3 seconds). Its files are about 1% larger. With Zstandard, two carquet
encoding workers already reach 42.0 seconds; higher counts provide no consistent
improvement in this single-run sweep. These figures measure end-to-end recording.

Every recorded variant has identical simulation counters and identical SHA-256
hashes for all 22 event columns in physical order. Verification also checks
result/realization ordering, continuous event indices, row-group limits and
compression. One interrupted execution-environment session was discarded and
its incomplete case rerun successfully. The small raw report, source hashes,
environment and QA details are in `test/bench/results/carquet-20261111.json`.
Generated databases and sidecars were removed after successful verification.

The benchmark used the initial carquet prototype before the full migration.
The current implementation uses carquet for all Parquet recording and the
restored nanoarrow implementation for IPC. The recording API, schema,
transactions and bounded queue are preserved; downstream readers need no
changes. The prototype figures above retain their original build provenance.

Validation includes both native concurrency configurations for both backends,
the full carquet-enabled Python suite (563 passed, one expected skip), strict
warnings builds, formatting, lint/types, and ASan/UBSan writer tests. Leak
checking was disabled because the sandbox uses ptrace. A full clang-tidy run
found one signed-index warning, fixed and rechecked in all changed writer files;
there were no other project diagnostics. The raw report describes validation of that prototype; the migration PR
also runs the platform wheel and documentation workflows.


## Migration recording validation (2026-11-11)

The migration was checked separately on the same full Besel2020 workload after
restoring nanoarrow IPC and making carquet unconditional. Background load
differed from the earlier sweep, so these rows validate the final recording
paths rather than provide a new cross-backend speedup comparison. The core
uses the same Release flags; carquet's upstream implicit LTO is disabled so
ordinary Clang builds can use their configured linker.

| Recording | Encoding workers | Wall time | Sidecar | Peak process RSS |
|---|---:|---:|---:|---:|
| nanoarrow IPC | — | 33.1 s | 7.50 GiB | 1.70 GiB |
| carquet Parquet, none | 4 | 31.6 s | 4.05 GiB | 2.07 GiB |
| carquet Parquet, Zstandard | 4 | 32.5 s | 3.77 GiB | 2.08 GiB |

All 45,762,568 events match the original Arrow C++/IPC baseline in every one
of the 22 physical-order column hashes and all simulation counters. Full-file
checks again verify row-group limits, codecs, ordering and ordinal continuity.
The raw records are in `test/bench/results/carquet-nanoarrow-20261111.json`.

## Collator ownership benchmark (2026-11-13)

This isolates the ownership refactor: both revisions already use nanoarrow and
carquet. The baseline is `6fe54eb`; the refactored implementation, including the
GCC portability fix, is `772576a`. The same 45,762,568-event Besel2020 workload
above runs in a fresh process for each case, with three repeats and interleaved
builds. Case order rotates between repeats. No local builds or QA jobs run
during the final sweep; normal desktop load and the warm OS cache remain.
Both binaries use Clang 22.1.8, Release `-O3 -DNDEBUG -march=native`, the system
default linker and the same runtime dependencies on the hardware described above.

Parquet uses Zstandard level 3 and four encoding workers. Simulation uses
default TBB concurrency throughout. Python delivery iterates Arrow batches and
counts their rows, retaining only the current batch. It includes native handoff,
array/table conversion and consumption in the timed region. Model preparation
and full-file verification are excluded; simulation, encoding, publication,
fsync and database commit/close remain included. RSS is sampled before
verification and includes preparation.

Wall times below are median (minimum–maximum), in seconds. RSS columns show
medians of process peaks before / after, in GiB. Negative changes mean less wall time.

| Recording | Python delivery | Before wall time | After wall time | Change | Peak RSS before / after |
|---|---|---:|---:|---:|---:|
| nanoarrow IPC | no | 29.87 (28.72–30.86) | 29.23 (28.98–30.43) | -2.1% | 1.70 / 1.70 |
| nanoarrow IPC | yes | 28.81 (28.81–30.21) | 29.22 (28.71–35.94) | +1.4% | 1.90 / 1.90 |
| carquet Parquet, Zstandard | no | 32.07 (31.95–34.19) | 32.13 (32.00–34.05) | +0.2% | 2.09 / 2.10 |
| carquet Parquet, Zstandard | yes | 32.68 (32.26–33.86) | 32.38 (32.20–32.67) | -0.9% | 2.12 / 2.12 |

Median differences are smaller than the observed run-to-run variation; the
ranges overlap in every combination. There is no clear throughput or peak-RSS
change in this workload. Sidecars remain 7.50 GiB for IPC and 3.77 GiB for
Parquet. All 24 runs match every event-column hash and simulation counter;
Python-delivery counts also match all recorded events.

A preliminary sweep of `6fe54eb` versus `f8d6c22` was interrupted after five
completed cases to fix GCC's variant-move warning. The final sweep restarts all
24 cases against `772576a`. The raw report preserves those five preliminary
observations separately, including its slower 47.17-second IPC sample, alongside
all final samples, binary hashes and methodology:
`test/bench/results/collator-20261113.json`.

Reproduce with separately configured builds of the two revisions:

```bash
python test/bench/parquet_matrix.py /path/to/model.duckdb "$HOME/.scratch/collator-bench" \
  --python /path/to/environment/bin/python --delivery-matrix \
  --before-build "$HOME/.scratch/before-build" --after-build "$HOME/.scratch/after-build" \
  --before-ref 6fe54eb --after-ref 772576a --workers 4 --repeats 3 --realizations 5000
```
