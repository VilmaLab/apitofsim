"""End-to-end recording benchmark using a private, cached model database.

Run with PYTHONPATH pointing at the chosen CMake build's python/Release.
Setup/preparation and full output verification are outside the measured region.
Execution, event publication/fsync, database close and database fsync are timed.
Each invocation needs a fresh output path. All scratch belongs on disk.
"""

import argparse
import hashlib
import json
import resource
from collections import Counter
from dataclasses import replace
from pathlib import Path
from time import perf_counter, process_time

import duckdb
import native_recording
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from apitofsim.recording import sync_file
from apitofsim.workflow import ExperimentDatabase, ExperimentRunner, SimulationMode


def verify(path, mode, expected):
    digests = {}
    observed = Counter()
    previous = (-1, -1, -1)
    with pa.memory_map(str(path), "r") as source:
        if mode == "arrow":
            reader = pa.ipc.open_file(source)
            batches = (reader.get_batch(i) for i in range(reader.num_record_batches))
        else:
            reader = pq.ParquetFile(source)
            assert all(
                reader.metadata.row_group(i).num_rows <= 65536
                for i in range(reader.metadata.num_row_groups)
            )
            codec = "ZSTD" if mode == "zstd" else "UNCOMPRESSED"
            assert all(
                reader.metadata.row_group(i).column(j).compression == codec
                for i in range(reader.metadata.num_row_groups)
                for j in range(reader.metadata.num_columns)
            )
            batches = reader.iter_batches(batch_size=65536)
        for batch in batches:
            for name in batch.schema.names:
                values = batch.column(name).to_numpy(zero_copy_only=True)
                digests.setdefault(name, hashlib.sha256()).update(
                    memoryview(values).cast("B")
                )
            result, realization, event = (
                batch.column(name).to_numpy(zero_copy_only=True)
                for name in ("result_id", "realization_index", "event_index")
            )
            old_result = np.r_[previous[0], result[:-1]]
            old_realization = np.r_[previous[1], realization[:-1]]
            old_event = np.r_[previous[2], event[:-1]]
            same_result = old_result == result
            same_history = same_result & (old_realization == realization)
            assert not np.any(old_result > result)
            assert not np.any(same_result & (old_realization > realization))
            assert np.all(event == np.where(same_history, old_event + 1, 0))
            previous = tuple(int(array[-1]) for array in (result, realization, event))
            keys, counts = np.unique(result, return_counts=True)
            observed.update(dict(zip(map(int, keys), map(int, counts), strict=True)))
    assert dict(observed) == expected
    return {name: digest.hexdigest() for name, digest in digests.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("base", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "--mode", choices=("summary", "arrow", "parquet", "zstd"), required=True
    )
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--realizations", type=int, default=5000)
    parser.add_argument("--roots", type=int, nargs="+", default=[1, 35])
    args = parser.parse_args()
    assert not args.output.exists(), "Use a fresh benchmark output path"
    # Include the newer cache when present; new_database ignores absent tables.
    native_recording.MODEL_TABLES += ("fragmentation_dos",)
    native_recording.new_database(args.base, args.output)
    db = ExperimentDatabase(
        args.output,
        recording_format="parquet" if args.mode in ("parquet", "zstd") else "arrow",
        recording_compression="zstd" if args.mode == "zstd" else "none",
        parquet_column_workers=args.workers,
    )
    row = next(db.iter_configs())
    config = row.config
    config["N"] = args.realizations
    runner = ExperimentRunner(db)
    runner.start_run(row.id, simulation_mode=SimulationMode.CLUSTER_TREE)
    prepared = runner.prepare_from_config(config, cached_densityandrate=(1, 2))
    groups = tuple(g for g in prepared.groups if g.cluster_id in args.roots)
    assert [g.cluster_id for g in groups] == args.roots
    prepared = replace(prepared, groups=groups)
    rss_before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    start, cpu_start = perf_counter(), process_time()
    result = runner.run_experiment(
        prepared, record_events=args.mode != "summary", seed=42
    )
    sidecar = db.events_path
    db.close()
    sync_file(args.output)
    wall, cpu = perf_counter() - start, process_time() - cpu_start
    peak_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    assert result.status == "complete"
    counters = [
        list(map(int, root.result.counters[:-1])) + [root.result.counters[-1].tolist()]
        for root in result.roots
    ]
    counts = {
        root.result.recording.result_id: root.result.recording.event_count
        for root in result.roots
        if root.result.recording is not None
    }
    report = dict(
        wall_seconds=wall,
        cpu_seconds=cpu,
        rss_before_kib=rss_before,
        peak_rss_kib=peak_rss,
        sidecar_bytes=sidecar.stat().st_size if counts else None,
        event_counts=counts,
        counters_sha256=hashlib.sha256(json.dumps(counters).encode()).hexdigest(),
        roots=args.roots,
        realizations_per_root=args.realizations,
        mode=args.mode,
        workers=args.workers,
        duckdb=duckdb.__version__,
        pyarrow=pa.__version__,
    )
    if counts:
        report["column_sha256"] = verify(sidecar, args.mode, counts)
    args.output.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
