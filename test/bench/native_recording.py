"""Benchmark durable native IPC with new-schema copies of a warm model database.

Input tables/caches are copied, not old event schemas. Initial setup and shared
preparation are excluded; execution, metadata, IPC append/journal/fsync and DB
close are timed. --seed-realizations first creates an untimed archive so the
measured run also exercises reopening/appending to existing event payloads.
"""

import argparse
import json
from pathlib import Path
from time import perf_counter, process_time

from apitofsim.recording import sync_file
from apitofsim.workflow import ExperimentDatabase, ExperimentRunner, SimulationMode

MODEL_TABLES = (
    "cluster",
    "pathway",
    "histogram_params",
    "cluster_dos",
    "products_dos",
    "k_rate",
    "rf_basis",
    "vib_energy_index",
    "experiment_config",
)


def new_database(base, output):
    db = ExperimentDatabase(output)
    db.create_tables()
    db.db.execute("attach '" + str(base).replace("'", "''") + "' as source (read_only)")
    available = {
        row[0]
        for row in db.db.execute(
            "select table_name from information_schema.tables where table_catalog='source'"
        ).fetchall()
    }
    for name in MODEL_TABLES:
        if name in available:
            db.db.execute(f"insert into {name} select * from source.{name}")
    # Copied model IDs must not collide with future cache insertions. Result
    # sequences are deliberately new; no old events/results are copied.
    source_sequences = dict(
        db.db.execute(
            "select sequence_name,last_value from duckdb_sequences() where database_name='source'"
        ).fetchall()
    )
    for (name,) in db.db.execute(
        "select sequence_name from duckdb_sequences() where database_name != 'source'"
    ).fetchall():
        if name.startswith("experiment_") and name != "experiment_config_sequence":
            continue
        value = source_sequences.get(name)
        if value:
            db.db.execute("select nextval(?) from range(?)", [name, value]).fetchall()
    db.db.execute("detach source")
    db.close()


def prepare(db, N):
    row = next(db.iter_configs())
    config = row.config
    config["N"] = N
    config["vib_energy_sampler"] = "dos-only"
    runner = ExperimentRunner(db)
    runner.start_run(row.id, simulation_mode=SimulationMode.CLUSTER_TREE)
    return runner, runner.prepare_from_config(config, cached_densityandrate=(1, 2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("base", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--record-events", action="store_true")
    parser.add_argument("--realizations", type=int, default=100)
    parser.add_argument("--seed-realizations", type=int, default=0)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("output must be new")
    new_database(args.base, args.output)
    if args.seed_realizations:
        with ExperimentDatabase(args.output) as db:
            runner, prepared = prepare(db, args.seed_realizations)
            runner.run_experiment(prepared, record_events=True)
    db = ExperimentDatabase(args.output)
    runner, prepared = prepare(db, args.realizations)
    assert db.db.execute(
        "select count(*) from duckdb_constraints() where constraint_type='FOREIGN KEY'"
    ).fetchone() == (0,)
    old_bytes = db.events_path.stat().st_size if db.events_path.exists() else 0
    start, cpu = perf_counter(), process_time()
    result = runner.run_experiment(prepared, record_events=args.record_events)
    db.close()
    sync_file(args.output)
    wall, cpu = perf_counter() - start, process_time() - cpu
    assert result.status == "complete"
    counters = [
        list(map(int, root.result.counters[:-1])) + [root.result.counters[-1].tolist()]
        for root in result.roots
    ]
    with ExperimentDatabase(args.output, readonly=True) as read:
        count = read.events_query(run_id=prepared.run_id).count("*").fetchall()[0][0]
        if args.record_events:
            assert count > 0
            assert read.events_query(run_id=prepared.run_id).aggregate(
                "count(*) as n"
            ).fetchall()[0][0] == sum(
                root.result.recording.event_count for root in result.roots
            )
            invalid = (
                read.events_query(run_id=prepared.run_id)
                .aggregate(
                    "count(*) as n, min(event_index) as first, max(event_index) as last, count(distinct event_index) as unique_events",
                    "result_id,realization_index",
                )
                .filter("first != 0 or n != last + 1 or n != unique_events")
            )
            assert invalid.fetchall() == []
        else:
            assert count == 0
    archive = args.output.with_suffix(".events.arrow")
    report = dict(
        record_events=args.record_events,
        roots=len(prepared.groups),
        N=args.realizations,
        seeded_N=args.seed_realizations,
        total_s=wall,
        cpu_s=cpu,
        events=count,
        previous_archive_bytes=old_bytes,
        archive_bytes=archive.stat().st_size if archive.exists() else 0,
        database_bytes=args.output.stat().st_size,
        counters=counters,
    )
    args.output.with_suffix(".json").write_text(json.dumps(report, indent=2))
    print(
        json.dumps({key: value for key, value in report.items() if key != "counters"})
    )


if __name__ == "__main__":
    main()
