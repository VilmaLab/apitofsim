"""Isolate FK and event PK/index costs with prebuilt Arrow columns.

Run: uv run python test/bench/ingest_events.py --rows 661468 --repeat 3
The no-PK variant is a benchmark control, not a supported database format.
"""

import argparse
import re
from statistics import median
from time import perf_counter

import duckdb
from apitofsim.workflow import RealizationDatabase, sql_files
from apitofsim.workflow.db import connection_scope


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, default=661468)
    parser.add_argument("--repeat", type=int, default=3)
    args = parser.parse_args()
    if args.rows < 1 or args.repeat < 1:
        parser.error("--rows and --repeat must be positive")
    with duckdb.connect(config={"threads": 1}) as source:
        # Allocate once, outside all measured sections. Interleaved event kinds
        # exercise the shared-state table and both detail tables.
        events = source.execute(
            """select i::integer id, (i % 100 + 1)::integer realization_id,
                case when i % 6 = 0 then 'fragmentation'
                     when i % 3 = 0 then 'collision' else 'escape' end event_type,
                {'x': 0.0::double, 'y': 0.0::double, 'z': 0.0::double,
                 't': i * 1e-9} postime,
                {'x': 0.0::double, 'y': 0.0::double, 'z': 1.0::double} velocity,
                {'x': 0.0::double, 'y': 0.0::double, 'z': 0.0::double} omega,
                0.0::double rot_energy, 0.0::double vibrational_energy,
                0::integer particle_index from range(1, ?) r(i)""",
            [args.rows + 1],
        ).to_arrow_table()
        source.register("events", events)
        collisions = source.execute(
            """select id, 0.0::double theta, 1.0::double u_norm, true accepted
               from events where event_type = 'collision'"""
        ).to_arrow_table()
        fragmentations = source.execute(
            """select id, 0::integer transition_index, 1::integer next_particle_index,
                      1::integer pathway_id from events where event_type = 'fragmentation'"""
        ).to_arrow_table()
    samples = {mode: [] for mode in ("PK + FK", "PK only", "no event PK/FK")}
    for _ in range(args.repeat):
        for mode, times in samples.items():
            with connection_scope(RealizationDatabase, ":memory:") as db:
                if mode == "no event PK/FK":
                    ddl = (
                        "\n".join(db.TABLES[:-1])
                        + "\n"
                        + re.sub(r"\bprimary key\b", "", sql_files.realizations)
                    )
                    db.db.execute(sql_files.strip_foreign_keys(ddl))
                    db.refresh_views()
                else:
                    db.create_tables(foreign_keys=mode == "PK + FK")
                db.db.execute("insert into cluster(id) values (1)")
                db.db.execute("insert into pathway values (1, 1, 1, 1)")
                db.db.execute(
                    "insert into realization(id) select range from range(1, 101)"
                )
                tables = {
                    "event_info": events,
                    "collision_event": collisions,
                    "fragmentation_event": fragmentations,
                }
                for table, arrow in tables.items():
                    db.db.register(f"arrow_{table}", arrow)
                start = perf_counter()
                db.db.execute("begin")
                for table in tables:
                    db.db.execute(f"insert into {table} select * from arrow_{table}")
                db.db.execute("commit")
                times.append(perf_counter() - start)
                for table, arrow in tables.items():
                    assert db.db.table(table).count("*").fetchone()[0] == arrow.num_rows
                assert db.check_foreign_keys() == {}
    print(f"DuckDB {duckdb.__version__}; {args.rows} events; one DB thread; in-memory")
    for mode, times in samples.items():
        print(f"{mode}: median {median(times):.3f}s; samples {times}")


if __name__ == "__main__":
    main()
