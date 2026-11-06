"""Measure phase-space cold preparation, warm loading and first-simulation entry.

Usage: python test/bench/fragmentation_preparation.py prepared.duckdb ~/.scratch/dos-bench.duckdb
The destination must not exist. Existing normal DOS/rates are reused; simulation
stops at entry to the first run_mass_spec call, without running realizations.
"""

import argparse
import json
from contextlib import closing
from pathlib import Path
from time import perf_counter

from apitofsim.workflow import DerivedDataPreparer, ExperimentDatabase, ExperimentRunner
from native_recording import new_database


class FirstSimulation(BaseException):
    pass


class TimedPreparer(DerivedDataPreparer):
    def _prepare_fragmentation_products(self, *args, **kwargs):
        start = perf_counter()
        super()._prepare_fragmentation_products(*args, **kwargs)
        self.product_seconds = perf_counter() - start


class TimedRunner(ExperimentRunner):
    def __init__(self, db):
        super().__init__(db)
        self.preparer = TimedPreparer(db)

    def _prepare_cluster_tree_lookup(self, *args, **kwargs):
        start = perf_counter()
        result = super()._prepare_cluster_tree_lookup(*args, **kwargs)
        self.lookup_seconds = perf_counter() - start
        return result

    def run_mass_spec(self, *args, **kwargs):
        raise FirstSimulation()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    if args.destination.exists():
        parser.error("destination must not exist")
    new_database(args.source, args.destination)
    results = []
    for phase in ("cold", "warm"):
        with closing(ExperimentDatabase(str(args.destination))) as db:
            db.ensure_fragmentation_dos_cache()
            if phase == "cold":
                db.db.execute("delete from fragmentation_dos")
            config = next(db.iter_configs()).config
            config.update(
                fragmentation_state_method="phase-space", simulation_mode="cluster-tree"
            )
            runner = TimedRunner(db)
            start = perf_counter()
            try:
                runner.run_from_config(config)
            except FirstSimulation:
                pass
            else:
                raise RuntimeError("No simulation reached")
            results.append(
                dict(
                    phase=phase,
                    first_simulation_seconds=perf_counter() - start,
                    product_dos_seconds=runner.preparer.product_seconds,
                    tree_lookup_seconds=runner.lookup_seconds,
                    prepared_pairs=len(runner.preparer.fragmentation_products),
                    cached_payloads=db.db.execute(
                        "select count(*) from fragmentation_dos"
                    ).fetchone()[0],
                )
            )
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
