"""Run parquet_recording.py sequentially against two configured CMake builds."""

import argparse
import json
import os
import subprocess
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("base", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--arrow-build", type=Path, required=True)
    parser.add_argument("--carquet-build", type=Path, required=True)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--workers", type=int, nargs="+", default=[1, 2, 4, 8])
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--realizations", type=int, default=5000)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    cases = [("arrow", "summary", 1), ("arrow", "arrow", 1)]
    for index, workers in enumerate(args.workers):
        for mode in ("parquet", "zstd"):
            for backend in (
                ("carquet", "arrow") if index % 2 == 0 else ("arrow", "carquet")
            ):
                cases.append((backend, mode, workers))
    reports = []
    expected_counters = None
    expected_columns = None
    for repeat in range(args.repeats):
        # Rotate order when repeating to reduce drift favouring one backend.
        rotated = cases[repeat % len(cases) :] + cases[: repeat % len(cases)]
        for backend, mode, workers in rotated:
            name = f"r{repeat + 1}-{backend}-{mode}-w{workers}"
            output = args.output / f"{name}.duckdb"
            report_path = output.with_suffix(".json")
            if not report_path.exists():
                build = args.arrow_build if backend == "arrow" else args.carquet_build
                env = os.environ | {
                    "PYTHONPATH": str(build.resolve() / "python" / "Release"),
                    "PTABLE_INTERACTIVE": "0",
                }
                print(f"START {name}", flush=True)
                with output.with_suffix(".log").open("w") as log:
                    subprocess.run(
                        [
                            str(args.python),
                            str(Path(__file__).with_name("parquet_recording.py")),
                            str(args.base),
                            str(output),
                            "--mode",
                            mode,
                            "--workers",
                            str(workers),
                            "--realizations",
                            str(args.realizations),
                        ],
                        env=env,
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        check=True,
                    )
            report = json.loads(report_path.read_text())
            assert report["realizations_per_root"] == args.realizations, name
            assert report["mode"] == mode and report["workers"] == workers, name
            if expected_counters is None:
                expected_counters = report["counters_sha256"]
            assert report["counters_sha256"] == expected_counters, name
            if "column_sha256" in report:
                if expected_columns is None:
                    expected_columns = report["column_sha256"]
                assert report["column_sha256"] == expected_columns, name
            reports.append(report | {"backend": backend, "repeat": repeat + 1})
            (args.output / "results.json").write_text(
                json.dumps(reports, indent=2) + "\n"
            )
            print(
                f"DONE {name}: {report['wall_seconds']:.3f}s, "
                f"{report['peak_rss_kib'] / 2**20:.3f} GiB RSS, "
                f"{(report['sidecar_bytes'] or 0) / 2**30:.3f} GiB file",
                flush=True,
            )
            # Each case owns these fresh files; keep the small reports/logs.
            for path in (
                output,
                output.with_suffix(".events.arrow"),
                output.with_suffix(".events.parquet"),
                Path(str(output.with_suffix(".events.arrow")) + ".lock"),
            ):
                path.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
