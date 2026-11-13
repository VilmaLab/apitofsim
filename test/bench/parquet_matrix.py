"""Run parquet_recording.py sequentially against two configured CMake builds."""

import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("base", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--arrow-build", "--before-build", type=Path, required=True)
    parser.add_argument("--carquet-build", "--after-build", type=Path, required=True)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--delivery-matrix", action="store_true")
    parser.add_argument("--before-ref")
    parser.add_argument("--after-ref")
    parser.add_argument("--workers", type=int, nargs="+", default=[1, 2, 4, 8])
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--realizations", type=int, default=5000)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.delivery_matrix:
        assert len(args.workers) == 1, "Choose one Parquet worker count"
        cases = []
        for index, (mode, callback) in enumerate(
            [("arrow", False), ("arrow", True), ("zstd", False), ("zstd", True)]
        ):
            for backend in (
                ("before", "after") if index % 2 == 0 else ("after", "before")
            ):
                cases.append(
                    (backend, mode, args.workers[0] if mode == "zstd" else 1, callback)
                )
    else:
        cases = [("arrow", "summary", 1, False), ("arrow", "arrow", 1, False)]
        for index, workers in enumerate(args.workers):
            for mode in ("parquet", "zstd"):
                for backend in (
                    ("carquet", "arrow") if index % 2 == 0 else ("arrow", "carquet")
                ):
                    cases.append((backend, mode, workers, False))
    reports = []
    expected_counters = None
    expected_columns = None
    for repeat in range(args.repeats):
        # Rotate order when repeating to reduce drift favouring one backend.
        rotated = cases[repeat % len(cases) :] + cases[: repeat % len(cases)]
        for backend, mode, workers, callback in rotated:
            name = f"r{repeat + 1}-{backend}-{mode}-w{workers}" + (
                "-python" if callback else ""
            )
            output = args.output / f"{name}.duckdb"
            report_path = output.with_suffix(".json")
            if not report_path.exists():
                build = (
                    args.arrow_build
                    if backend in ("arrow", "before")
                    else args.carquet_build
                )
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
                        ]
                        + (["--python-events"] if callback else []),
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
            assert report.get("python_events", False) == callback, name
            build = (
                args.arrow_build
                if backend in ("arrow", "before")
                else args.carquet_build
            )
            binary = next((build / "python/Release/apitofsim").glob("apitofsimraw*.so"))
            reports.append(
                report
                | {
                    "backend": backend,
                    "repeat": repeat + 1,
                    "revision": args.before_ref
                    if backend in ("arrow", "before")
                    else args.after_ref,
                    "binary_sha256": hashlib.sha256(binary.read_bytes()).hexdigest(),
                }
            )
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
