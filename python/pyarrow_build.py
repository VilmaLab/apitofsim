"""Locate the bundled C++ libraries in the pinned PyArrow wheel."""

import json
import tomllib
from pathlib import Path

import pyarrow


def main():
    project = Path(__file__).resolve().parents[1] / "pyproject.toml"
    dependencies = tomllib.loads(project.read_text(encoding="utf-8"))["project"][
        "dependencies"
    ]
    (expected,) = [
        dep.removeprefix("pyarrow==")
        for dep in dependencies
        if dep.startswith("pyarrow==")
    ]
    if pyarrow.__version__ != expected:
        raise SystemExit(f"PyArrow {expected} is required, found {pyarrow.__version__}")
    major, minor, *_ = expected.split(".")
    print(
        json.dumps(
            {
                "version": expected,
                "libdir": str(Path(pyarrow.__file__).resolve().parent),
                "include": pyarrow.get_include(),
                "so_version": int(major) * 100 + int(minor),
            }
        )
    )


if __name__ == "__main__":
    main()
