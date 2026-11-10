"""Exercise the Parquet writer and its PyArrow dependency in installed wheels."""

import os
import subprocess
import sys
from importlib.metadata import files, requires, version
from pathlib import Path

import pytest
from apitofsim._pyarrow_version import PYARROW_PARQUET_VERSION

for imports in (
    "import apitofsim; import pyarrow.parquet",
    "import pyarrow.parquet; import apitofsim",
):
    subprocess.run(
        [
            sys.executable,
            "-c",
            f"{imports}; assert apitofsim.apitofsimraw.NATIVE_PARQUET",
        ],
        check=True,
        timeout=30,
    )

assert PYARROW_PARQUET_VERSION == version("pyarrow")
assert f"pyarrow=={PYARROW_PARQUET_VERSION}" in (requires("apitofsim") or [])

installed_files = files("apitofsim")
assert installed_files is not None
assert not any(
    file.name.startswith(("libarrow", "libparquet"))
    or file.name.lower() in {"arrow.dll", "parquet.dll"}
    for file in installed_files
), "The wheel should load Arrow and Parquet from PyArrow"

test_dir = Path(__file__).resolve().parent
os.environ["DATA_DIR"] = str(test_dir / "data")
raise SystemExit(pytest.main([str(test_dir / "test_parquet_recording.py")]))
