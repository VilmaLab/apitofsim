"""Exercise the private Arrow runtime in an installed, repaired wheel."""

import os
import platform
import subprocess
import sys
from importlib.metadata import distribution
from pathlib import Path

import pytest

wheel = distribution("apitofsim")
assert wheel.files is not None
if sys.platform == "darwin" and platform.machine() == "arm64":
    assert "macosx_12_0_arm64" in (wheel.read_text("WHEEL") or "")

prefix, suffix = {
    "linux": ("lib", ".so"),
    "darwin": ("lib", ".dylib"),
    "win32": ("", ".dll"),
}[sys.platform]
library_dirs = set()
for name in ("arrow", "parquet"):
    libraries = [
        file.locate()
        for file in wheel.files
        if file.name.startswith(prefix + name) and suffix in file.name
    ]
    assert libraries, f"The wheel must bundle {name}"
    library_dirs.update(str(file.parent) for file in libraries)

# The package imports PyArrow for its Python APIs. Load the native extension on
# its own to check that its DLLs and shared libraries work without those imports.
module_dir = str(wheel.locate_file("apitofsim"))
subprocess.run(
    [
        sys.executable,
        "-I",
        "-S",
        "-c",
        f"""
import os
import sys
if sys.platform == "win32":
    dll_dirs = [os.add_dll_directory(path) for path in {sorted(library_dirs)!r}]
sys.path.insert(0, {module_dir!r})
import apitofsimraw
assert apitofsimraw.NATIVE_PARQUET
""",
    ],
    check=True,
    timeout=30,
)

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

test_dir = Path(__file__).resolve().parent
os.environ["DATA_DIR"] = str(test_dir / "data")
raise SystemExit(pytest.main([str(test_dir / "test_parquet_recording.py")]))
