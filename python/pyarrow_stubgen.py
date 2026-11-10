"""Load PyArrow's DLLs before nanobind imports the extension on Windows."""

import runpy
import sys

import pyarrow.parquet  # noqa: F401

runpy.run_path(sys.argv.pop(1), run_name="__main__")
