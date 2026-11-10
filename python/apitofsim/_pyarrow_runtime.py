"""Load the pinned PyArrow runtime before the extension in release wheels."""

from importlib import import_module
from importlib.metadata import version

from ._pyarrow_version import PYARROW_PARQUET_VERSION

if PYARROW_PARQUET_VERSION:
    installed = version("pyarrow")
    if installed != PYARROW_PARQUET_VERSION:
        raise ImportError(
            f"This apitofsim wheel requires pyarrow=={PYARROW_PARQUET_VERSION} "
            f"for native Parquet recording. Found pyarrow=={installed}."
        )
    import_module("pyarrow.parquet")
