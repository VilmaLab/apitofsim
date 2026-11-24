"""Storage for fixed-length rows of native doubles."""

import mmap
import os
from collections.abc import Sequence
from pathlib import Path
from threading import Lock

import numpy
import portalocker

from .db_paths import DatabasePaths

_DTYPE = numpy.dtype(numpy.float64)


class HistogramStore:
    """Share a read-only mapping cache between database connections in one process."""

    def __init__(self, paths: DatabasePaths | None = None, *, readonly: bool = False):
        self.paths = paths
        self.readonly = readonly
        self._mappings: dict[Path, mmap.mmap] = {}
        self._mapping_lock = Lock()
        self._memory_rows: dict[int, list[numpy.ndarray]] = {}
        # Windows requires unmapping a file before extending it. Only a
        # read-only store retains mappings there.
        self._use_mmap = readonly or os.name != "nt"

    def append(self, values: numpy.ndarray) -> range:
        """Persist a batch before its row numbers can be published in DuckDB."""
        if self.readonly:
            raise PermissionError("Cannot append to a read-only histogram store")
        values = numpy.ascontiguousarray(values, dtype=_DTYPE)
        count, length = values.shape
        if length == 0 or count == 0:
            raise ValueError("Expected a nonempty batch of histograms")
        if self.paths is None:
            with self._mapping_lock:
                rows = self._memory_rows.setdefault(length, [])
                start = len(rows)
                copy = values.copy()
                copy.flags.writeable = False
                rows.extend(copy)
                return range(start, start + count)

        from apitofsim.recording import sync_directory

        path = self.paths.histograms(length)
        stride = length * _DTYPE.itemsize
        with portalocker.Lock(
            self.paths.root / "histograms.lock", mode="a", timeout=60
        ):
            path.touch(exist_ok=True)
            with path.open("r+b") as file:
                size = os.fstat(file.fileno()).st_size
                # A failed append can leave an unreferenced tail. Reserve whole
                # rows and never reuse addresses that might have been published.
                start = (size + stride - 1) // stride
                end = start + count
                file.truncate(end * stride)
                file.seek(start * stride)
                file.write(memoryview(values).cast("B"))
                file.flush()
                os.fsync(file.fileno())
            sync_directory(path.parent)
        return range(start, end)

    def _mapping(self, path: Path, required: int = 0) -> mmap.mmap | None:
        with self._mapping_lock:
            mapping = self._mappings.get(path)
            if mapping is not None and len(mapping) >= required:
                return mapping
            with path.open("rb") as file:
                size = os.fstat(file.fileno()).st_size
                if size < required:
                    raise ValueError(f"Histogram row exceeds the size of {path}")
                if not size:
                    return None
                mapping = mmap.mmap(file.fileno(), size, access=mmap.ACCESS_READ)
            # Exported arrays retain their mapping even when the cache is replaced.
            self._mappings[path] = mapping
            return mapping

    def premap(self):
        """Map the available files without reading their pages."""
        if self.paths is not None and self._use_mmap:
            for path in self.paths.root.glob("histograms_*.mmap"):
                if path.stem.removeprefix("histograms_").isdigit():
                    self._mapping(path)

    def release_mappings(self):
        with self._mapping_lock:
            self._mappings.clear()

    def read_rows(self, length: int, rows: Sequence[int]) -> list[numpy.ndarray]:
        """Prefetch the selected file pages and return read-only rows in request order."""
        if not rows:
            return []
        if length <= 0 or min(rows) < 0:
            raise ValueError("Invalid histogram address")
        if self.paths is None:
            stored = self._memory_rows.get(length, [])
            if max(rows) >= len(stored):
                raise ValueError(
                    "Histogram row exceeds the size of the in-memory store"
                )
            return [stored[row] for row in rows]

        path = self.paths.histograms(length)
        stride = length * _DTYPE.itemsize
        required = (max(rows) + 1) * stride
        if not self._use_mmap:
            with path.open("rb") as file:
                if os.fstat(file.fileno()).st_size < required:
                    raise ValueError(f"Histogram row exceeds the size of {path}")
                result = []
                for row in rows:
                    file.seek(row * stride)
                    result.append(numpy.frombuffer(file.read(stride), dtype=_DTYPE))
                return result

        mapping = self._mapping(path, required)
        assert mapping is not None
        _prefetch(mapping, stride, rows)
        return [
            numpy.frombuffer(mapping, dtype=_DTYPE, count=length, offset=row * stride)
            for row in rows
        ]


def _prefetch(mapping: mmap.mmap, stride: int, rows: Sequence[int]):
    if not hasattr(mapping, "madvise") or not hasattr(mmap, "MADV_WILLNEED"):
        return
    page = mmap.PAGESIZE
    start = end = 0
    try:
        for row in sorted(set(rows)):
            left = row * stride // page * page
            right = min(len(mapping), ((row + 1) * stride + page - 1) // page * page)
            if left > end:
                if end > start:
                    mapping.madvise(mmap.MADV_WILLNEED, start, end - start)
                start = left
            end = right
        mapping.madvise(mmap.MADV_WILLNEED, start, end - start)
    except OSError:
        # Advice is optional even when the kernel exposes the syscall.
        pass
