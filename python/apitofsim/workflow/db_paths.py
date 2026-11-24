from dataclasses import dataclass
from os import PathLike
from pathlib import Path


@dataclass(frozen=True, init=False)
class DatabasePaths:
    """Paths owned by one workflow database directory."""

    root: Path

    def __init__(self, root: str | PathLike[str]):
        path = Path(root)
        if path.suffix == ".duckdb" or (path.exists() and not path.is_dir()):
            raise NotADirectoryError(f"Expected an APi-ToF database directory: {path}")
        object.__setattr__(self, "root", path.resolve())

    @property
    def duckdb(self):
        return self.root / "database.duckdb"

    @property
    def ase(self):
        return self.root / "ase.sqlite.db"

    @property
    def arrow(self):
        return self.root / "events.arrow"

    @property
    def event_lock(self):
        return self.arrow.with_suffix(".arrow.lock")

    @property
    def parquet(self):
        return self.root / "events.parquet"

    def histograms(self, length: int):
        return self.root / f"histograms_{length}.mmap"

    def create(self):
        self.root.mkdir()
        # Make the lock available even when the directory is served read-only.
        self.event_lock.touch()

    def require_database(self):
        if not self.duckdb.is_file():
            raise FileNotFoundError(f"No database.duckdb in {self.root}")
        return self.duckdb
