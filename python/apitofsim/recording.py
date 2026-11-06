"""Native IPC destinations and recoverable transactions for event sidecars."""

import base64
import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

import portalocker
import pyarrow as pa

from . import apitofsimraw as raw

FLOAT_FIELDS = (
    "x y z t vx vy vz ox oy oz rot_energy vibrational_energy theta u_norm".split()
)
INDEX_FIELDS = "realization_index event_kind particle_index transition_index next_particle_index accepted result_id event_index".split()
EVENT_SCHEMA = pa.schema(
    [(name, pa.float64()) for name in FLOAT_FIELDS]
    + [(name, pa.int64()) for name in INDEX_FIELDS]
)


def event_batch_dataset(path, ranges, columns=None):
    """Expose selected IPC batches as a reusable, projectable Arrow dataset."""
    import pyarrow.dataset as ds

    if columns is None:
        schema = EVENT_SCHEMA
        options = None
    else:
        schema = pa.schema([EVENT_SCHEMA.field(name) for name in columns])
        options = pa.ipc.IpcReadOptions(
            included_fields=[EVENT_SCHEMA.get_field_index(name) for name in columns]
        )
    if not ranges:
        return ds.dataset(pa.Table.from_batches([], schema=schema))

    # The native IPC writer emits uncompressed buffers, so mmap keeps batch
    # bodies file-backed. On Windows, live mappings would prevent an append
    # from resizing the archive even after the database releases its lock.
    open_source = pa.OSFile if os.name == "nt" else pa.memory_map
    with (
        open_source(str(path), "r") as source,
        pa.ipc.open_file(source, options=options) as reader,
    ):
        batches = [
            reader.get_batch(index)
            for start, count in ranges
            for index in range(start, start + count)
        ]
    return ds.dataset(batches)


def events_filename(database):
    path = Path(database)
    return (
        path.with_suffix(".events.arrow")
        if path.suffix == ".duckdb"
        else Path(str(path) + ".events.arrow")
    )


def sync_file(path):
    # Windows' FlushFileBuffers requires a handle with write access.
    with Path(path).open("r+b") as file:
        os.fsync(file.fileno())


def sync_directory(path):
    # Windows has no fsync-directory interface; os.replace supplies atomic
    # publication there. Every data/journal file is still explicitly synced.
    if os.name != "nt":
        fd = os.open(path, os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)


def recording_lock(path, *, shared=False):
    return portalocker.Lock(
        str(path) + ".lock",
        mode="a+b",
        timeout=0,
        flags=(portalocker.LOCK_SH if shared else portalocker.LOCK_EX)
        | portalocker.LOCK_NB,
    )


def archive_metadata(path):
    with pa.OSFile(str(path), "rb") as source, pa.ipc.open_file(source) as reader:
        metadata = reader.schema.metadata or {}
        if metadata.get(b"apitofsim:event_schema") != b"1":
            raise ValueError(
                "Unsupported event archive schema; an explicit migration is required"
            )
        if not reader.schema.equals(EVENT_SCHEMA, check_metadata=False):
            # Non-nullability is part of the on-disk schema, not pa.schema's
            # nullable defaults used to construct an empty scan.
            if [(f.name, f.type) for f in reader.schema] != [
                (f.name, f.type) for f in EVENT_SCHEMA
            ]:
                raise ValueError("Incompatible event archive schema")
        identity = metadata[b"apitofsim:archive_id"].decode()
        executions = json.loads(
            (reader.metadata or {}).get(b"apitofsim:executions", b"{}")
        )
        next_id = max((int(k) for k in executions), default=0) + 1
        return identity, next_id, reader.num_record_batches, executions


@dataclass(frozen=True)
class IpcRecording:
    """An explicit standalone destination; existing files require an explicit mode."""

    path: str | Path
    mode: str = "create"

    def __post_init__(self):
        if self.mode not in ("create", "append", "overwrite"):
            raise ValueError("Recording mode must be create, append or overwrite")


@dataclass(frozen=True)
class RecordingInfo:
    path: Path
    archive_id: str
    result_id: int
    event_count: int
    batch_start: int
    batch_count: int
    status: str


class RecordingTransaction:
    """One writer, with payload durability preceding metadata publication.

    The journal contains the overwritten ending, not previous event payloads.
    Database generations decide recovery; standalone archives use a committed
    journal marker. Only archives produced by this implementation are appended.
    """

    def __init__(self, destination, *, database=None):
        self.spec = (
            destination
            if isinstance(destination, IpcRecording)
            else IpcRecording(destination)
        )
        self.path = Path(self.spec.path).resolve()
        self.database = database
        self.work_path = (
            Path(str(self.path) + ".replacement")
            if self.spec.mode == "overwrite"
            else self.path
        )
        self.journal_path = Path(str(self.work_path) + ".journal")
        self.lock = recording_lock(self.path)
        self.writer = None
        self.done = False
        self.committed = False
        self.started = False
        self.journal = None
        self.archive_id = ""
        self.next_result_id = 1
        self.generation = 0
        self.executions = {}
        self.failure_rollbacks = []
        self.active_execution = None

    def _write_journal(self):
        temporary = Path(str(self.journal_path) + ".new")
        with temporary.open("wb") as file:
            file.write(json.dumps(self.journal).encode())
            file.flush()
            os.fsync(file.fileno())
        os.replace(temporary, self.journal_path)
        sync_directory(self.path.parent)

    @staticmethod
    def recover(path, *, generation=None, expected_archive_id=None):
        """Called only with the archive's exclusive lock held."""
        path = Path(path)
        journal_path = Path(str(path) + ".journal")
        if not journal_path.exists():
            return
        data = json.loads(journal_path.read_bytes())
        if generation is None and data.get("owner") == "database":
            raise ValueError(
                "Recover this sidecar by opening its matching database first"
            )
        if data["version"] != 1:
            raise ValueError("Unsupported event recovery journal")
        if (
            expected_archive_id is not None
            and data["archive_id"] != expected_archive_id
        ):
            raise ValueError("Event recovery journal belongs to a different database")
        if generation is not None and generation not in (
            data["generation"] - 1,
            data["generation"],
        ):
            raise ValueError("Event journal generation does not match this database")
        committed = (
            data.get("committed", False)
            if generation is None
            else generation == data["generation"]
        )
        if committed:
            if not path.exists() or path.stat().st_size != data["new_size"]:
                raise ValueError("Committed event archive is missing or truncated")
            identity, _, _, _ = archive_metadata(path)
            if identity != data["archive_id"]:
                raise ValueError("Event archive identity mismatch during recovery")
        elif data["old_size"]:
            tail = base64.b64decode(data["tail"], validate=True)
            if (
                hashlib.sha256(tail).hexdigest() != data["tail_sha256"]
                or len(tail) != data["old_size"] - data["offset"]
            ):
                raise ValueError("Corrupt event recovery journal")
            with path.open("r+b") as file:
                file.seek(data["offset"])
                file.write(tail)
                file.truncate(data["old_size"])
                file.flush()
                os.fsync(file.fileno())
        else:
            path.unlink(missing_ok=True)
        journal_path.unlink()
        Path(str(journal_path) + ".new").unlink(missing_ok=True)
        sync_directory(path.parent)
        return committed

    def __enter__(self):
        self.lock.acquire()
        try:
            db = self.database
            if db is not None:
                identity, generation, size = db.db.execute(
                    "select archive_id,generation,file_size from event_recording_state"
                ).fetchall()[0]
                self.archive_id, self.generation = identity, generation + 1
                self.recover(
                    self.path, generation=generation, expected_archive_id=identity
                )
                if (self.path.stat().st_size if self.path.exists() else 0) != size:
                    raise ValueError(
                        "Event sidecar does not match the database's committed size"
                    )
            else:
                self.recover(self.path)
                self.recover(Path(str(self.path) + ".replacement"))
            if self.spec.mode == "overwrite":
                if self.path.exists():
                    try:
                        with (
                            pa.OSFile(str(self.path), "rb") as source,
                            pa.ipc.open_file(source) as existing,
                        ):
                            identity = (existing.schema.metadata or {}).get(
                                b"apitofsim:archive_id", b""
                            )
                    except pa.ArrowInvalid:
                        identity = b""
                    if identity.startswith(b"database:"):
                        raise ValueError(
                            "Overwrite database-owned sidecars through their database workflow"
                        )
                self.recover(self.work_path)
                if self.work_path.exists():
                    raise FileExistsError(
                        f"Unmanaged recording staging file already exists: {self.work_path}"
                    )
            exists = self.work_path.exists()
            if self.spec.mode == "create" and exists:
                raise FileExistsError(
                    f"Event archive already exists: {self.path}; choose append or overwrite explicitly"
                )
            if exists:
                identity, self.next_result_id, _, self.executions = archive_metadata(
                    self.work_path
                )
                if self.archive_id and identity != self.archive_id:
                    raise ValueError("Event sidecar belongs to a different database")
                if db is None and identity.startswith("database:"):
                    raise ValueError(
                        "Append database-owned sidecars through their database workflow"
                    )
                self.archive_id = identity
            elif not self.archive_id:
                self.archive_id = "standalone:" + str(uuid4())
            old_size, offset, tail = 0, 0, b""
            if exists:
                old_size = self.work_path.stat().st_size
                with self.work_path.open("rb") as file:
                    file.seek(-10, os.SEEK_END)
                    footer_size = int.from_bytes(file.read(4), "little")
                    offset = old_size - footer_size - 18
                    if offset < 8:
                        raise ValueError("Invalid event archive ending")
                    file.seek(offset)
                    tail = file.read()
                if tail[:8] != b"\xff\xff\xff\xff\0\0\0\0":
                    raise ValueError(
                        "Unsupported IPC ending; explicit conversion is required"
                    )
            self.journal = dict(
                version=1,
                owner="database" if db is not None else "standalone",
                archive_id=self.archive_id,
                generation=self.generation,
                old_size=old_size,
                offset=offset,
                tail=base64.b64encode(tail).decode(),
                tail_sha256=hashlib.sha256(tail).hexdigest(),
            )
            self._write_journal()
            if db is not None:
                db.db.execute("begin")
                self.started = True
            self.writer = raw.IpcEventWriter(
                str(self.work_path), exists, self.archive_id
            )
            return self
        except BaseException:
            self.rollback()
            raise

    def finish(self):
        if self.done:
            return
        assert self.writer is not None
        assert self.journal is not None
        self.writer.set_metadata(json.dumps(self.executions))
        self.writer.close()
        sync_file(self.work_path)
        sync_directory(self.path.parent)
        self.journal["new_size"] = self.work_path.stat().st_size
        self.journal["committed"] = (
            self.database is None and self.spec.mode != "overwrite"
        )
        self._write_journal()
        db = self.database
        if db is not None:
            db.db.execute(
                "update event_recording_state set generation=?,file_size=?",
                [self.generation, self.journal["new_size"]],
            )
            db.db.execute("commit")
            self.started = False
        if self.spec.mode == "overwrite":
            os.replace(self.work_path, self.path)
            sync_directory(self.path.parent)
        self.done = True
        self.committed = True
        self.failure_rollbacks.clear()
        try:
            self.journal_path.unlink()
            sync_directory(self.path.parent)
        except OSError:
            # Publication is already durable. A remaining journal is harmless
            # and will be verified/cleaned on the next open.
            pass

    def rollback(self):
        if self.done:
            self.lock.release()
            return
        if self.writer is not None:
            self.writer.abort()
        self.writer = None
        if self.started:
            import duckdb

            assert self.database is not None
            try:
                self.database.db.execute("rollback")
            except duckdb.TransactionException as error:
                if "no transaction is active" not in str(error).lower():
                    raise
            self.started = False
        # COMMIT may have succeeded just before an interruption. The actual
        # database generation is authoritative; never undo committed payloads.
        generation = None
        if self.database is not None:
            generation = self.database.db.execute(
                "select generation from event_recording_state"
            ).fetchall()[0][0]
        if generation is not None:
            self.committed = generation == self.generation
        try:
            if self.journal is not None:
                self.committed = bool(
                    self.recover(
                        self.work_path,
                        generation=generation,
                        expected_archive_id=self.archive_id,
                    )
                )
            self.done = True
        finally:
            callbacks, self.failure_rollbacks = self.failure_rollbacks, []
            if not self.committed:
                for clear_failure in callbacks:
                    clear_failure()
        self.lock.release()

    def __exit__(self, exc_type, exc, tb):
        try:
            if not self.done:
                if exc is None:
                    self.finish()
                else:
                    self.rollback()
        except BaseException:
            self.rollback()
            raise
        finally:
            self.writer = None
            self.lock.release()
