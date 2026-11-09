"""Convert a database-owned IPC archive to a clustered Parquet sidecar."""

import os
import shutil
import sys
from collections.abc import Callable
from pathlib import Path
from tempfile import TemporaryDirectory

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq

from .recording import (
    FLOAT_FIELDS,
    IPC_VERSION,
    PARQUET_VERSION,
    RecordingTransaction,
    archive_metadata,
    event_schema_metadata,
    events_filename,
    recording_lock,
    sync_directory,
    sync_file,
)

_ROW_GROUP_SIZE = 65_536


def _parquet_file(path, identity, generation):
    # Checksum validation scans every row group. Keep Arrow from prefetching
    # the full payload when opening the batch reader.
    archive = pq.ParquetFile(path, pre_buffer=False, page_checksum_verification=True)
    try:
        metadata = event_schema_metadata(archive.schema_arrow)
        if (
            metadata[b"apitofsim:archive_id"] != identity.encode()
            or metadata.get(b"apitofsim:generation") != str(generation).encode()
        ):
            raise ValueError(
                "Parquet event sidecar schema/identity/generation does not match this database"
            )
        return archive
    except BaseException:
        archive.close()
        raise


def validate_parquet(path, identity, generation):
    """Check the Parquet footer before opening a scan."""
    with _parquet_file(path, identity, generation):
        pass


def _validate_payload(path, identity, generation, source_size):
    with _parquet_file(path, identity, generation) as archive:
        metadata = archive.schema_arrow.metadata or {}
        if metadata.get(b"apitofsim:ipc_size") != str(source_size).encode():
            raise ValueError("Parquet sidecar does not match the source IPC size")
        rows = sum(len(batch) for batch in archive.iter_batches())
        if rows != archive.metadata.num_rows:
            raise ValueError("Incomplete Parquet event sidecar")
        return rows


def _clear_staging(destination, owner):
    prefix = f".{destination.name}.convert-"
    for path in destination.parent.iterdir():
        if path.name.startswith(prefix) and not path.is_symlink() and path.is_dir():
            marker = path / "owner"
            if (
                not marker.is_symlink()
                and marker.is_file()
                and marker.read_bytes() == owner
            ):
                shutil.rmtree(path)


def _publish_parquet(staged, destination):
    """Install the synced file without replacing an existing destination."""
    if sys.platform == "win32":
        import ctypes
        from ctypes import wintypes

        move_file = ctypes.WinDLL("kernel32", use_last_error=True).MoveFileExW
        move_file.argtypes = [wintypes.LPCWSTR, wintypes.LPCWSTR, wintypes.DWORD]
        move_file.restype = wintypes.BOOL
        write_through = 0x8  # MOVEFILE_WRITE_THROUGH
        if not move_file(str(staged), str(destination), write_through):
            raise ctypes.WinError(ctypes.get_last_error())
    else:
        os.link(staged, destination)


def _write_parquet(
    source, staged, identity, generation, source_size, compression, report
):
    with pa.OSFile(str(source), "rb") as file, pa.ipc.open_file(file) as ipc:
        metadata = {
            **(ipc.schema.metadata or {}),
            b"apitofsim:archive_id": identity.encode(),
            b"apitofsim:generation": str(generation).encode(),
            b"apitofsim:ipc_size": str(source_size).encode(),
            b"apitofsim:ipc_executions": (ipc.metadata or {}).get(
                b"apitofsim:executions", b"{}"
            ),
        }
        report(f"Writing Parquet in IPC batch order (compression: {compression})")
        rows = 0
        with pq.ParquetWriter(
            staged,
            ipc.schema.with_metadata(metadata),
            compression=compression,
            # PyArrow accepts per-column lists for these encoder options.
            use_dictionary=[  # pyright: ignore[reportArgumentType]
                "event_kind",
                "result_id",
                "accepted",
                "particle_index",
                "next_particle_index",
                "transition_index",
            ],
            use_byte_stream_split=FLOAT_FIELDS if compression == "zstd" else False,  # pyright: ignore[reportArgumentType]
            write_page_checksum=True,
        ) as writer:
            for index in range(ipc.num_record_batches):
                batch = ipc.get_batch(index)
                if len(batch):
                    # Keep execution boundaries from the recorder and split
                    # older, larger IPC batches without rearranging their rows.
                    writer.write_batch(batch, row_group_size=_ROW_GROUP_SIZE)
                    rows += len(batch)
                if (index + 1) % 256 == 0:
                    report(f"Converted {index + 1:,} IPC batches ({rows:,} events)")
    return rows


def convert_events(
    database: str | Path,
    *,
    compression: str = "none",
    progress: Callable[[str], None] | None = None,
) -> Path:
    """Convert a completed database's managed sidecar in place and return its path.

    Close readers and writers first. Re-running completes interrupted publication
    or cleanup using the already-written Parquet file. The archive directory
    needs space for the output. Pass compression="zstd" to compress it.
    """
    if compression not in ("none", "zstd"):
        raise ValueError("Event compression must be 'none' or 'zstd'")
    database = Path(database).resolve(strict=True)
    if not database.is_file():
        raise ValueError("Expected a file-backed event recording database")
    source = events_filename(database)
    destination = events_filename(database, parquet=True)

    def report(message):
        if progress is not None:
            progress(message)

    with recording_lock(source), duckdb.connect(database) as db:
        try:
            state = db.execute(
                "select schema_version,archive_id,generation,file_size from event_recording_state"
            ).fetchone()
        except duckdb.CatalogException as error:
            raise ValueError("Expected an event recording database") from error
        if state is None:
            raise ValueError("Event recording state is missing")
        version, identity, generation, size = state

        if version == IPC_VERSION:
            if source.is_symlink():
                raise ValueError(
                    "Convert the recording using a regular IPC sidecar beside its database"
                )
            RecordingTransaction.recover(
                source, generation=generation, expected_archive_id=identity
            )
            if not size or not source.is_file():
                raise FileNotFoundError("No matching .events.arrow sidecar to convert")
            if source.stat().st_size != size or archive_metadata(source)[0] != identity:
                raise ValueError(
                    "IPC event sidecar identity/size does not match this database"
                )

            owner = f"{identity}:{generation + 1}".encode()
            _clear_staging(destination, owner)
            if os.path.lexists(destination):
                # A crash after publishing the file but before committing DuckDB
                # can leave a complete destination. Verify it before resuming.
                try:
                    if destination.is_symlink():
                        raise ValueError("Destination is a symbolic link")
                    report("Validating Parquet left by an interrupted conversion")
                    _validate_payload(destination, identity, generation + 1, size)
                    sync_file(destination)
                except (OSError, ValueError) as error:
                    raise FileExistsError(
                        f"Parquet destination already exists and cannot be resumed: {destination}"
                    ) from error
            else:
                free = shutil.disk_usage(destination.parent).free
                report(
                    f"IPC is {size / 2**30:.2f} GiB, with {free / 2**30:.2f} GiB free. "
                    "Allow roughly the IPC size for uncompressed output "
                    "alongside the source. Actual size depends on the data and compression."
                )
                with TemporaryDirectory(
                    dir=destination.parent, prefix=f".{destination.name}.convert-"
                ) as scratch:
                    scratch = Path(scratch)
                    marker = scratch / "owner"
                    marker.write_bytes(owner)
                    sync_file(marker)
                    sync_directory(scratch)
                    sync_directory(scratch.parent)
                    staged = scratch / destination.name
                    rows = _write_parquet(
                        source,
                        staged,
                        identity,
                        generation + 1,
                        size,
                        compression,
                        report,
                    )
                    report("Validating Parquet schema, row count and page checksums")
                    if (
                        _validate_payload(staged, identity, generation + 1, size)
                        != rows
                    ):
                        raise ValueError(
                            "Parquet row count does not match the IPC source"
                        )
                    sync_file(staged)
                    _publish_parquet(staged, destination)

            sync_directory(destination.parent)
            # The database marker decides which complete file readers use.
            # IPC batch offsets have no meaning once Parquet is authoritative.
            db.execute("begin")
            db.execute("update event_execution set batch_start=0,batch_count=0")
            db.execute(
                "update event_recording_state set schema_version=?,generation=?,file_size=?",
                [PARQUET_VERSION, generation + 1, destination.stat().st_size],
            )
            db.execute("commit")
        elif version == PARQUET_VERSION:
            if destination.stat().st_size != size:
                raise ValueError(
                    "Parquet event sidecar size does not match this database"
                )
            with _parquet_file(destination, identity, generation) as archive:
                metadata = archive.schema_arrow.metadata or {}
                size = int(metadata[b"apitofsim:ipc_size"])
        else:
            raise ValueError(
                "Unsupported event recording schema; explicit migration is required"
            )

        # Read-only database snapshots copy the DuckDB file itself, so persist
        # the switch there before removing the source.
        db.execute("checkpoint")
        if (
            not source.is_symlink()
            and source.is_file()
            and source.stat().st_size == size
        ):
            try:
                source_identity = archive_metadata(source)[0]
            except (OSError, ValueError, KeyError):
                source_identity = None
            if source_identity == identity:
                if version == PARQUET_VERSION:
                    report(
                        "Validating the published Parquet payload before IPC cleanup"
                    )
                    _validate_payload(destination, identity, generation, size)
                source.unlink()
                sync_directory(source.parent)
        report(f"Parquet sidecar ready: {destination}")
    return destination
