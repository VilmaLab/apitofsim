"""Convert a database-owned IPC archive to an analytical sidecar."""

import hashlib
import json
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
    EVENT_FORMATS,
    EVENT_SCHEMA,
    FLOAT_FIELDS,
    IPC_VERSION,
    PARQUET_VERSION,
    VORTEX_VERSION,
    RecordingTransaction,
    archive_metadata,
    event_schema_metadata,
    events_filename,
    recording_lock,
    sync_directory,
    sync_file,
)

_ROW_GROUP_SIZE = 65_536
_VORTEX_METADATA = "__apitofsim_"


def _load_vortex(db):
    try:
        db.load_extension("vortex")
    except duckdb.Error as error:
        raise ValueError(
            "Vortex requires DuckDB's vortex extension. On supported platforms "
            "run INSTALL vortex in the same DuckDB version before opening or converting."
        ) from error


def vortex_scan(db, path, identity=None, generation=None):
    """Validate the file schema and hide its reserved archive metadata column."""
    _load_vortex(db)
    # Keep a scan relation that can be installed as a view. DuckDB
    # materializes parameterized SELECT relations on construction.
    literal = str(path).replace("'", "''")
    # Early extension versions do not implement Hive partition discovery.
    (supports_hive,) = db.execute(
        "select exists (select 1 from duckdb_functions() where "
        "function_name='read_vortex' and list_contains(parameters, 'hive_partitioning'))"
    ).fetchall()[0]
    options = ", hive_partitioning=false" if supports_hive else ""
    try:
        scan = db.sql(f"select * from read_vortex('{literal}'{options})")
    except duckdb.IOException as error:
        if path.is_file() and any(char.isspace() for char in str(path)):
            raise ValueError(
                "The DuckDB Vortex extension could not open this path. "
                "Try a path without whitespace."
            ) from error
        raise
    columns = EVENT_SCHEMA.names
    expected_types = [
        "DOUBLE" if field.name in FLOAT_FIELDS else "BIGINT" for field in EVENT_SCHEMA
    ]
    if (
        scan.columns[:-1] != columns
        or list(map(str, scan.types)) != [*expected_types, "BOOLEAN"]
        or not scan.columns[-1].startswith(_VORTEX_METADATA)
    ):
        raise ValueError("Invalid Vortex event sidecar schema")
    # The extension has no custom file metadata API. A constant column keeps
    # archive identity in the schema even when the recording has zero rows.
    try:
        archive_id, archive_generation, ipc_size = json.loads(
            bytes.fromhex(scan.columns[-1].removeprefix(_VORTEX_METADATA))
        )
    except (ValueError, TypeError) as error:
        raise ValueError("Invalid Vortex event sidecar metadata") from error
    if (
        not isinstance(archive_id, str)
        or type(archive_generation) is not int
        or type(ipc_size) is not int
        or ipc_size <= 0
        or (identity is not None and archive_id != identity)
        or (generation is not None and archive_generation != generation)
    ):
        raise ValueError(
            "Vortex event sidecar identity/generation does not match this database"
        )
    return scan.select(", ".join(columns)), ipc_size


def vortex_batches(scan):
    # DuckDB 1.4 exposes the streaming reader under its former name.
    reader = getattr(scan, "to_arrow_reader", None)
    return (
        reader(_ROW_GROUP_SIZE) if reader else scan.fetch_record_batch(_ROW_GROUP_SIZE)
    )


def _event_fingerprint(batches):
    """Compare primitive values bit for bit, independent of batch boundaries."""
    digests = [hashlib.sha256() for _ in EVENT_SCHEMA]
    rows = 0
    for batch in batches:
        if not len(batch):
            continue
        rows += len(batch)
        for digest, column in zip(digests, batch.columns, strict=True):
            if column.null_count:
                raise ValueError(
                    "Native event fields must be non-null for Vortex conversion"
                )
            start = column.offset * 8
            values = column.buffers()[1]
            assert values is not None
            digest.update(memoryview(values)[start : start + 8 * len(column)])
    return rows, tuple(digest.digest() for digest in digests)


def _validate_vortex_payload(db, source, path, identity, generation, source_size):
    scan, ipc_size = vortex_scan(db, path, identity, generation)
    if ipc_size != source_size:
        raise ValueError("Vortex sidecar does not match the source IPC size")
    with pa.OSFile(str(source), "rb") as file, pa.ipc.open_file(file) as ipc:
        original = _event_fingerprint(
            ipc.get_batch(i) for i in range(ipc.num_record_batches)
        )
    with vortex_batches(scan) as batches:
        if _event_fingerprint(batches) != original:
            raise ValueError("Vortex event values or order do not match the IPC source")
    return original[0]


def _write_vortex(db, source, staged, identity, generation, source_size, report):
    _load_vortex(db)
    metadata = (
        _VORTEX_METADATA
        + json.dumps([identity, generation, source_size]).encode().hex()
    )
    report("Writing Vortex in IPC batch order")
    with pa.OSFile(str(source), "rb") as file, pa.ipc.open_file(file) as ipc:

        def batches():
            for index in range(ipc.num_record_batches):
                if index and index % 256 == 0:
                    report(f"Read {index:,} IPC batches for Vortex")
                yield ipc.get_batch(index)

        db.register(
            "_apitofsim_vortex_source",
            pa.RecordBatchReader.from_batches(ipc.schema, batches()),
        )
        try:
            return db.execute(
                f"COPY (SELECT *, false AS {metadata} FROM _apitofsim_vortex_source) "
                "TO ? (FORMAT VORTEX)",
                [str(staged)],
            ).fetchall()[0][0]
        finally:
            db.unregister("_apitofsim_vortex_source")


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


def _publish_sidecar(staged, destination):
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
    format: str = "parquet",
    compression: str = "none",
    progress: Callable[[str], None] | None = None,
) -> Path:
    """Convert a completed database's managed sidecar in place and return its path.

    Close readers and writers first. Re-running completes interrupted publication
    or cleanup using the already-written sidecar. Pass compression="zstd"
    for compressed Parquet or format="vortex" for Vortex's native encodings.
    """
    versions = {"parquet": PARQUET_VERSION, "vortex": VORTEX_VERSION}
    if format not in versions:
        raise ValueError("Event output format must be 'parquet' or 'vortex'")
    if compression not in ("none", "zstd"):
        raise ValueError("Event compression must be 'none' or 'zstd'")
    if format == "vortex" and compression != "none":
        raise ValueError(
            "Vortex uses native encodings. --compression applies to Parquet."
        )
    target_version = versions[format]
    format_name = {"parquet": "Parquet", "vortex": "Vortex"}[format]
    database = Path(database).resolve(strict=True)
    if not database.is_file():
        raise ValueError("Expected a file-backed event recording database")
    source = events_filename(database)
    destination = events_filename(database, format=format)

    def report(message):
        if progress is not None:
            progress(message)

    # A single DuckDB worker preserves the ordered IPC stream through COPY and
    # through the readback comparison. Parquet keeps its direct PyArrow path.
    with recording_lock(source), duckdb.connect(database) as db:
        if format == "vortex":
            db.execute("set threads=1; set preserve_insertion_order=true")
        try:
            state = db.execute(
                "select schema_version,archive_id,generation,file_size from event_recording_state"
            ).fetchone()
        except duckdb.CatalogException as error:
            raise ValueError("Expected an event recording database") from error
        if state is None:
            raise ValueError("Event recording state is missing")
        version, identity, generation, size = state

        def validate(path, generation, size):
            if format == "vortex":
                return _validate_vortex_payload(
                    db, source, path, identity, generation, size
                )
            return _validate_payload(path, identity, generation, size)

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
                    report(
                        f"Validating {format_name} left by an interrupted conversion"
                    )
                    validate(destination, generation + 1, size)
                    sync_file(destination)
                except (duckdb.Error, OSError, ValueError) as error:
                    raise FileExistsError(
                        f"{format_name} destination already exists and cannot be resumed: {destination}"
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
                    if format == "vortex":
                        rows = _write_vortex(
                            db, source, staged, identity, generation + 1, size, report
                        )
                        report("Validating Vortex schema, event values and order")
                    else:
                        rows = _write_parquet(
                            source,
                            staged,
                            identity,
                            generation + 1,
                            size,
                            compression,
                            report,
                        )
                        report(
                            "Validating Parquet schema, row count and page checksums"
                        )
                    if validate(staged, generation + 1, size) != rows:
                        raise ValueError(
                            f"{format_name} row count does not match the IPC source"
                        )
                    sync_file(staged)
                    _publish_sidecar(staged, destination)

            sync_directory(destination.parent)
            # The database marker decides which complete file readers use.
            # IPC batch offsets apply only to the source archive.
            db.execute("begin")
            db.execute("update event_execution set batch_start=0,batch_count=0")
            db.execute(
                "update event_recording_state set schema_version=?,generation=?,file_size=?",
                [target_version, generation + 1, destination.stat().st_size],
            )
            db.execute("commit")
        elif version == target_version:
            if destination.stat().st_size != size:
                raise ValueError(
                    f"{format_name} event sidecar size does not match this database"
                )
            if format == "vortex":
                _, size = vortex_scan(db, destination, identity, generation)
            else:
                with _parquet_file(destination, identity, generation) as archive:
                    metadata = archive.schema_arrow.metadata or {}
                    size = int(metadata[b"apitofsim:ipc_size"])
        elif version in EVENT_FORMATS:
            raise ValueError(
                f"Recording is already {EVENT_FORMATS[version]}. Conversion requires an Arrow IPC source."
            )
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
                if version == target_version:
                    report(
                        f"Validating the published {format_name} payload before IPC cleanup"
                    )
                    validate(destination, generation, size)
                source.unlink()
                sync_directory(source.parent)
        report(f"{format_name} sidecar ready: {destination}")
    return destination
