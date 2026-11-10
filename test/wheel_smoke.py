"""Exercise the native CLI and PyArrow interoperability in every platform wheel."""

import json
import subprocess
from pathlib import Path
from tempfile import TemporaryDirectory

import apitofsim
import pyarrow as pa
from apitofsim.recording import RecordingTransaction

assert apitofsim.precompute_mesh(100.0, 1.0, quantities_strict=False).size == 100
with TemporaryDirectory() as directory:
    cli = subprocess.run(
        ["apitofsim-mass-spec"],
        input="invalid configuration\n",
        capture_output=True,
        text=True,
        cwd=directory,
        timeout=10,
        check=False,
    )
    assert "Error while reading input configuration" in cli.stderr, cli
    path = Path(directory) / "évents.arrow"
    for mode, status in [
        ("create", "cancelled"),
        ("append", "complete"),
        ("overwrite", "complete"),
    ]:
        with RecordingTransaction(apitofsim.IpcRecording(path, mode)) as recording:
            recording.executions["1"] = {"status": status}
        with pa.OSFile(str(path), "rb") as source, pa.ipc.open_file(source) as reader:
            assert reader.schema.field("event_index").type == pa.int64()
            assert reader.num_record_batches == 0
            assert (
                json.loads(reader.metadata[b"apitofsim:executions"])["1"]["status"]
                == status
            )
