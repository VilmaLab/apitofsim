"""Column ownership, adaptive drains, and native/control-stream contracts."""

import gc

import numpy as np
import pytest
from apitofsim import api
from test_recording_stream import prepare


def native(prepared, *, initial=2, maximum=7, n=5):
    return api.apitofsimraw.MassSpecIterator(
        prepared.mass_spec.into_cpp(),
        prepared.groups[0].substance,
        n,
        logconf=(0, True),
        initial_batch_size=initial,
        max_batch_size=maximum,
    )


def test_adaptive_native_batches_and_retained_buffers(prepared_case):
    prepared = prepare(prepared_case, pathways=[1])
    stream = native(prepared)
    stream.join_if_joinable()
    retained = []
    expected = []
    controls = []
    for index in range(4):
        batch = next(stream)
        columns = batch.columns
        size = len(columns["event_kind"])
        assert size + len(batch.items) <= (2 if index == 0 else 7)
        controls.extend(batch.items)
        if size:
            assert all(not column.flags.writeable for column in columns.values())
            retained.append(columns)
            expected.append({name: column.copy() for name, column in columns.items()})
        if len(retained) == 4:
            break
    # Retained Python owners must prevent reuse of either pool slot.
    while True:
        try:
            controls.extend(next(stream).items)
        except StopIteration:
            break
    del batch, columns, stream
    gc.collect()
    for columns, saved in zip(retained, expected, strict=True):
        for name in columns:
            np.testing.assert_array_equal(columns[name], saved[name])
    assert len(controls) == 6  # Five partial counters and exactly one final result.


def test_released_native_columns_are_reused(prepared_case):
    prepared = prepare(prepared_case, pathways=[1])
    stream = native(prepared, initial=1, maximum=1, n=1)
    stream.join_if_joinable()
    addresses = set()
    for _ in range(20):
        batch = next(stream)
        columns = batch.columns
        if len(columns["x"]):
            addresses.add(columns["x"].ctypes.data)
        del columns, batch
    assert 1 <= len(addresses) <= 2
    stream.request_cancel()
    stream.join_if_joinable()


def test_repeated_column_exports_keep_the_unique_batch_alive(prepared_case):
    prepared = prepare(prepared_case, pathways=[1])
    stream = native(prepared)
    stream.join_if_joinable()
    batch = next(stream)
    first = batch.columns
    second = batch.columns
    saved = {name: values.copy() for name, values in first.items()}
    assert all(np.shares_memory(first[name], second[name]) for name in first)
    while True:
        try:
            next(stream)
        except StopIteration:
            break
    del batch, first, stream
    gc.collect()
    for name, values in second.items():
        np.testing.assert_array_equal(values, saved[name])


@pytest.mark.parametrize("initial,maximum", [(0, 1), (1, 0), (-1, 2), (3, 2)])
def test_invalid_sizes_before_native_start(prepared_case, initial, maximum):
    prepared = prepare(prepared_case, pathways=[1])
    with pytest.raises(ValueError):
        api._mass_spec_batches(
            prepared.mass_spec,
            prepared.groups[0].substance,
            0,
            initial_batch_size=initial,
            max_batch_size=maximum,
        )
