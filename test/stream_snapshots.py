"""Decode Arrow batches into test snapshots for existing physics assertions.

Production recording must never use this per-row conversion.
"""

from types import SimpleNamespace

import numpy as np
import pyarrow as pa
from apitofsim import api

NAMES = (
    "InitEvent",
    "CollisionEvent",
    "PostCollisionEvent",
    "FragmentationEvent",
    "PostFragmentationEvent",
    "EscapeEvent",
)


class SnapshotType(type):
    def __instancecheck__(cls, value):
        return type(value).__name__ == cls.__name__


for _name in NAMES:
    globals()[_name] = SnapshotType(_name, (SimpleNamespace,), {})


def event_rows(table):
    for row in table.to_pylist():
        state = SimpleNamespace(
            realization=row["realization"],
            postime=np.array([row[k] for k in ("x", "y", "z", "t")]),
            velocity=np.array([row[k] for k in ("vx", "vy", "vz")]),
            omega=np.array([row[k] for k in ("ox", "oy", "oz")]),
            rot_energy=row["rot_energy"],
            vibrational_energy=row["vibrational_energy"],
            particle_index=row["particle_index"],
        )
        event = globals()[NAMES[row["event_kind"]]](state=state)
        if row["event_kind"] == 1:
            event.theta, event.u_norm, event.accepted = (
                row["theta"],
                row["u_norm"],
                bool(row["accepted"]),
            )
        elif row["event_kind"] == 3:
            event.pathway_index = row["transition_index"]
            event.next_particle_index = row["next_particle_index"]
        yield event


def event_table(events=()):
    values = {
        k: []
        for k in (
            "x",
            "y",
            "z",
            "t",
            "vx",
            "vy",
            "vz",
            "ox",
            "oy",
            "oz",
            "rot_energy",
            "vibrational_energy",
            "theta",
            "u_norm",
        )
    }
    indices = {
        k: []
        for k in (
            "realization",
            "event_kind",
            "particle_index",
            "transition_index",
            "next_particle_index",
            "accepted",
        )
    }
    for event in events:
        state = event.state
        row = (
            *state.postime,
            *state.velocity,
            *state.omega,
            state.rot_energy,
            state.vibrational_energy,
            getattr(event, "theta", 0),
            getattr(event, "u_norm", 0),
        )
        for name, value in zip(values, row, strict=True):
            values[name].append(value)
        ids = (
            state.realization,
            NAMES.index(type(event).__name__),
            state.particle_index,
            getattr(event, "pathway_index", -1),
            getattr(event, "next_particle_index", -1),
            int(getattr(event, "accepted", False)),
        )
        for name, value in zip(indices, ids, strict=True):
            indices[name].append(value)
    return pa.table(
        {
            **{k: pa.array(v, type=pa.float64()) for k, v in values.items()},
            **{k: pa.array(v, type=pa.int64()) for k, v in indices.items()},
        }
    )


def batch_from_items(items):
    if len(items) == 1 and isinstance(items[0], api.MassSpecBatch):
        return items[0]
    return api.MassSpecBatch(
        event_table(x for x in items if hasattr(x, "state")),
        [x for x in items if not hasattr(x, "state")],
    )


def flatten_batches(stream):
    for item in stream:
        if isinstance(item, api.MassSpecBatch):
            yield from event_rows(item.events)
            yield from item.items
        else:
            yield item
