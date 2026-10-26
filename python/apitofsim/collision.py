"""Reusable instantaneous collision-frequency profiles; no trajectories or DB writes."""

from copy import deepcopy

import numpy as np

from . import apitofsimraw
from .api import Q_, ClusterData, MassSpectrometer, QuantityProcessor
from .config import import_raw_config


class CollisionProfile:
    """Reusable collision physics prepared by :func:`collision_profile`."""

    def __init__(self, mass_spec, gas, cluster_radius):
        self._profile = apitofsimraw.CollisionProfile(
            mass_spec.into_cpp(), gas.into_cpp(), cluster_radius
        )

    def frequency(self, z, velocity=None, *, quantities_strict=True):
        """Return total collision frequency (including rejected collisions) in 1/s.

        Positions have shape (n,); velocity is an instrument-frame (vx, vy, vz)
        vector of shape (3,) or (n, 3), defaulting to zero. Both accept Pint
        quantities; plain SI arrays require quantities_strict=False.
        Unsorted, repeated and empty positions are supported. Invalid shapes,
        nonfinite values and positions outside the instrument raise ValueError.
        Both skimmer endpoints belong to the skimmer; gas flow is subtracted
        from particle velocity there. Collision-event velocities are pre-update.
        """
        process = QuantityProcessor(quantities_strict)
        z = np.asarray(process("z", z, "m"), dtype=float)
        if z.ndim != 1:
            raise ValueError("z must be a 1-D array")
        velocity = (
            np.zeros(3)
            if velocity is None
            else np.asarray(process("velocity", velocity, "m/s"), dtype=float)
        )
        if not np.isfinite(velocity).all():
            raise ValueError("velocity must be finite")
        if velocity.shape == (3,):
            velocity = np.broadcast_to(velocity, (len(z), 3))
        elif velocity.shape != (len(z), 3):
            raise ValueError("velocity must have shape (3,) or (len(z), 3)")
        return Q_(self._profile.frequency(z, np.asfortranarray(velocity)), "1/s")


def collision_profile(config, cluster, *, skimmer=None, mesh_skimmer=None):
    """Prepare reusable collision physics from an effective experiment config.

    Supply ClusterData or a Pint collision radius and a prepared skimmer array
    from skimmer(config). Three-column SI skimmer data [velocity, temperature,
    pressure] also accepts a Pint mesh_skimmer spacing. Config may be normalized
    or serialized. Evaluation neither runs trajectories nor accesses a database.
    """
    missing = [k for k in ("lengths", "pressures", "T", "gas") if config.get(k) is None]
    if missing:
        raise ValueError(
            f"Missing collision profile configuration: {', '.join(missing)}"
        )
    config = import_raw_config(deepcopy(config))
    if skimmer is None:
        raise ValueError("Prepared skimmer required; call skimmer(config)")
    lengths = np.asarray(config["lengths"].to("m").magnitude, dtype=float)
    pressures = np.asarray(config["pressures"].to("Pa").magnitude, dtype=float)
    temperature = config["T"].to("K").magnitude
    gas = config["gas"]
    mass = gas.mass.to("kg").magnitude
    gas_radius = gas.radius.to("m").magnitude
    radius = (
        apitofsimraw.compute_derived(cluster.into_cpp()).radius
        if isinstance(cluster, ClusterData)
        else QuantityProcessor()("cluster", cluster, "m")
    )
    if (
        lengths.shape != (5,)
        or not np.isfinite(lengths).all()
        or (lengths < 0).any()
        or lengths[-1] <= 0
        or pressures.shape != (2,)
        or not np.isfinite(pressures).all()
        or (pressures < 0).any()
        or not np.isfinite([temperature, mass, gas_radius, radius]).all()
        or min(temperature, mass, radius) <= 0
        or gas_radius < 0
    ):
        raise ValueError(
            "Invalid lengths, pressures, temperature, gas mass/radius or cluster radius"
        )
    skimmer = np.array(skimmer, dtype=float, copy=True)
    if skimmer.ndim != 2 or skimmer.shape[1] not in (3, 6) or len(skimmer) < 2:
        raise ValueError("skimmer must have at least two rows and 3 or 6 columns")
    if not np.isfinite(skimmer).all():
        raise ValueError("skimmer must be finite")
    ms = MassSpectrometer(
        skimmer,
        config["lengths"],
        Q_(np.zeros(5), "V"),
        config["T"],
        config["pressures"],
        mesh_skimmer=mesh_skimmer,
    )
    assert ms.mesh_skimmer is not None
    mesh = ms.mesh_skimmer.to("m").magnitude
    if not np.isfinite(mesh) or mesh <= 0:
        raise ValueError("mesh_skimmer must be finite and positive")
    if skimmer.shape[1] == 6 and (
        not np.isclose(skimmer[0, 0], 0, atol=1e-14)
        or not np.allclose(np.diff(skimmer[:, 0]), mesh, rtol=1e-8, atol=1e-14)
    ):
        raise ValueError("skimmer positions must start at zero and be uniformly spaced")
    # The simulator holds the final row over the last mesh cell.
    if lengths[-1] > len(ms.skimmer) * mesh * (1 + 1e-12):
        raise ValueError("skimmer does not cover the configured skimmer length")
    if (ms.skimmer[:, 1] <= 0).any() or (ms.skimmer[:, 2] < 0).any():
        raise ValueError(
            "skimmer temperature must be positive and pressure nonnegative"
        )
    return CollisionProfile(ms, gas, radius)
