import os
import numpy
import json
from typing import Any, Callable
from dataclasses import dataclass
from enum import Enum
from pandas import DataFrame
from pint import get_application_registry, Quantity

from _apitofsim import (
    skimmer as _skimmer,
    ClusterData as _ClusterData,
    Gas as _Gas,
    densityandrate as _densityandrate,
    Histogram as _Histogram,
    Quadrupole as _Quadrupole,
    pinhole as _pinhole,
)


ureg = get_application_registry()
ureg.define(
    "halfturn = 2 * π * radian = _ = halfrevolution = halfcycle = halfcircle = multiple_of_PI"
)
Q_ = ureg.Quantity


@dataclass
class ClusterData:
    mass: Quantity
    electronic_energy: Quantity
    rotations: Any
    frequencies: Any

    def into_cpp(self):
        return _ClusterData(
            self.mass.to("amu").magnitude,
            self.electronic_energy.to("hartree").magnitude,
            self.rotations,
            numpy.asfortranarray(self.frequencies, dtype=numpy.float64),
        )


@dataclass
class Gas:
    radius: Quantity
    mass: Quantity
    adiabatic_index: float

    def into_cpp(self):
        return _Gas(
            self.radius.to("m").magnitude,
            self.mass.to("kg").magnitude,
            self.adiabatic_index,
        )


@dataclass
class Histogram:
    x: Quantity
    y: numpy.ndarray

    @classmethod
    def from_cpp(cls, histogram: _Histogram):
        return cls(Q_(histogram.x, "kelvin"), histogram.y)

    def into_cpp(self):
        return _Histogram(self.x.to("kelvin").magnitude, self.y)


@dataclass
class Quadrupole:
    dc_field: Quantity
    ac_field: Quantity
    radiofrequency: Quantity
    r_quadrupole: Quantity

    def into_cpp(self):
        return _Quadrupole(
            self.dc_field.to("volts").magnitude,
            self.ac_field.to("volts").magnitude,
            self.radiofrequency.to("hertz").magnitude,
            self.r_quadrupole.to("m").magnitude,
        )


SKIMMER_COLUMNS = [
    "r",
    "vel",
    "T",
    "P",
    "rho",
    "speed_of_sound",
]


MaybeQuantity = Quantity | float
MaybeQuantityArray = Quantity | numpy.ndarray


class QuantityProcessor:
    def __init__(self, locals, quantities_strict=True):
        self.locals = locals
        self.quantities_strict = quantities_strict

    def __call__(self, name, unit):
        arg = self.locals[name]
        if isinstance(arg, Quantity):
            return arg.to(unit).magnitude
        elif not self.quantities_strict:
            return arg
        else:
            raise ValueError(
                f"Argument {name} (Value: {arg}) must be a pint.Quantity when `quantities_strict` is True"
            )


def skimmer(
    T0: MaybeQuantity,
    P0: MaybeQuantity,
    rmax: MaybeQuantity,
    dc: MaybeQuantity,
    alpha_factor: MaybeQuantity,
    gas: Gas | _Gas,
    N: int,
    M: int,
    resolution: int,
    tolerance: float,
    *,
    output_pandas=False,
    quantities_strict=True,
):
    process_arg = QuantityProcessor(locals(), quantities_strict)
    T0 = process_arg("T0", "kelvin")
    P0 = process_arg("P0", "pascal")
    rmax = process_arg("rmax", "meters")
    dc = process_arg("dc", "meters")
    alpha_factor = process_arg("alpha_factor", "halfturn")
    if isinstance(gas, Gas):
        gas = gas.into_cpp()
    out = _skimmer(T0, P0, rmax, dc, alpha_factor, gas, N, M, resolution, tolerance)
    if output_pandas:
        return DataFrame(out, columns=SKIMMER_COLUMNS)
    else:
        return out


def densityandrate(
    cluster_0: ClusterData,
    cluster_1: ClusterData,
    cluster_2: ClusterData,
    energy_max: MaybeQuantity,
    energy_max_rate: MaybeQuantity,
    bin_width: MaybeQuantity,
    fragmentation_energy: MaybeQuantity | None = None,
    *,
    quantities_strict=True,
):
    process_arg = QuantityProcessor(locals(), quantities_strict)
    energy_max = process_arg("energy_max", "kelvin")
    energy_max_rate = process_arg("energy_max_rate", "kelvin")
    bin_width = process_arg("bin_width", "kelvin")
    if fragmentation_energy is None:
        fragmentation_energy = 0
    else:
        fragmentation_energy = process_arg("fragmentation_energy", "kelvin")
    density_cluster, rate_const = _densityandrate(
        cluster_0.into_cpp(),
        cluster_1.into_cpp(),
        cluster_2.into_cpp(),
        energy_max,
        energy_max_rate,
        bin_width,
        fragmentation_energy,
    )
    return Histogram.from_cpp(density_cluster), Histogram.from_cpp(rate_const)


def pinhole(
    cluster_0: ClusterData,
    cluster_1: ClusterData,
    cluster_2: ClusterData,
    gas: Gas,
    density_cluster: Histogram,
    rate_const: Histogram,
    skimmer: numpy.ndarray,
    lengths: MaybeQuantityArray,
    voltages: MaybeQuantityArray,
    T: MaybeQuantity,
    pressure_first: MaybeQuantity,
    pressure_second: MaybeQuantity,
    N: int,
    *,
    mesh_skimmer: float | None = None,
    quadrupole: Quadrupole | None = None,
    cluster_charge_sign: int = 1,
    fragmentation_energy: MaybeQuantity | None = None,
    seed: int = 42,
    log_callback: Callable[[int, str], None] | None = None,
    result_callback: Callable[[numpy.ndarray], None] | None = None,
    quantities_strict=True,
):
    process_arg = QuantityProcessor(locals(), quantities_strict)
    lengths = process_arg("lengths", "meters")
    voltages = process_arg("voltages", "volts")
    T = process_arg("T", "kelvin")
    pressure_first = process_arg("pressure_first", "pascals")
    pressure_second = process_arg("pressure_second", "pascals")
    if skimmer.shape[1] == 3:
        if mesh_skimmer is None:
            raise ValueError(
                "mesh_skimmer must be supplied when 3 column array is given for skimmer"
            )
    elif skimmer.shape[1] == 6:
        if mesh_skimmer is not None:
            raise ValueError(
                "mesh_skimmer should not be supplied when 6 column array is given for skimmer"
            )
        mesh_skimmer = float(skimmer[1, 0] - skimmer[0, 0])
        skimmer = skimmer[:, 1:4]
    else:
        raise ValueError("skimmer must have 3 or 6 columns")
    return _pinhole(
        cluster_0.into_cpp(),
        cluster_1.into_cpp(),
        cluster_2.into_cpp(),
        gas.into_cpp(),
        density_cluster.into_cpp(),
        rate_const.into_cpp(),
        skimmer,
        mesh_skimmer,
        lengths,
        voltages,
        T,
        pressure_first,
        pressure_second,
        N,
        fragmentation_energy=fragmentation_energy,
        quadrupole=quadrupole and quadrupole.into_cpp(),
        cluster_charge_sign=cluster_charge_sign,
        seed=seed,
        log_callback=log_callback,
        result_callback=result_callback,
    )
