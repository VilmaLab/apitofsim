import numpy
from typing import Callable, List, cast
from dataclasses import dataclass
from pandas import DataFrame
from pint import get_application_registry, Quantity
from pint._typing import Magnitude
from abc import ABC, abstractmethod

from .apitofsimraw import (
    skimmer as _skimmer,
    ClusterData as _ClusterData,
    Gas as _Gas,
    densityandrate as _densityandrate,
    Histogram as _Histogram,
    Quadrupole as _Quadrupole,
    pinhole as _pinhole,
    KTotalInput,
    compute_density_of_states_batch as _compute_density_of_states_batch,
    compute_k_total_batch as _compute_k_total_batch,
    FragmentationPathway,
)


__all__ = [
    "ClusterLike",
    "ClusterData",
    "ProductsCluster",
    "Gas",
    "Quadrupole",
    "Histogram",
    "densityandrate",
    "pinhole",
    "skimmer",
    "compute_density_of_states_batch",
    "compute_k_total_batch",
    "KTotalInput",
    "FragmentationPathway",
]


ureg = get_application_registry()
ureg.define(
    "halfturn = 2 * π * radian = _ = halfrevolution = halfcycle = halfcircle = multiple_of_PI"
)
Q_ = ureg.Quantity


class ClusterLike(ABC):
    @abstractmethod
    def get_frequencies(self) -> numpy.ndarray: ...


@dataclass
class ClusterData(ClusterLike):
    mass: Quantity[float]
    electronic_energy: Quantity[float]
    rotations: numpy.ndarray
    frequencies: numpy.ndarray

    def into_cpp(self) -> _ClusterData:
        return _ClusterData(
            self.mass.to("amu").magnitude,
            self.electronic_energy.to("hartree").magnitude,
            self.rotations,
            self.get_frequencies(),
        )

    def get_frequencies(self) -> numpy.ndarray:
        return numpy.asfortranarray(self.frequencies, dtype=numpy.float64)


@dataclass
class ProductsCluster(ClusterLike):
    cluster1: ClusterData
    cluster2: ClusterData

    def get_frequencies(self) -> numpy.ndarray:
        return numpy.asfortranarray(
            numpy.concatenate(
                (self.cluster1.get_frequencies(), self.cluster2.get_frequencies())
            ),
            dtype=numpy.float64,
        )


@dataclass
class Gas:
    radius: Quantity[float]
    mass: Quantity[float]
    adiabatic_index: float

    def into_cpp(self) -> _Gas:
        return _Gas(
            self.radius.to("m").magnitude,
            self.mass.to("kg").magnitude,
            self.adiabatic_index,
        )


@dataclass
class Histogram:
    x: Quantity[numpy.ndarray]
    y: numpy.ndarray

    @classmethod
    def from_mesh(cls, bin_width, x_max, y):
        bin_width_mag = bin_width.to("kelvin").magnitude
        m_max = int(x_max.to("kelvin").magnitude / bin_width_mag)
        return cls.from_cpp(_Histogram(bin_width_mag, m_max, y))

    @classmethod
    def from_cpp(cls, histogram: _Histogram):
        return cls(Q_(histogram.x, "kelvin"), histogram.y)

    def into_cpp(self) -> _Histogram:
        return _Histogram(self.x.to("kelvin").magnitude, self.y)


@dataclass
class Quadrupole:
    dc_field: Quantity[float]
    ac_field: Quantity[float]
    radiofrequency: Quantity[float]
    r_quadrupole: Quantity[float]

    def into_cpp(self) -> _Quadrupole:
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


type MaybeQuantity = Quantity[float] | float
type MaybeQuantityArray = Quantity[numpy.ndarray] | numpy.ndarray


class QuantityProcessor:
    def __init__(self, quantities_strict=True):
        self.quantities_strict = quantities_strict

    def __call__[T: Magnitude](self, name: str, arg: Quantity[T] | T, unit: str) -> T:
        if isinstance(arg, Quantity):
            return arg.to(unit).magnitude
        elif not self.quantities_strict:
            # This is obviously the T -> T case, but pyright won't accept it
            return cast(Magnitude, arg)  # pyright: ignore [reportReturnType]
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
    """
    This function precomputes various parameters including gas velocity, temperature and pressure at fixed points along the skimmer's' length.
    """
    process_arg = QuantityProcessor(quantities_strict)
    T0 = process_arg("T0", T0, "kelvin")
    P0 = process_arg("P0", P0, "pascal")
    rmax = process_arg("rmax", rmax, "meters")
    dc = process_arg("dc", dc, "meters")
    alpha_factor = process_arg("alpha_factor", alpha_factor, "halfturn")
    if isinstance(gas, Gas):
        gas = gas.into_cpp()
    out = _skimmer(T0, P0, rmax, dc, alpha_factor, gas, N, M, resolution, tolerance)
    if output_pandas:
        # Ignore this because Pandas' types are broken
        return DataFrame(out, columns=SKIMMER_COLUMNS)  # pyright: ignore [reportArgumentType]
    else:
        return out


def compute_density_of_states_batch(
    clusters: List[ClusterLike],
    energy_max: MaybeQuantity,
    bin_width: MaybeQuantity,
    use_old_impl=False,
    *,
    quantities_strict=True,
):
    process_arg = QuantityProcessor(quantities_strict)
    energy_max = process_arg("energy_max", energy_max, "kelvin")
    bin_width = process_arg("bin_width", bin_width, "kelvin")
    frequencies = [cluster.get_frequencies() for cluster in clusters]
    return _compute_density_of_states_batch(
        frequencies, energy_max, bin_width, use_old_impl=use_old_impl
    )


def compute_k_total_batch(
    inputs: List[KTotalInput],
    energy_max_rate: MaybeQuantity,
    bin_width: MaybeQuantity,
    mesh_mode: int,
    *,
    quantities_strict=True,
):
    process_arg = QuantityProcessor(quantities_strict)
    energy_max_rate = process_arg("energy_max", energy_max_rate, "kelvin")
    bin_width = process_arg("bin_width", bin_width, "kelvin")
    return _compute_k_total_batch(inputs, energy_max_rate, bin_width, mesh_mode)


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
    """
    This function precomputes the density of states and rate constants histograms for a given set of clusters.
    """
    process_arg = QuantityProcessor(quantities_strict)
    energy_max = process_arg("energy_max", energy_max, "kelvin")
    energy_max_rate = process_arg("energy_max_rate", energy_max_rate, "kelvin")
    bin_width = process_arg("bin_width", bin_width, "kelvin")
    if fragmentation_energy is None:
        fragmentation_energy = 0
    else:
        fragmentation_energy = process_arg(
            "fragmentation_energy", fragmentation_energy, "kelvin"
        )
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
    sample_mode: int = 0,
    loglevel: int = 0,
    mesh_skimmer: float | None = None,
    quadrupole: Quadrupole | None = None,
    cluster_charge_sign: int = -1,
    fragmentation_energy: MaybeQuantity | None = None,
    seed: int = 42,
    log_callback: Callable[[str, str], None] | None = None,
    result_callback: Callable[[numpy.ndarray], None] | None = None,
    quantities_strict=True,
):
    """
    This function runs the main simulation of the APi-ToF mass spectrometer.
    """
    process_arg = QuantityProcessor(quantities_strict)
    lengths = process_arg("lengths", lengths, "meters")
    voltages = process_arg("voltages", voltages, "volts")
    T = process_arg("T", T, "kelvin")
    pressure_first = process_arg("pressure_first", pressure_first, "pascals")
    pressure_second = process_arg("pressure_second", pressure_second, "pascals")
    if fragmentation_energy is not None:
        fragmentation_energy = process_arg(
            "fragmentation_energy", fragmentation_energy, "kelvin"
        )
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
        sample_mode=sample_mode,
        loglevel=loglevel,
    )
