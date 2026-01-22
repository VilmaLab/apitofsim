import numpy
from typing import Callable, List, cast
from dataclasses import dataclass, KW_ONLY, MISSING
from pandas import DataFrame
from pint import get_application_registry, Quantity
from pint._typing import Magnitude
from abc import ABC, abstractmethod
from collections import namedtuple

from .apitofsimraw import (
    skimmer as _skimmer,
    ClusterData as _ClusterData,
    Gas as _Gas,
    densityandrate as _densityandrate,
    Histogram as _Histogram,
    Quadrupole as _Quadrupole,
    MassSpectrometer as _MassSpectrometer,
    validate_max_energies as _validate_max_energies,
    mass_spec as _mass_spec,
    KTotalInput,
    precompute_mesh as _precompute_mesh,
    compute_density_of_states_batch as _compute_density_of_states_batch,
    compute_k_total_batch as _compute_k_total_batch,
    MassSpecInputFragmentationPathway as _MassSpecInputFragmentationPathway,
    MassSpecSubstanceInput as _MassSpecSubstanceInput,
    FragmentationPathway,
    Counter as Counter,
    # Exceptions
    ApiTofError,
    ApiTofArgumentError,
    ApiTofOverflowError,
    ApiTofDosOverflow,
    ApiTofRateConstantOverflow,
    ApiTofMaxCollisions,
    ApiTofUnexpectedNumericalError,
    # Enums
    MeshMode,
    SampleMode,
)


__all__ = [
    "ClusterLike",
    "ClusterData",
    "ProductsCluster",
    "Gas",
    "Quadrupole",
    "Histogram",
    "densityandrate",
    "mass_spec",
    "skimmer",
    "compute_density_of_states_batch",
    "compute_k_total_batch",
    "KTotalInput",
    "MassSpecInputFragmentationPathway",
    "MassSpecSubstanceInput",
    "FragmentationPathway",
    # Exceptions
    "ApiTofError",
    "ApiTofArgumentError",
    "ApiTofOverflowError",
    "ApiTofDosOverflow",
    "ApiTofRateConstantOverflow",
    "ApiTofMaxCollisions",
    "ApiTofUnexpectedNumericalError",
    # Enums
    "MeshMode",
    "SampleMode",
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
            int(self.mass.to("amu").magnitude + 0.5),
            self.electronic_energy.to("hartree").magnitude,
            self.rotations,
            self.get_frequencies(),
        )

    def get_frequencies(self) -> numpy.ndarray:
        return numpy.asfortranarray(self.frequencies, dtype=numpy.float64)

    def is_atom_like_product(self) -> bool:
        return self.frequencies is None


@dataclass
class ProductsCluster(ClusterLike):
    cluster1: ClusterData
    cluster2: ClusterData

    def get_frequencies(self) -> numpy.ndarray:
        if self.cluster2.is_atom_like_product():
            frequencies = self.cluster1.get_frequencies()
        else:
            frequencies = numpy.concatenate(
                (self.cluster1.get_frequencies(), self.cluster2.get_frequencies())
            )
        return numpy.asfortranarray(frequencies, dtype=numpy.float64)


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


@dataclass
class MassSpectrometer:
    skimmer: numpy.ndarray
    lengths: Quantity[numpy.ndarray]
    voltages: Quantity[numpy.ndarray]
    T: Quantity[float]
    pressures: Quantity[numpy.ndarray]
    _: KW_ONLY
    # Only None during init, but can't specify this annoyingly
    mesh_skimmer: Quantity[float] | None = None
    quadrupole: Quadrupole | None = None
    radius_pinhole: Quantity[float] | None = Q_(1, "mm")

    def __post_init__(self):
        if self.skimmer.shape[1] == 3:
            if self.mesh_skimmer is None:
                raise ValueError(
                    "mesh_skimmer must be supplied when 3 column array is given for skimmer"
                )
        elif self.skimmer.shape[1] == 6:
            if self.mesh_skimmer is not None:
                raise ValueError(
                    "mesh_skimmer should not be supplied when 6 column array is given for skimmer"
                )
            self.mesh_skimmer = Q_(float(self.skimmer[1, 0] - self.skimmer[0, 0]), "m")
            self.skimmer = self.skimmer[:, 1:4]
        else:
            raise ValueError("skimmer must have 3 or 6 columns")

    def into_cpp(self):
        assert self.mesh_skimmer is not None
        return _MassSpectrometer(
            numpy.asfortranarray(self.skimmer),
            self.mesh_skimmer.to("m").magnitude,
            self.lengths.to("m").magnitude,
            self.voltages.to("volts").magnitude,
            self.T.to("K").magnitude,
            self.pressures.to("pascals").magnitude,
            self.quadrupole and self.quadrupole.into_cpp(),
            self.radius_pinhole and self.radius_pinhole.to("m").magnitude,
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


def precompute_mesh(
    energy_max_rate: MaybeQuantity,
    bin_width: MaybeQuantity,
    mesh_mode: MeshMode = MeshMode.compute_mesh_diagonal_multithreaded,
    *,
    quantities_strict=True,
):
    process_arg = QuantityProcessor(quantities_strict)
    energy_max_rate = process_arg("energy_max", energy_max_rate, "kelvin")
    bin_width = process_arg("bin_width", bin_width, "kelvin")
    return _precompute_mesh(energy_max_rate, bin_width, mesh_mode)


def compute_k_total_batch(
    inputs: List[KTotalInput],
    energy_max_rate: MaybeQuantity,
    bin_width: MaybeQuantity,
    mesh: MeshMode | numpy.ndarray = MeshMode.compute_mesh_diagonal_multithreaded,
    *,
    quantities_strict=True,
):
    process_arg = QuantityProcessor(quantities_strict)
    energy_max_rate = process_arg("energy_max", energy_max_rate, "kelvin")
    bin_width = process_arg("bin_width", bin_width, "kelvin")
    return _compute_k_total_batch(inputs, energy_max_rate, bin_width, mesh)


class ArgGetter:
    def __init__(self, args, kwargs):
        self.args = args
        self.kwargs = kwargs

    def __call__(self, name: str, position: int, default=MISSING):
        if name in self.kwargs:
            return self.kwargs[name]
        elif position < len(self.args):
            return self.args[position]
        else:
            if default is not MISSING:
                return default
            raise ValueError(f"Argument {name} at position {position} not found")


def MassSpecInputFragmentationPathway(*args, **kwargs):
    process_arg = QuantityProcessor(kwargs.get("quantities_strict", True))

    def proc_bonding_energy(bonding_energy):
        if bonding_energy is not None:
            return process_arg("bonding_energy", bonding_energy, "kelvin")

    get = ArgGetter(args, kwargs)
    if len(args) >= 1 and isinstance(args[0], ClusterLike) or "cluster_0" in kwargs:
        return _MassSpecInputFragmentationPathway(
            cluster_0=get("cluster_0", 0).into_cpp(),
            cluster_1=get("cluster_1", 1).into_cpp(),
            cluster_2=get("cluster_2", 2).into_cpp(),
            rate_const=get("rate_const", 3).into_cpp(),
            bonding_energy=proc_bonding_energy(get("bonding_energy", 4, None)),
        )
    else:
        return _MassSpecInputFragmentationPathway(
            rate_const=get("rate_const", 0).into_cpp(),
            bonding_energy=proc_bonding_energy(get("bonding_energy", 1, None)),
        )


def MassSpecSubstanceInput(*args, **kwargs):
    get = ArgGetter(args, kwargs)
    if len(args) >= 1 and isinstance(args[0], ClusterLike) or "cluster_0" in kwargs:
        return _MassSpecSubstanceInput(
            cluster_0=get("cluster_0", 0).into_cpp(),
            cluster_1=get("cluster_1", 1).into_cpp(),
            cluster_2=get("cluster_2", 2).into_cpp(),
            gas=get("gas", 3).into_cpp(),
            density_cluster=get("density_cluster", 4).into_cpp(),
            rate_const=get("rate_const", 5).into_cpp(),
            fragmentation_energy=get("fragmentation_energy", 6, None),
            cluster_charge_sign=get("cluster_charge_sign", 7, 1),
        )
    else:
        return _MassSpecSubstanceInput(
            cluster_charge_sign=get("cluster_charge_sign", 0),
            m_ion=get("m_ion", 1),
            R_cluster=get("R_cluster", 2),
            density_cluster=get("R_cluster", 3).into_cpp(),
            pathway=get("pathway", 4),
            gas=get("gas", 5).into_cpp(),
        )


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


Counters = namedtuple("Counters", [t.name for t in Counter])
Timings = namedtuple("Timings", ["loop", "total"])


def mass_spec(
    mass_spec: MassSpectrometer,
    subs: _MassSpecSubstanceInput,
    N: int,
    *,
    sample_mode: SampleMode = SampleMode.rejection,
    strict=True,
    loglevel: int = 0,
    seed: int = 42,
    log_callback: Callable[[str, str], None] | None = None,
    result_callback: Callable[[numpy.ndarray], None] | None = None,
    quantities_strict=True,
    output_named_tuple=False,
    output_timings=False,
):
    """
    This function runs the main simulation of the APi-ToF mass spectrometer.
    """
    counters, loop_time, total_time = _mass_spec(
        mass_spec.into_cpp(),
        subs,
        N,
        seed=seed,
        log_callback=log_callback,
        result_callback=result_callback,
        sample_mode=sample_mode,
        strict=strict,
        loglevel=loglevel,
    )
    if output_named_tuple:
        counters = Counters(*counters)
    if output_timings:
        return counters, Timings(loop_time, total_time)
    else:
        return counters


def validate_max_energies(
    fragmentation_energy,
    energy_max,
    energy_max_rate,
    bin_width,
    quantities_strict=True,
):
    process_arg = QuantityProcessor(quantities_strict)
    fragmentation_energy = process_arg(
        "fragmentation_energy", fragmentation_energy, "kelvin"
    )
    energy_max = process_arg("energy_max", energy_max, "kelvin")
    energy_max_rate = process_arg("energy_max_rate", energy_max_rate, "kelvin")
    bin_width = process_arg("bin_width", bin_width, "kelvin")
    _validate_max_energies(fragmentation_energy, energy_max, energy_max_rate, bin_width)
