from abc import ABC, abstractmethod
from collections import namedtuple
from dataclasses import KW_ONLY, MISSING, dataclass
from typing import Callable, List, Optional, Tuple, cast

import numpy
from pandas import DataFrame
from pint import Quantity, get_application_registry
from pint._typing import Magnitude

from .apitofsimraw import (
    DEFAULT_LOGLEVEL,
    ApiTofArgumentError,
    ApiTofDosOverflow,
    ApiTofError,
    ApiTofMaxCollisions,
    ApiTofOverflowError,
    ApiTofRateConstantOverflow,
    ApiTofUnexpectedNumericalError,
    CollisionEvent,
    EscapeEvent,
    FragmentationEvent,
    FragmentationPathway,
    KTotalInput,
    MeshMode,
    ParticleState,
    SampleMode,
    defaults,
)
from .apitofsimraw import (
    ClusterData as _ClusterData,
)
from .apitofsimraw import (
    Counter as Counter,
)
from .apitofsimraw import (
    Gas as _Gas,
)
from .apitofsimraw import (
    Histogram as _Histogram,
)
from .apitofsimraw import (
    MassSpecInputFragmentationPathway as _MassSpecInputFragmentationPathway,
)
from .apitofsimraw import (
    MassSpecIterator as _MassSpecIterator,
)
from .apitofsimraw import (
    MassSpecSubstanceInput as _MassSpecSubstanceInput,
)
from .apitofsimraw import (
    MassSpectrometer as _MassSpectrometer,
)
from .apitofsimraw import (
    Quadrupole as _Quadrupole,
)
from .apitofsimraw import (
    compute_density_of_states_batch as _compute_density_of_states_batch,
)
from .apitofsimraw import (
    compute_k_total_batch as _compute_k_total_batch,
)
from .apitofsimraw import (
    densityandrate as _densityandrate,
)
from .apitofsimraw import (
    mass_spec as _mass_spec,
)
from .apitofsimraw import (
    precompute_mesh as _precompute_mesh,
)
from .apitofsimraw import (
    skimmer as _skimmer,
)
from .apitofsimraw import (
    validate_max_energies as _validate_max_energies,
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
    # EventMessage
    "ParticleState",
    "CollisionEvent",
    "FragmentationEvent",
    "EscapeEvent",
    # Enums
    "MeshMode",
    "SampleMode",
    # Defaults
    "defaults",
]


ureg = get_application_registry()
ureg.define(
    "halfturn = π * radian = _ = halfrevolution = halfcycle = halfcircle = multiple_of_PI"
)
Q_ = ureg.Quantity


class ClusterLike(ABC):
    """
    A base class for cluster-like things.
    """

    @abstractmethod
    def get_frequencies(self) -> Optional[numpy.ndarray]:
        """
        This method returns an array of vibrational temperatures in Kelvin.
        In the case of an atom-like product this will return None.
        """
        ...


@dataclass
class ClusterData(ClusterLike):
    """
    The basic physical data for a cluster.
    """

    mass: Quantity[float]
    """The cluster's mass"""
    electronic_energy: Quantity[float]
    """The cluster's electronic energy"""
    rotations: numpy.ndarray
    """From quantum chemistry calcuations, the rotational temperatures in Kelvin for the cluster. This is a 3 element array."""
    frequencies: numpy.ndarray
    """From quantum chemistry calcuations, the vibrational temperatures in Kelvin for the cluster."""

    def into_cpp(self) -> _ClusterData:
        frequencies = self.get_frequencies()
        if frequencies is None:
            frequencies = numpy.empty(0, dtype=numpy.float64, order="F")
        return _ClusterData(
            int(self.mass.to("amu").magnitude + 0.5),
            self.electronic_energy.to("hartree").magnitude,
            self.rotations,
            frequencies,
        )

    def get_frequencies(self) -> Optional[numpy.ndarray]:
        if self.is_atom_like_product():
            return None
        else:
            return numpy.asfortranarray(self.frequencies, dtype=numpy.float64)

    def is_atom_like_product(self) -> bool:
        return self.frequencies is None


@dataclass
class ProductsCluster(ClusterLike):
    """
    A combination of two clusters representing the products of a fragmentation pathway.
    This is used to compute the density of states and derived quantities for the pathway products at the point of collision.
    """

    cluster1: ClusterData
    """
    One cluster
    """
    cluster2: ClusterData
    """
    The other cluster
    """

    def get_frequencies(self) -> Optional[numpy.ndarray]:
        frequencies1 = self.cluster1.get_frequencies()
        frequencies2 = self.cluster2.get_frequencies()
        if frequencies1 is None and frequencies2 is None:
            raise ValueError(
                "Cannot have a ProductCluster with both clusters being atom-like products"
            )
        if frequencies2 is None:
            frequencies = frequencies1
        elif frequencies1 is None:
            frequencies = frequencies2
        else:
            frequencies = numpy.concatenate((frequencies1, frequencies2))
        return numpy.asfortranarray(frequencies, dtype=numpy.float64)


@dataclass
class Gas:
    """
    The physical quantities related to gas in the mass spectrometer
    """

    radius: Quantity[float]
    """
    The radius of the gas particle
    """
    mass: Quantity[float]
    """
    The mass of the gas particle
    """
    adiabatic_index: float
    """
    The adiabatic index of the gas particle
    """

    def into_cpp(self) -> _Gas:
        return _Gas(
            self.radius.to("m").magnitude,
            self.mass.to("kg").magnitude,
            self.adiabatic_index,
        )


@dataclass
class Histogram:
    """
    A container for histogrammed data used to store precomputed density of states and rate constants.

    You should not typically need to construc this yourself.
    """

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
    """
    Configuration values related to the quadrupole mass filter in the mass spectrometer, if present
    """

    dc_field: Quantity[float]
    """
    The DC voltage applied to the quadrupole rods
    """
    ac_field: Quantity[float]
    """
    The AC voltage applied to the quadrupole rods
    """
    radiofrequency: Quantity[float]
    """
    The radiofrequency of the AC voltage applied to the quadrupole rods
    """
    r_quadrupole: Quantity[float]
    """
    The distance from the center of the quadrupole to the rods
    """

    def into_cpp(self) -> _Quadrupole:
        return _Quadrupole(
            self.dc_field.to("volts").magnitude,
            self.ac_field.to("volts").magnitude,
            self.radiofrequency.to("hertz").magnitude,
            self.r_quadrupole.to("m").magnitude,
        )


@dataclass
class MassSpectrometer:
    """
    The configuration values needed to simulate the mass spectrometer as well as the precomputed, histogrammed skimmer values.
    """

    skimmer: numpy.ndarray
    """
    A 2D array of values along the skimmer, with either XXXXCHECK 3 columns (r, vel, T) or 6 columns (x, r, vel, T, P, rho)
    """
    lengths: Quantity[numpy.ndarray]
    """
    An array of the lengths of the different sections of the mass spectrometer
    """
    voltages: Quantity[numpy.ndarray]
    """
    The voltages applied at different points on the mass spectrometer
    """
    T: Quantity[float]
    """
    The temperature in the mass spectrometer.
    """
    pressures: Quantity[numpy.ndarray]
    """
    The pressures in the two chambers of the mass spectrometer
    """
    _: KW_ONLY
    # Only None during init, but can't specify this annoyingly
    mesh_skimmer: Quantity[float] | None = None
    """
    The histogram mesh size used for the precomputed, histogrammed skimmer quantities.
    If not given this will be computed from the skimmer array if it has 6 columns, otherwise it must be supplied.
    """
    quadrupole: Quadrupole | None = None
    """
    The quadrupole configuration, if a quadrupole is present in the mass spectrometer.
    """
    radius_pinhole: Quantity[float] | None = Q_(1, "mm")
    """
    The radius of the pinhole in the skimmer, if present.
    """

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
    frequencies = []
    for i, cluster in enumerate(clusters):
        frequencies_cluster = cluster.get_frequencies()
        if frequencies_cluster is None:
            raise ValueError(
                f"Cannot compute density of states for a atom-like product {cluster!r} at index {i}"
            )
        frequencies.append(frequencies_cluster)
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
    progress_callback: Callable[[int], None] | None = None,
    *,
    quantities_strict=True,
):
    process_arg = QuantityProcessor(quantities_strict)
    energy_max_rate = process_arg("energy_max", energy_max_rate, "kelvin")
    bin_width = process_arg("bin_width", bin_width, "kelvin")
    return _compute_k_total_batch(
        inputs, energy_max_rate, bin_width, mesh, progress_callback
    )


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
    """
    Construct a MassSpecInputFragmentationPathway
    """
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
    """
    Construct a MassSpecSubstanceInput
    """
    get = ArgGetter(args, kwargs)
    if len(args) >= 1 and isinstance(args[0], ClusterLike) or "cluster_0" in kwargs:
        if len(args) >= 2 and isinstance(args[1], ClusterLike) or "cluster_1" in kwargs:
            return _MassSpecSubstanceInput(
                cluster_0=get("cluster_0", 0).into_cpp(),
                cluster_1=get("cluster_1", 1).into_cpp(),
                cluster_2=get("cluster_2", 2).into_cpp(),
                gas=get("gas", 3).into_cpp(),
                density_cluster=get("density_cluster", 4).into_cpp(),
                rate_const=get("rate_const", 5).into_cpp(),
                fragmentation_energy=get("fragmentation_energy", 6, None),
                cluster_charge_sign=get(
                    "cluster_charge_sign", 7, defaults.cluster_charge_sign
                ),
            )
        else:
            return _MassSpecSubstanceInput(
                cluster_0=get("cluster_0", 0).into_cpp(),
                pathways=get("pathways", 1),
                gas=get("gas", 2).into_cpp(),
                density_cluster=get("density_cluster", 3).into_cpp(),
                cluster_charge_sign=get(
                    "cluster_charge_sign", 4, defaults.cluster_charge_sign
                ),
            )
    else:
        return _MassSpecSubstanceInput(
            cluster_charge_sign=get("cluster_charge_sign", 0),
            m_ion=get("m_ion", 1),
            R_cluster=get("R_cluster", 2),
            density_cluster=get("density_cluster", 3).into_cpp(),
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


def counters_named_tuple(counters):
    return Counters(*counters[: len(Counter) - 1], counters[len(Counter) - 1 :])


def mass_spec(
    mass_spec: MassSpectrometer,
    subs: _MassSpecSubstanceInput,
    N: int,
    *,
    sample_mode: SampleMode = SampleMode.rejection,
    strict=True,
    logconf: Tuple[int, bool] = (DEFAULT_LOGLEVEL, False),
    seed: int = 42,
    log_callback: Callable[[str, str], None] | None = None,
    result_callback: Callable[[numpy.ndarray], None] | None = None,
    event_callback: Callable[
        [ParticleState | CollisionEvent | FragmentationEvent | EscapeEvent], None
    ]
    | None = None,
    named_tuple_counters=False,
    output_timings=False,
):
    """
    This function runs the main simulation of the APi-ToF mass spectrometer.
    """

    def convert_counters(counters):
        if named_tuple_counters:
            return counters_named_tuple(counters)
        else:
            return counters

    def wrap_callback(callback):
        if callback is None:
            return None

        def inner(counters):
            return callback(convert_counters(counters))

        return inner

    counters, loop_time, total_time = _mass_spec(
        mass_spec.into_cpp(),
        subs,
        N,
        seed=seed,
        log_callback=log_callback,
        result_callback=wrap_callback(result_callback),
        event_callback=event_callback,
        sample_mode=sample_mode,
        strict=strict,
        logconf=logconf,
    )
    if named_tuple_counters:
        counters = convert_counters(counters)
    if output_timings:
        return counters, Timings(loop_time, total_time)
    else:
        return counters


@dataclass
class MassSpecIntermediateCounter:
    counters: Counters


@dataclass
class MassSpecLogItem:
    type: str
    name: str


@dataclass
class MassSpecFinalResult:
    counters: Counters
    timings: Timings


class MassSpecIterator(_MassSpecIterator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __iter__(self):
        return self

    def __next__(self):
        val = super().__next__()
        if isinstance(val, tuple):
            if len(val) == 2:
                return MassSpecLogItem(*val)
            elif isinstance(val[0], numpy.ndarray):
                return MassSpecFinalResult(
                    counters=counters_named_tuple(val[0]), timings=Timings(*val[1:])
                )
        elif isinstance(val, numpy.ndarray):
            return MassSpecIntermediateCounter(counters_named_tuple(val))
        else:
            return val

    def __enter__(self):  # pyright: ignore [reportMissingSuperCall]
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):  # pyright: ignore [reportMissingSuperCall]
        self.join_if_joinable()


def mass_spec_iter(
    mass_spec: MassSpectrometer,
    subs: _MassSpecSubstanceInput,
    N: int,
    *,
    sample_mode: SampleMode = SampleMode.rejection,
    strict=True,
    logconf: Tuple[int, bool] = (DEFAULT_LOGLEVEL, False),
    seed: int = 42,
):
    return MassSpecIterator(
        mass_spec.into_cpp(),
        subs,
        N,
        sample_mode=sample_mode,
        strict=strict,
        logconf=logconf,
        seed=seed,
    )


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
