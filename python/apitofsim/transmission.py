from collections.abc import Callable
from dataclasses import dataclass

import numpy
from numpy.polynomial.polynomial import polyval
from pandas import Series
from pint import Quantity

from .api import (
    ureg,
)

Q_ = ureg.Quantity

OLD_TRANSMISSION_COEFFS = numpy.array(
    [
        0.013902063099905,
        3.675651157495839e-05,
        5.933105419193590e-08,
        1.382508354374988e-12,
    ]
)

VOLTAGES_NEGATIVE = Q_(
    numpy.array(
        [
            -19,
            -9,
            -7,
            -6,
            11,
        ]
    ),
    "volts",
)

VOLTAGES_POSITIVE = Q_(
    numpy.array(
        [
            25,
            12,
            -4,
            -8,
            -17,
        ]
    ),
    "volts",
)


def old_transmission(mass):
    return polyval(mass, OLD_TRANSMISSION_COEFFS)


def new_transmission_neg(mass):
    return 2.646625470731280 * numpy.exp(-0.004315042768417 * mass)


def new_transmsision_pos(mass):
    return 0.433321942556931 * numpy.exp(
        -(((mass - 2.419326344811581e02) / 1.589675479743877e02) ** 2)
    )


@dataclass
class TransmissionModel:
    name: str
    func: Callable[[Series], Series]
    cluster_charge: int
    voltages: Quantity[numpy.ndarray]


TRANSMISSION_MODELS: dict[str, TransmissionModel] = {
    "old": TransmissionModel("Old model", old_transmission, -1, VOLTAGES_NEGATIVE),
    "new_neg": TransmissionModel(
        "New model (negative ions)", new_transmission_neg, -1, VOLTAGES_NEGATIVE
    ),
    "new_pos": TransmissionModel(
        "New model (positive ions)", new_transmsision_pos, 1, VOLTAGES_POSITIVE
    ),
}


def transform_intensity(df, model_transmission):
    if model_transmission is not None:
        model_transmission_func = TRANSMISSION_MODELS[model_transmission].func
        df["intensity"] = model_transmission_func(df["atomic_mass"]) * df["intensity"]
