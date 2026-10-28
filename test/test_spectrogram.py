from types import SimpleNamespace

import numpy as np
import pytest
from apitofsim.plotting.spectrogram import (
    get_intensities_multipathway,
    get_intensities_singlepathway,
)
from apitofsim.workflow.base import SimulationMode
from apitofsim.workflow.db import ExperimentDatabase, connection_scope


@pytest.mark.parametrize(
    "mode,counts,escaped,expected,shared_product",
    [
        ("SINGLE_CLUSTER", [800], 200, {"parent": 0.2, "product": 0.8}, False),
        (
            "CLUSTER_TREE",
            [300, 500],
            200,
            {"parent": 0.2, "product": 0.3, "terminal": 0.5},
            False,
        ),
        ("SINGLE_CLUSTER", [0], 1000, {"parent": 1.0, "product": 0.0}, False),
        ("CLUSTER_TREE", [], 1000, {"parent": 1.0}, False),
        *[
            (mode, counts, 400, {"parent": 0.4, "product": 0.6}, True)
            for mode in ("SINGLE_CLUSTER", "CLUSTER_TREE")
            for counts in ([300, 300], [200, 400])
        ],
    ],
)
def test_spectrum_product_intensities(mode, counts, escaped, expected, shared_product):
    with connection_scope(ExperimentDatabase, ":memory:") as db:
        db.create_tables()
        for name, charge in (
            ("parent", -1),
            ("product", -1),
            ("terminal", -1),
            ("neutral", 0),
        ):
            db.insert_cluster(name, 100, charge, 0, [1, 1, 1], [])
        db.insert_pathway(1, 2, 4)
        if shared_product:
            # In tree mode, two different final edges lead to the same ion.
            db.insert_pathway(3 if mode == "CLUSTER_TREE" else 1, 2, 4)
        else:
            db.insert_pathway(2, 3, 4)
        config_id = db.insert_config("spectrum", "{}")
        run_id = db.insert_run(config_id, simulation_mode=SimulationMode[mode])
        db.record_result(
            run_id,
            SimpleNamespace(
                nwarnings=0,
                n_escaped_total=escaped,
                n_fragmented_total=np.array(counts),
                ncoll_total=0,
                counter_collision_rejections=0,
            ),
            cluster_id=1,
            pathway_ids=list(range(1, len(counts) + 1)),
        )

        spectrum = get_intensities_multipathway(db, run_id, cluster_id=1)
        assert spectrum.product_name.is_unique
        charged = spectrum[spectrum.product_name != "neutral"]
        assert dict(zip(charged.product_name, charged.intensity)) == pytest.approx(
            expected
        )
        assert charged.intensity.sum() == pytest.approx(1.0)
        assert (
            spectrum.loc[spectrum.product_name == "neutral", "intensity"] == 0
        ).all()


@pytest.mark.parametrize(
    "counts,expected",
    [
        ([0], {"parent": 1.0, "product": 0.0}),
        ([200], {"parent": 0.8, "product": 0.2}),
        ([1000], {"parent": 0.0, "product": 1.0}),
        (
            [200, 500],
            {
                "parent": 0.4,
                "product": 0.1 + 0.1 * 2 / 7,
                "other product": 0.4 + 0.1 * 5 / 7,
            },
        ),
    ],
)
def test_singlepathway_fragmentation_intensity(counts, expected):
    with connection_scope(ExperimentDatabase, ":memory:") as db:
        db.create_tables()
        for name, charge in (
            ("parent", -1),
            ("product", -1),
            ("other product", -1),
            ("neutral", 0),
        ):
            db.insert_cluster(name, 100, charge, 0, [1, 1, 1], [])
        db.insert_pathway(1, 2, 4)
        db.insert_pathway(1, 3, 4)
        config_id = db.insert_config("spectrum", "{}")
        run_id = db.insert_run(
            config_id, simulation_mode=SimulationMode.PATHWAY_AT_A_TIME
        )
        for pathway_id, fragmented in enumerate(counts, start=1):
            db.record_result(
                run_id,
                SimpleNamespace(
                    nwarnings=0,
                    n_escaped_total=1000 - fragmented,
                    n_fragmented_total=np.array([fragmented]),
                    ncoll_total=0,
                    counter_collision_rejections=0,
                ),
                pathway_id=pathway_id,
            )

        spectrum = get_intensities_singlepathway(db, run_id, cluster_id=1)
        assert dict(zip(spectrum.product_name, spectrum.intensity)) == pytest.approx(
            expected
        )
        assert spectrum.intensity.sum() == pytest.approx(1.0)
