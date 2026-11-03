"""Shared prepared simulation/database fixture for recording tests."""

import numpy as np
import pytest
from apitofsim import api
from apitofsim.workflow import ExperimentDatabase, ExperimentRunner, RealizationDatabase
from apitofsim.workflow.db import connection_scope


@pytest.fixture(params=[True, False], ids=["fk", "nofk"])
def foreign_keys(request):
    return request.param


@pytest.fixture(params=[ExperimentDatabase, RealizationDatabase])
def prepared_case(request, monkeypatch, foreign_keys):
    with connection_scope(request.param, ":memory:") as db:
        db.create_tables(foreign_keys=foreign_keys)
        # Interleave parents in insertion order, with two outgoing pathways each.
        for name, mass, charge, energy in (
            ("parent", 200, -1, -1),
            ("other", 200, -1, -1),
            ("ion", 100, -1, -0.5),
            ("neutral", 100, 0, -0.49),
            ("other-neutral", 100, 0, -0.48),
        ):
            db.insert_cluster(name, mass, charge, energy, [1, 2, 3], [10, 20, 30])
        for parent, neutral in ((1, 4), (2, 4), (1, 5), (2, 5)):
            db.insert_pathway(parent, 3, neutral)
        db.db.execute(
            "insert into histogram_params (bin_width, max) values (100, 100000), (50, 100000), (50, 200)"
        )
        for cluster in (1, 2):
            db.db.execute(
                "insert into cluster_dos (cluster_id, histogram_params_id, data) values (?, 1, ?)",
                [cluster, np.ones(1000)],
            )
        for pathway in (1, 2, 3, 4):
            db.db.execute(
                "insert into k_rate (pathway_id, histogram_params_id, data) values (?, 2, ?)",
                [pathway, np.zeros(2000)],
            )
        config = dict(
            N=5,
            gas=api.Gas(api.Q_(2.46e-10, "m"), api.Q_(4.85e-26, "kg"), 1.4),
            lengths=api.Q_(np.full(5, 0.001), "m"),
            voltages=api.Q_(np.arange(5.0) * 10, "V"),
            T=api.Q_(300, "K"),
            pressures=api.Q_([100.0, 100.0], "Pa"),
            bin_width=api.Q_(999, "K"),
            energy_max=api.Q_(999, "K"),
            energy_max_rate=api.Q_(999, "K"),
            fragmentation_energy=api.Q_(1e9, "K"),
        )
        config_id = db.insert_config("stream", config)
        runner = ExperimentRunner(db)
        runner.start_run(config_id)
        skimmer = np.array(
            [[0.0, 0.0, 300.0, 100.0, 0.0, 0.0], [0.0005, 0.0, 300.0, 100.0, 0.0, 0.0]]
        )
        monkeypatch.setattr(runner.preparer, "_run_skimmer", lambda config: skimmer)
        yield db, runner, config, config_id
