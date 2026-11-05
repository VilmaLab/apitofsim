"""Preparation contracts shared by the native event and workflow APIs."""

import json

import numpy as np
import pytest
from apitofsim import api
from apitofsim.workflow import ExperimentRunner, SimulationMode


def prepare(case, **kwargs):
    _, runner, config, _ = case
    return runner.prepare_from_config(config, cached_densityandrate=(1, 2), **kwargs)


def assert_counters(actual, expected):
    for a, b in zip(actual, expected, strict=True):
        np.testing.assert_array_equal(a, b)


def test_preparation_and_provenance(prepared_case, capsys, monkeypatch):
    import progress_table

    monkeypatch.setattr(
        progress_table,
        "ProgressTable",
        lambda *a, **k: pytest.fail("terminal table constructed"),
    )
    db, runner, config, _ = prepared_case
    prepared = prepare(prepared_case, pathways=[4, 1, 3])
    assert [(g.cluster_id, g.pathway_ids) for g in prepared.groups] == [
        (1, (1, 3)),
        (2, (4,)),
    ]
    assert prepared.groups[0].product_ids == ((3, 4), (3, 5))
    assert (prepared.dos_histogram_id, prepared.rate_histogram_id) == (1, 2)
    for group in prepared.groups:
        substance = group.substance
        assert substance.cluster_charge_sign == -1
        np.testing.assert_allclose(
            api.Q_(substance.density_cluster.x, "J").to("K", "boltzmann").magnitude,
            np.arange(50, 100000, 100),
        )
        for pathway in substance.pathways:
            np.testing.assert_allclose(
                api.Q_(pathway.rate_const.x, "J").to("K", "boltzmann").magnitude,
                np.arange(25, 100000, 50),
            )
            assert pathway.bonding_energy < 1e9 * api.consts.boltzmann
    metadata = json.loads(
        db.db.execute(
            "select run_config from experiment_run where id = ?", [prepared.run_id]
        ).fetchone()[0]
    )
    assert (metadata["dos_histogram_id"], metadata["rate_histogram_id"]) == (1, 2)
    assert db.db.execute("select count(*) from experiment_run").fetchone() == (1,)
    with runner._iter_mass_spec(prepared, prepared.groups[0], N=0) as stream:
        terminal = list(stream)[-1]
    curves = db.pathway_rate_curves(prepared.run_id, 1, terminal.result_id)
    assert [c.pathway_id for c in curves] == [1, 3]
    assert all(c.participating for c in curves)
    assert all(len(c.histogram.x) == 2000 for c in curves)
    assert not capsys.readouterr().out


def test_selection_validation(prepared_case):
    db, runner, config, config_id = prepared_case
    unstarted = ExperimentRunner(db)
    with pytest.raises(RuntimeError, match="start_run"):
        unstarted.prepare_from_config(config)
    for options in (
        {"pathways": []},
        {"pathways": [999]},
        {"pathways": [1, 999]},
        {"parent": 999},
        {"parent": 1, "pathways": [1]},
    ):
        with pytest.raises(ValueError):
            prepare(prepared_case, **options)
    by_name = prepare(prepared_case, pathways=[("parent", "neutral", "ion")])
    assert by_name.groups[0].pathway_ids == (1,)
    with pytest.raises(ValueError, match="not found"):
        prepare(
            prepared_case,
            pathways=[("parent", "ion", "neutral"), ("missing", "ion", "neutral")],
        )
    assert len(prepare(prepared_case, parent="parent").groups) == 1
    runner.start_run(config_id, simulation_mode=SimulationMode.PATHWAY_AT_A_TIME)
    with pytest.raises(ValueError, match="SINGLE_CLUSTER"):
        prepare(prepared_case)


@pytest.mark.parametrize("cache", [(999, 2), (1, 999), (1, 3)])
def test_missing_caches(prepared_case, cache):
    _, runner, config, _ = prepared_case
    with pytest.raises(ValueError, match="miss"):
        runner.prepare_from_config(config, cached_densityandrate=cache)


def test_invalid_mesh(prepared_case):
    db, _, _, _ = prepared_case
    db.db.execute("update k_rate set data = [1, 2] where pathway_id = 1")
    with pytest.raises(ValueError, match="histogram configuration"):
        prepare(prepared_case, pathways=[1])
