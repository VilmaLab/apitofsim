from types import SimpleNamespace

import duckdb
import pytest
from apitofsim.workflow.base import SimulationMode
from apitofsim.workflow.db import ExperimentDatabase, connection_scope


@pytest.fixture
def report_db():
    with connection_scope(ExperimentDatabase, ":memory:") as db:
        db.create_tables()
        db.db.execute(
            "insert into cluster (id, common_name) values "
            "(1, 'root'), (2, 'child'), (3, 'terminal'), (4, 'neutral')"
        )
        db.db.execute(
            "insert into pathway values (1, 1, 2, 4), (2, 2, 3, 4), (3, 1, 3, 4)"
        )
        db.insert_config("reports", "{}")
        yield db


def record_counts(db, run_id, escaped, counts, **kwargs):
    return db.record_result(
        run_id,
        SimpleNamespace(
            nwarnings=0,
            n_escaped_total=escaped,
            n_fragmented_total=counts,
            ncoll_total=0,
            counter_collision_rejections=0,
        ),
        **kwargs,
    )


@pytest.mark.parametrize("mode", list(SimulationMode))
def test_run_mode_enum(report_db, mode):
    run_id = report_db.insert_run(1, simulation_mode=mode)
    assert report_db.db.execute(
        "select simulation_mode from experiment_run where id = ?", [run_id]
    ).fetchone() == (mode.name,)
    # Empty runs still carry their mode, without inferring it from results.
    summary = report_db.report_df("experiment_summary").iloc[0]
    assert summary.simulation_mode == mode.name
    assert bool(summary.is_single_pathway) == (mode == SimulationMode.PATHWAY_AT_A_TIME)
    with pytest.raises(duckdb.ConversionException):
        report_db.db.execute(
            "update experiment_run set simulation_mode = 'unknown' where id = ?",
            [run_id],
        )


@pytest.mark.parametrize(
    "mode", [SimulationMode.SINGLE_CLUSTER, SimulationMode.CLUSTER_TREE]
)
def test_root_survival_and_pathway_parent(report_db, mode):
    db = report_db
    run_id = db.insert_run(1, simulation_mode=mode)
    second_path = 2 if mode == SimulationMode.CLUSTER_TREE else 3
    record_counts(
        db, run_id, 200, [300, 500], cluster_id=1, pathway_ids=[1, second_path]
    )
    record_counts(db, run_id, 700, [300], cluster_id=2, pathway_ids=[2])

    clusters = db.report_df("experiment_cluster_report").set_index("cluster_id")
    assert clusters.survival_rate.to_dict() == pytest.approx({1: 0.2, 2: 0.7})
    assert clusters.realizations.to_dict() == {1: 1000, 2: 1000}
    assert set(clusters.simulation_mode) == {mode.name}

    pathways = db.report_df("experiment_pathway_report")
    root_rows = pathways[pathways.cluster_id == 1].sort_values("pathway_id")
    assert root_rows.pathway_id.tolist() == [1, second_path]
    assert root_rows.pathway_cluster_id.tolist() == [1, 2 if second_path == 2 else 1]
    assert root_rows.pathway_cluster_common_name.tolist() == [
        "root",
        "child" if second_path == 2 else "root",
    ]
    assert set(root_rows.simulation_mode) == {mode.name}


def test_pathway_at_a_time_survival(report_db):
    db = report_db
    run_id = db.insert_run(1, simulation_mode=SimulationMode.PATHWAY_AT_A_TIME)
    record_counts(db, run_id, 800, [200], pathway_id=1)
    record_counts(db, run_id, 500, [500], pathway_id=3)
    db.record_failure(run_id, "ValueError", "failed pathway", pathway_id=1)
    cluster = db.report_df("experiment_cluster_report").iloc[0]
    assert cluster.survival_rate == pytest.approx(0.8 * 0.5)
    assert cluster.realizations == 2000
    assert cluster.is_single_pathway


def test_tree_empty_pathways_and_failures(report_db):
    db = report_db
    run_id = db.insert_run(1, simulation_mode=SimulationMode.CLUSTER_TREE)
    record_counts(db, run_id, 1000, [], cluster_id=1, pathway_ids=[])
    record_counts(db, run_id, 1000, [0], cluster_id=2, pathway_ids=[2])
    db.record_failure(run_id, "ValueError", "failed root", cluster_id=3)
    clusters = db.report_df("experiment_cluster_report").set_index("cluster_id")
    assert clusters.loc[[1, 2], "survival_rate"].tolist() == [1.0, 1.0]
    assert clusters.loc[[1, 2], "realizations"].tolist() == [1000, 1000]
    assert clusters.loc[[3], "survival_rate"].isna().all()
    pathways = db.report_df("experiment_pathway_report").set_index("cluster_id")
    assert sorted(pathways.index) == [1, 2, 3]
    assert pathways.loc[3, "outcome_type"] == "failure"


def test_repeated_root_results(report_db):
    db = report_db
    run_id = db.insert_run(1, simulation_mode=SimulationMode.CLUSTER_TREE)
    record_counts(db, run_id, 200, [800], cluster_id=1, pathway_ids=[1])
    record_counts(db, run_id, 50, [50], cluster_id=1, pathway_ids=[1])
    cluster = db.report_df("experiment_cluster_report").iloc[0]
    assert cluster.realizations == 1100
    assert cluster.survival_rate == pytest.approx(250 / 1100)
