"""Optional foreign keys keep reports and non-FK integrity constraints intact."""

import duckdb
import pytest
from apitofsim.cli import check_integrity, prepare
from apitofsim.workflow.db import (
    ClusterDatabase,
    ExperimentDatabase,
    SuperClusterDatabase,
    connection_scope,
)
from click.testing import CliRunner


@pytest.mark.parametrize(
    "db_type",
    [ClusterDatabase, SuperClusterDatabase, ExperimentDatabase],
)
@pytest.mark.parametrize("foreign_keys", [None, True, False])
def test_schema_constraints(db_type, foreign_keys):
    with connection_scope(db_type, ":memory:") as db:
        db.create_tables(
            **({} if foreign_keys is None else {"foreign_keys": foreign_keys})
        )
        constraints = db.db.execute(
            "select constraint_type, count(*) from duckdb_constraints() group by 1"
        ).fetchall()
        counts = dict(constraints)
        assert (counts.get("FOREIGN KEY", 0) > 0) == bool(foreign_keys)
        assert counts["PRIMARY KEY"] > 0
        assert counts["NOT NULL"] > 0
        db.db.execute("insert into cluster(id) values (1)")
        with pytest.raises(duckdb.ConstraintException):
            db.db.execute("insert into cluster(id) values (1)")
        assert db.check_foreign_keys() == {}
        if foreign_keys:
            with pytest.raises(duckdb.ConstraintException):
                db.db.execute("insert into pathway values (1, 1, 2, 1)")
        else:
            db.db.execute("insert into pathway values (1, 1, 2, 1)")
            assert db.check_foreign_keys() == {"pathway(product1_id) -> cluster(id)": 1}


def test_integrity_composite_and_nullable_keys():
    with connection_scope(ExperimentDatabase, ":memory:") as db:
        db.create_tables(foreign_keys=False)
        db.db.execute("insert into cluster(id) values (1)")
        db.db.execute("insert into pathway values (1, 1, 1, 1)")
        db.db.execute("insert into cluster_tree_node values (1, 0, 1), (2, 1, 1)")
        db.db.execute(
            "insert into cluster_tree_transition values "
            "(1, 0, 0, 1, NULL), (1, 1, 0, 1, 1)"
        )
        # A matching node index from a different result does not satisfy the FK;
        # the NULL destination of the terminal transition is valid.
        expected = {
            "cluster_tree_transition(experiment_result_id, destination_node_index) -> "
            "cluster_tree_node(experiment_result_id, node_index)": 1
        }
        assert db.check_foreign_keys() == expected
        db.db.execute("insert into cluster_tree_node values (1, 1, 1)")
        assert db.check_foreign_keys() == {}


def test_integrity_cli_and_reopen(tmp_path):
    path = str(tmp_path / "results.duckdb")
    with connection_scope(ExperimentDatabase, path) as db:
        db.create_tables(foreign_keys=False)
    runner = CliRunner()
    assert runner.invoke(check_integrity, [path]).exit_code == 0
    with connection_scope(ExperimentDatabase, path) as db:
        db.db.execute("insert into cluster_tree_node values (42, 0, 999)")
    result = runner.invoke(check_integrity, [path])
    assert result.exit_code == 1
    assert (
        "cluster_tree_node(cluster_id) -> cluster(id): 1 orphan row(s)" in result.output
    )


@pytest.mark.parametrize("foreign_keys", [None, True, False])
def test_prepare_cli(tmp_path, foreign_keys):
    config = tmp_path / "config.toml"
    config.write_text("pathways = []\n")
    path = str(tmp_path / "results.duckdb")
    args = ["create", str(config), path, "--db-type=experiment"]
    if foreign_keys is not None:
        args.append("--foreign-keys" if foreign_keys else "--no-foreign-keys")
    runner = CliRunner(catch_exceptions=False)
    assert runner.invoke(prepare, args).exit_code == 0
    with connection_scope(ExperimentDatabase, path) as db:
        assert bool(
            db.db.execute(
                "select count(*) from duckdb_constraints() where constraint_type = 'FOREIGN KEY'"
            ).fetchone()[0]
        ) == bool(foreign_keys)
    assert runner.invoke(prepare, ["append", str(config), path]).exit_code == 0
    result = runner.invoke(prepare, ["append", str(config), path, "--foreign-keys"])
    assert result.exit_code == 2
    assert "only supported in create mode" in result.output


def test_attached_clusters_and_caller_transaction(tmp_path):
    from apitofsim.workflow import sql_files_nofk

    clusters = str(tmp_path / "clusters.duckdb")
    with connection_scope(SuperClusterDatabase, clusters) as db:
        db.create_tables()
        db.db.execute("insert into cluster(id) values (1)")
    with duckdb.connect(config={"threads": 1}) as conn:
        conn.execute(f"attach '{clusters}' as clusters_db (read_only)")
        conn.execute(f"attach '{tmp_path / 'results.duckdb'}' as results_db")
        conn.execute("use results_db")
        conn.execute("set search_path='results_db,clusters_db'")
        conn.execute(sql_files_nofk.experiment + sql_files_nofk.event_recording)
        db = ExperimentDatabase((conn, None))
        conn.execute("begin")
        conn.execute("insert into cluster_tree_node values (1, 0, 2)")
        assert db.check_foreign_keys() == {
            "cluster_tree_node(cluster_id) -> cluster(id)": 1
        }
        conn.execute("rollback")
        assert db.check_foreign_keys() == {}
