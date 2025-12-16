import os
import pytest

from apitofsim.config import ConfigFile
from apitofsim.db import ExperimentDatabase, ExperimentRunner, ingest_legacy_one


def test_runner():
    data_dir = os.environ.get("DATA_DIR")
    assert data_dir is not None, "DATA_DIR environment variable not set"
    config_filename = data_dir + "/config.in"
    db = ExperimentDatabase(":memory:")
    db.create_tables()
    ingest_legacy_one(db, config_filename)
    config = ConfigFile(filename=config_filename)
    config = config.into_json_config()
    config["N"] = 2
    db.insert_config("test", config)
    runner = ExperimentRunner(db)
    runner.run_prepared_config()
    df = db.experiment_summary_df()
    if not (df["successes"].iloc[0] == 1 and df["failures"].iloc[0] == 0):
        if df["successes"].iloc[0] == 0 and df["failures"].iloc[0] == 1:
            fail_df = db.db.table("experiment_failure").fetchdf()
            exc_name = fail_df["exc_name"].iloc[0]
            msg = fail_df["msg"].iloc[0]
            pytest.fail(f"Test run failed with exception {exc_name}: {msg}")
        assert df["successes"].iloc[0] == 1 and df["failures"].iloc[0] == 0, (
            "Unexpected number of successes/failures"
        )
