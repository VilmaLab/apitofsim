import sys

print(sys.path)
import os

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
    assert df["successes"].iloc[0] == 1
    assert df["failures"].iloc[0] == 0
