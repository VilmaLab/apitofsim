import os
from sys import argv
from os import unlink
import pint
import orjson

from apitofsim.workflow import ingest_tree, ExperimentDatabase


def iter_raw_configs(json):
    for config in json.get("configs", []):
        yield config["name"], {**json.get("default_config", {}), **config}


ureg = pint.UnitRegistry()
Q_ = ureg.Quantity


def main():
    infn = argv[1]
    db_name = argv[2]
    if os.path.exists(db_name):
        unlink(db_name)
    db = ExperimentDatabase(db_name)
    db.create_tables()

    with open(infn, "rb") as f:
        source = orjson.loads(f.read())

    ingest_tree(db, source["pathways"])

    for name, config in iter_raw_configs(source):
        db.insert_config(name, config)


if __name__ == "__main__":
    main()
