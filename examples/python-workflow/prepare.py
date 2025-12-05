import os
from glob import glob
from sys import argv
from json import load
from os import unlink
import pickle
import pint

from apitofsim.db import ingest_legacy, ClusterDatabase
from apitofsim.config import (
    ConfigFile,
    TOPLEVEL,
)


ureg = pint.UnitRegistry()
Q_ = ureg.Quantity


def main():
    infn = argv[1]
    out_path = argv[2]
    db_name = out_path + ".duckdb"
    if os.path.exists(db_name):
        unlink(db_name)
    db = ClusterDatabase(db_name)
    db.create_tables()

    with open(infn) as f:
        source = load(f)

    ingest_legacy(db, source["path"], source.get("backup_search"))
    filename = glob(source["path"], recursive=True)[0]
    config = ConfigFile(filename=filename)
    common_config_out = {
        k: config.get(k, by="short_name")
        for k in (["lengths", "voltages", "gas"] + TOPLEVEL)
    }
    with open(out_path + ".pkl", "wb") as outf:
        pickle.dump(common_config_out, outf)


if __name__ == "__main__":
    main()
