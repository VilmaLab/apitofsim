import sys

from apitofsim.db import ExperimentDatabase

in_path = sys.argv[1]
out_path = sys.argv[2]

db = ExperimentDatabase(in_path)
db.refresh_views()
print(db.experiment_summary_df().to_string(index=False))
er_id = None
while 1:
    er_id = input("Choose an experiment to output (-1 for all) > ")
    try:
        er_id = int(er_id)
    except ValueError:
        continue
    else:
        break

db.export(out_path, experiment_id=None if er_id == -1 else er_id)
