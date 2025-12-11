import sys
from apitofsim.db import ExperimentDatabase, ExperimentRunner
import pint


ureg = pint.UnitRegistry()
Q_ = ureg.Quantity

db = ExperimentDatabase(sys.argv[1])
runner = ExperimentRunner(db)
runner.run_prepared_config()
