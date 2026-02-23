import sys
from apitofsim.workflow import ExperimentDatabase, ExperimentRunner
import pint

ureg = pint.UnitRegistry()
Q_ = ureg.Quantity

db = ExperimentDatabase(sys.argv[1])
runner = ExperimentRunner(db)
print()
print("# Multiple pathway simulation")
print()
runner.run_prepared_config(
    [
        "improved",
        "improved, quadrupole",
    ],
    strict_dos=False,
    pathway_at_a_time=False,
)
print()
print()
print("# Pathway-at-a-time simulation")
print()
runner.run_prepared_config(
    [
        "improved",
        "improved, quadrupole",
    ],
    strict_dos=False,
    pathway_at_a_time=True,
)
