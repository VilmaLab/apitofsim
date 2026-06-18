from enum import Enum


class SimulationMode(Enum):
    PATHWAY_AT_A_TIME = 0
    SINGLE_CLUSTER = 1
    CLUSTER_TREE = 2
