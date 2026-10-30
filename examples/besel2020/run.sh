#!/usr/bin/bash

apitofsim db run --simulation-mode=single-cluster database.duckdb
apitofsim db run --simulation-mode=pathway-at-a-time database.duckdb
apitofsim db run --simulation-mode=cluster-tree database.duckdb
