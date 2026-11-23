#!/usr/bin/bash

apitofsim db run --simulation-mode=single-cluster database.apitofsim
apitofsim db run --simulation-mode=pathway-at-a-time database.apitofsim
apitofsim db run --simulation-mode=cluster-tree database.apitofsim
