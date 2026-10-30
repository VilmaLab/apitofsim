#!/bin/bash
shopt -s globstar
set -euo pipefail
IFS=$'\n\t'

7z x -y besel2020.7z -obesel2020

#(cd besel2020 && apitofsim generate pathways --guess-prefix xyz pathways.csv clusters.csv **/*.xyz)

bash ./prepare.sh
