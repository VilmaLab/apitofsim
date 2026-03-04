#!/bin/bash
#SBATCH -t 12:00:00
#SBATCH --job-name=apitofsim_validation
#SBATCH --partition=short
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --nodes=1

# Make sure to clone the repository to ~/apitofsim and put the following at ~/apitofsim/uv.toml
#
# no-build-isolation-package = ["apitofsim"]
# [config-settings]
# setup-args = [
#   "--native-file=~/apitofsim/meson/dev/clangrelease.ini",
#   "-Dbuildtype=release",
#]

set -euo pipefail
IFS=$'\n\t'

echo "Got allocated resources:"
echo "  Job ID: $SLURM_JOB_ID"
echo "  Nodes: $SLURM_JOB_NODELIST"
echo "  CPUs: $SLURM_CPUS_ON_NODE"
echo "  Memory: $SLURM_MEM_PER_NODE MB"

module load LLVM/18.1.8-GCCcore-13.3.0; module load Meson/1.8.2-GCCcore-14.3.0

cd ~/apitofsim
export PTABLE_INTERACTIVE=0
uv run python -m apitofsim db run validation.duckdb
uv run python -m apitofsim db run --pathway-at-a-time validation.duckdb

echo "DONE"
