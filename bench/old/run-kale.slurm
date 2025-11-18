#!/bin/bash
#SBATCH -t 1:00:00
#SBATCH --job-name=vms-bench
#SBATCH --clusters=kale
#SBATCH --partition=short
#SBATCH --exclusive
#SBATCH --mem=0 
#SBATCH --nodes=1

if [ -n $SLURM_JOB_ID ];  then
    SCRIPT_PATH=$(realpath $(scontrol show job $SLURM_JOBID | awk -F= '/Command=/{print $2}'))
else
    SCRIPT_PATH=$(realpath $0)
fi

export BENCHWORK=${SCRIPT_DIR}/../../benchwork_kale

SCRIPT_DIR=$(dirname $SCRIPT_PATH)
. $SCRIPT_DIR/slurm-setup.sh
. $SCRIPT_DIR/slurm-common-uh-hpc.sh