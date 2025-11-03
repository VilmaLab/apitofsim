#!/bin/bash
#SBATCH -t 1:00:00
#SBATCH --partition=small
#SBATCH --account=project_2014955
#SBATCH --exclusive
#SBATCH --mem=0 
#SBATCH --nodes=1

if [ -n $SLURM_JOB_ID ];  then
    SCRIPT_PATH=$(realpath $(scontrol show job $SLURM_JOBID | awk -F= '/Command=/{print $2}'))
else
    SCRIPT_PATH=$(realpath $0)
fi
SCRIPT_DIR=$(dirname $SCRIPT_PATH)
. $SCRIPT_DIR/slurm-setup.sh

cd $SCRIPT_DIR/..

echo
echo "Clean"
make clean-all

echo
echo "Compile GCC"
module load gcc/14.2.0
make all all-gcc

echo
echo "Compile Intel"
module load intel-oneapi-compilers/2025.0.4
make all-icx

echo
echo "Benchmark"
cd $SCRIPT_DIR

VMS_BIN=${SCRIPT_DIR}/../bin \
BASE_INPUTS=${SCRIPT_DIR}/../inputs/example \
BENCHWORK=${SCRIPT_DIR}/../../benchwork \
COMPILERS='icx,gcc' \
MAX_NUM_THREADS=80 \
python3.11 -u bench.py
