cd $SCRIPT_DIR/..

echo
echo "Clean"
make clean-all

echo
echo "Compile GCC"
module purge && module load foss/2024a
make all all-gcc

echo
echo "Compile Intel"
module purge && module load intel-compilers/2023.1.0
make all-icx

echo
echo "Benchmark"
cd $SCRIPT_DIR

module purge && module load Python/3.11.5-GCCcore-13.2.0

VMS_BIN=${SCRIPT_DIR}/../bin \
BASE_INPUTS=${SCRIPT_DIR}/../inputs/example/ \
COMPILERS='icx,gcc' \
MAX_NUM_THREADS=80 \
PRE_COMPILER_ICX='module purge && module load intel-compilers/2023.1.0 imkl/2024.2.0' \
PRE_COMPILER_GCC='module purge && module load foss/2024a' \
python3.11 -u bench.py