echo "Script path: $SCRIPT_PATH"

echo "Got allocated resources:"
echo "  Job ID: $SLURM_JOB_ID"
echo "  Nodes: $SLURM_JOB_NODELIST"
echo "  CPUs: $SLURM_CPUS_PER_TASK"
echo "  Memory: $SLURM_MEM_PER_NODE MB"
echo