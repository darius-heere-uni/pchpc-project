#!/bin/bash
#SBATCH --job-name=pchpc-smoke
#SBATCH --partition=scc-cpu
#SBATCH --time=00:05:00
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --output=slurm/pchpc-smoke-%j.out
#SBATCH --error=slurm/pchpc-smoke-%j.err

set -euo pipefail

echo "Job started on: $(date)"
echo "Submit directory: $SLURM_SUBMIT_DIR"
echo "Job ID: $SLURM_JOB_ID"
echo "Allocated nodes:"
scontrol show hostnames "$SLURM_JOB_NODELIST"

echo
echo "Changing to submit directory."
cd "$SLURM_SUBMIT_DIR"

echo
echo "Checking PROJECT_DIR:"
: "${PROJECT_DIR:?PROJECT_DIR is not set}"
echo "PROJECT_DIR=$PROJECT_DIR"

echo
echo "Activating conda environment."
if command -v conda >/dev/null 2>&1; then
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate mpi_course
else
    echo "ERROR: conda command not found."
    echo "You may need to load the cluster's conda/miniforge module before activating mpi_course."
    exit 1
fi

echo
echo "Python executable:"
which python
python --version

echo
echo "Testing mpi4py import under srun:"
srun python -c 'from mpi4py import MPI; import socket; comm = MPI.COMM_WORLD; print(f"rank {comm.Get_rank()} / {comm.Get_size()} on {socket.gethostname()}")'

echo
echo "Resolving dataset directory from configs/cluster.json:"
DATASET_DIR=$(python - <<'PY'
from dense_retrieval.config import load_config
from dense_retrieval.paths import get_dataset_dir

cfg = load_config("configs/cluster.json")
print(get_dataset_dir(cfg))
PY
)

echo "DATASET_DIR=$DATASET_DIR"

echo
echo "Generating dataset if missing."
if [[ ! -f "$DATASET_DIR/vectors.npy" || ! -f "$DATASET_DIR/queries.npy" || ! -f "$DATASET_DIR/metadata.json" ]]; then
    python generate_dataset.py --config configs/cluster.json
else
    echo "Dataset already exists. Skipping generation."
fi

echo
echo "Running correctness check."
srun python check_correctness.py --config configs/cluster.json

echo
echo "Running retrieval."
srun python run_retrieval.py --config configs/cluster.json

echo
echo "Job finished on: $(date)"
