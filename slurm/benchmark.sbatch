#!/bin/bash
#SBATCH --job-name=pchpc-bench
#SBATCH --partition=scc-cpu
#SBATCH --time=00:10:00
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --output=slurm/pchpc-bench-%j.out
#SBATCH --error=slurm/pchpc-bench-%j.err

set -euo pipefail

CONFIG_PATH="${1:-configs/cluster_benchmark.json}"

echo "Job started on: $(date)"
echo "Submit directory: $SLURM_SUBMIT_DIR"
echo "Job ID: $SLURM_JOB_ID"
echo "Config path: $CONFIG_PATH"
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
echo "Loading required modules."
module purge
module load miniforge3

echo
echo "Loaded modules:"
module list

echo
echo "Activating conda environment."
source activate mpi_course

echo
echo "Python executable:"
which python
python --version

echo
echo "MPI executable:"
which mpiexec

echo
echo "Python package check:"
python -c 'import numpy; import faiss; from mpi4py import MPI; print("environment OK"); print(MPI.Get_library_version())'

echo
echo "Testing MPI world under mpiexec:"
mpiexec -n "$SLURM_NTASKS" python -c 'from mpi4py import MPI; import socket, sys; comm = MPI.COMM_WORLD; print(f"rank {comm.Get_rank()} / {comm.Get_size()} on {socket.gethostname()} using {sys.executable}", flush=True)'

echo
echo "Resolving dataset directory:"
DATASET_DIR=$(python - "$CONFIG_PATH" <<'PY'
import sys
from dense_retrieval.config import load_config
from dense_retrieval.paths import get_dataset_dir

cfg = load_config(sys.argv[1])
print(get_dataset_dir(cfg))
PY
)

echo "DATASET_DIR=$DATASET_DIR"

echo
echo "Generating dataset if missing."
if [[ ! -f "$DATASET_DIR/vectors.npy" || ! -f "$DATASET_DIR/queries.npy" || ! -f "$DATASET_DIR/metadata.json" ]]; then
    python generate_dataset.py --config "$CONFIG_PATH"
else
    echo "Dataset already exists. Skipping generation."
fi

echo
echo "Running correctness check before benchmark."
mpiexec -n "$SLURM_NTASKS" python check_correctness.py --config "$CONFIG_PATH"

echo
echo "Running benchmark."
mpiexec -n "$SLURM_NTASKS" python run_benchmark.py --config "$CONFIG_PATH"

echo
echo "Job finished on: $(date)"
