#!/bin/bash
#SBATCH --job-name=pchpc-bench
#SBATCH --partition=scc-cpu
#SBATCH --time=00:03:00
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --output=slurm/pchpc-bench-%j.out
#SBATCH --error=slurm/pchpc-bench-%j.err

set -euo pipefail

CONFIG_PATH="${1:-configs/cluster_benchmark.json}"
RUN_CORRECTNESS="${RUN_CORRECTNESS:-1}"

echo "Job started on: $(date)"
echo "Submit directory: $SLURM_SUBMIT_DIR"
echo "Job ID: $SLURM_JOB_ID"
echo "Config path: $CONFIG_PATH"

echo
echo "Slurm resources:"
echo "  Nodes:            ${SLURM_JOB_NUM_NODES}"
echo "  Tasks total:      ${SLURM_NTASKS}"
echo "  Tasks per node:   ${SLURM_NTASKS_PER_NODE}"
echo "  CPUs per task:    ${SLURM_CPUS_PER_TASK:-1}"

echo
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
echo "Configuring thread environment."
CPUS_PER_TASK="${SLURM_CPUS_PER_TASK:-1}"

export OMP_NUM_THREADS="$CPUS_PER_TASK"
export OPENBLAS_NUM_THREADS="$CPUS_PER_TASK"
export MKL_NUM_THREADS="$CPUS_PER_TASK"
export NUMEXPR_NUM_THREADS="$CPUS_PER_TASK"
export OMP_PROC_BIND=close
export OMP_PLACES=cores

echo "OMP_NUM_THREADS=$OMP_NUM_THREADS"
echo "OPENBLAS_NUM_THREADS=$OPENBLAS_NUM_THREADS"
echo "MKL_NUM_THREADS=$MKL_NUM_THREADS"
echo "NUMEXPR_NUM_THREADS=$NUMEXPR_NUM_THREADS"
echo "OMP_PROC_BIND=$OMP_PROC_BIND"
echo "OMP_PLACES=$OMP_PLACES"

echo
echo "Python package check:"
python -c 'import numpy; import faiss; from mpi4py import MPI; print("environment OK"); print(MPI.Get_library_version())'

echo
echo "Checking config consistency:"
python - "$CONFIG_PATH" "$CPUS_PER_TASK" <<'PY'
import sys
from dense_retrieval.config import load_config

config_path = sys.argv[1]
cpus_per_task = int(sys.argv[2])

cfg = load_config(config_path)

search = cfg.get("search", {})
retrieval = cfg.get("retrieval", {})
benchmark = cfg.get("benchmark", {})

print(f"  run_id:             {cfg['project']['run_id']}")
print(f"  retrieval mode:     {retrieval.get('mode')}")
print(f"  search backend:     {search.get('backend')}")
print(f"  top_k:              {search.get('top_k')}")
print(f"  faiss_num_threads:  {search.get('faiss_num_threads')}")
print(f"  warmup_runs:        {benchmark.get('warmup_runs')}")
print(f"  measurement_runs:   {benchmark.get('measurement_runs')}")

faiss_threads = search.get("faiss_num_threads")
if search.get("backend") == "faiss" and faiss_threads is not None:
    if int(faiss_threads) != cpus_per_task:
        print()
        print("WARNING:")
        print(f"  Config faiss_num_threads={faiss_threads},")
        print(f"  but Slurm cpus-per-task={cpus_per_task}.")
        print("  For rank/thread experiments these should usually match.")
PY

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

if [[ "$RUN_CORRECTNESS" == "1" ]]; then
    echo
    echo "Running correctness check before benchmark."
    mpiexec -n "$SLURM_NTASKS" python check_correctness.py --config "$CONFIG_PATH"
else
    echo
    echo "Skipping correctness check because RUN_CORRECTNESS=$RUN_CORRECTNESS"
fi

echo
echo "Running benchmark."
mpiexec -n "$SLURM_NTASKS" python run_benchmark.py --config "$CONFIG_PATH"

echo
echo "Job finished on: $(date)"
