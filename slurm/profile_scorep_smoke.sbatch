#!/bin/bash
#SBATCH --job-name=pchpc-scorep
#SBATCH --partition=scc-cpu
#SBATCH --time=00:05:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --output=slurm/pchpc-scorep-%j.out
#SBATCH --error=slurm/pchpc-scorep-%j.err

set -euo pipefail

CONFIG_PATH="${1:-configs/cluster_scorep_smoke.json}"

echo "Job started on: $(date)"
echo "Submit directory: $SLURM_SUBMIT_DIR"
echo "Job ID: $SLURM_JOB_ID"
echo "Config path: $CONFIG_PATH"

cd "$SLURM_SUBMIT_DIR"

echo
echo "Loading profiling modules."
module purge
module load gcc/14.2.0
module load openmpi/4.1.7
module load scorep/8.4
module load miniforge3

echo
echo "Activating profiling conda environment."
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate mpi_course_scorep

echo
echo "Tool check:"
which python
which mpiexec
which scorep
python -c "from mpi4py import MPI; print(MPI.Get_library_version())"
python -c "import scorep; import numpy; import dense_retrieval; print('scorep profiling env OK')"

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
echo "Preparing Score-P output directory."
export SCOREP_EXPERIMENT_DIRECTORY="$SLURM_SUBMIT_DIR/scorep_runs/scorep_smoke_${SLURM_JOB_ID}"
export SCOREP_ENABLE_TRACING=true
export SCOREP_ENABLE_PROFILING=false
export SCOREP_TOTAL_MEMORY=512M

rm -rf "$SCOREP_EXPERIMENT_DIRECTORY"
mkdir -p "$(dirname "$SCOREP_EXPERIMENT_DIRECTORY")"

echo "SCOREP_EXPERIMENT_DIRECTORY=$SCOREP_EXPERIMENT_DIRECTORY"
echo "SCOREP_ENABLE_TRACING=$SCOREP_ENABLE_TRACING"
echo "SCOREP_ENABLE_PROFILING=$SCOREP_ENABLE_PROFILING"

echo
echo "Running Score-P trace."
mpiexec -n "$SLURM_NTASKS" python -m scorep --mpp=mpi --noinstrumenter run_benchmark.py --config "$CONFIG_PATH"

echo
echo "Generated Score-P files:"
find "$SCOREP_EXPERIMENT_DIRECTORY" -maxdepth 2 -type f | sort

echo
echo "Job finished on: $(date)"
