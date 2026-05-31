#!/bin/bash
#SBATCH --job-name=pchpc-gen-data
#SBATCH --partition=scc-cpu
#SBATCH --time=00:10:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --output=slurm/pchpc-gen-data-%j.out
#SBATCH --error=slurm/pchpc-gen-data-%j.err

set -euo pipefail

CONFIG_PATH="${1:-configs/cluster_benchmark_calibrate.json}"
FORCE="${FORCE:-0}"

echo "Job started on: $(date)"
echo "Submit directory: $SLURM_SUBMIT_DIR"
echo "Job ID: $SLURM_JOB_ID"
echo "Config path: $CONFIG_PATH"

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
echo "Checking whether dataset already exists."
if [[ "$FORCE" != "1" && -f "$DATASET_DIR/vectors.npy" && -f "$DATASET_DIR/queries.npy" && -f "$DATASET_DIR/metadata.json" ]]; then
    echo "Dataset already exists. Skipping generation."
    echo "Use FORCE=1 to overwrite/regenerate."
else
    echo "Generating dataset."
    python generate_dataset.py --config "$CONFIG_PATH"
fi

echo
echo "Generated dataset files:"
ls -lh "$DATASET_DIR"

echo
echo "Job finished on: $(date)"
