#!/bin/bash
#SBATCH --job-name=check-faiss-threads
#SBATCH --partition=scc-cpu
#SBATCH --time=00:01:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=2G
#SBATCH --output=slurm/check-faiss-threads-%j.out
#SBATCH --error=slurm/check-faiss-threads-%j.err

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"

module purge
module load miniforge3

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate mpi_course

export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export NUMEXPR_NUM_THREADS="${SLURM_CPUS_PER_TASK}"
export OMP_PROC_BIND=close
export OMP_PLACES=cores

echo "Slurm allocation:"
echo "  SLURM_JOB_ID=${SLURM_JOB_ID:-unset}"
echo "  SLURM_JOB_NODELIST=${SLURM_JOB_NODELIST:-unset}"
echo "  SLURM_NTASKS=${SLURM_NTASKS:-unset}"
echo "  SLURM_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK:-unset}"

echo
echo "Thread environment:"
echo "  OMP_NUM_THREADS=$OMP_NUM_THREADS"
echo "  OPENBLAS_NUM_THREADS=$OPENBLAS_NUM_THREADS"
echo "  MKL_NUM_THREADS=$MKL_NUM_THREADS"
echo "  NUMEXPR_NUM_THREADS=$NUMEXPR_NUM_THREADS"
echo "  OMP_PROC_BIND=$OMP_PROC_BIND"
echo "  OMP_PLACES=$OMP_PLACES"

echo
echo "Python / FAISS thread check:"
python - <<'PY'
import os
import faiss

print("python pid:", os.getpid())
print("faiss.omp_get_max_threads():", faiss.omp_get_max_threads())

faiss.omp_set_num_threads(int(os.environ["OMP_NUM_THREADS"]))
print("after faiss.omp_set_num_threads:")
print("faiss.omp_get_max_threads():", faiss.omp_get_max_threads())
PY
