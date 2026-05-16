# pchpc-project

### Generate a Dataset given a Configuration

```bash
python generate_dataset.py --config configs/local.json
```

### Run Dense Vector Search using MPI Ranks

```bash
mpiexec -n 1 python run_retrieval.py --config configs/local.json
```

###  Run Correctness Check by Comparing vs Sequential Run on Rank 0

```bash
mpiexec -n 4 python check_correctness.py --config configs/local.json
```


### Schedule a Slurm Batch Job via the respective Config File

```bash
sbatch slurm/smoke_test.sbatch
```



## Environment setup

The project uses a Conda environment called `mpi_course`.

### Local Ubuntu Laptop

Create and activate the environment:

```bash
conda create -n mpi_course -c conda-forge python=3.12 mpi4py numpy faiss-cpu
conda activate mpi_course
```

Quick Check:

```bash
python -c '
import numpy
import faiss
from mpi4py import MPI

print("environment OK")
'

mpiexec -n 2 python -c '
from mpi4py import MPI

comm = MPI.COMM_WORLD
print(f"hello from rank {comm.Get_rank()} of {comm.Get_size()}")
'
```

## Cluster

Create and activate the environment:

```bash
cd ~/pchpc-project

module purge
module load miniforge3

conda create -n mpi_course -c conda-forge python=3.12 mpi4py numpy faiss-cpu
conda activate mpi_course
```

Quick Check:

```bash
which python
python --version

python -c '
import numpy
import faiss
from mpi4py import MPI

print("environment OK")
print(MPI.Get_library_version())
'

which mpiexec

mpiexec -n 2 python -c '
from mpi4py import MPI

comm = MPI.COMM_WORLD
print(f"hello from rank {comm.Get_rank()} of {comm.Get_size()}")
'
```