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

### Run Configurable Slurm Job Script

Default 2 node test:

```bash
sbatch slurm/benchmark.sbatch configs/cluster_benchmark.json
```

Override Slurm resources from the command line:

```bash
sbatch --nodes=4 --ntasks-per-node=1 --cpus-per-task=1 slurm/benchmark.sbatch configs/cluster_benchmark.json
```

For a rank/thread experiment:

```bash
sbatch --nodes=1 --ntasks-per-node=4 --cpus-per-task=4 slurm/benchmark.sbatch configs/cluster_benchmark_faiss_threads4.json
```
_Note: faiss_num_threads inside the json config should match --cpus-per-task_

To skip the correctness check in larger benchmark runs:

```bash
sbatch --export=ALL,RUN_CORRECTNESS=0 --nodes=4 --ntasks-per-node=1 --cpus-per-task=1 slurm/benchmark.sbatch configs/cluster_benchmark.json
```


### Generate a Dataset using a Slurm Job

```bash
sbatch slurm/generate_dataset.sbatch configs/cluster_benchmark_calibrate.json
```

To force a re-generation:
```bash
sbatch --export=ALL,FORCE=1 slurm/generate_dataset.sbatch configs/cluster_benchmark_calibrate.json
```



### Collect Benchmark Results in CSV File

Finds all `benchmark_metrics.json` under `results/` and generates a CSV file for plotting.

```bash
python collect_benchmark_results.py \
  --results-root results \
  --output analysis/benchmark_results.csv
```

By default, if the output file already exists, the script 
adds a timestamp to the filename, so that the existing file 
is not overwritten. To force on overwriting of an existing result summary, use:

```bash
python collect_benchmark_results.py \
  --results-root results \
  --output analysis/benchmark_results.csv \
  --overwrite
```

Run it inside the conda environment if the Python on the cluster is too outdated:

```bash
module purge
module load miniforge3

conda run -n mpi_course python collect_benchmark_results.py \
  --results-root results \
  --output analysis/benchmark_results.csv
```

