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
