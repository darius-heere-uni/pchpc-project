# pchpc-project

### Generate a Dataset given a Configuration

```bash
python generate_dataset.py --config configs/local.json
```

### Run Dense Vector Search using MPI Ranks

```bash
mpiexec -n 1 python run_retrieval.py --config configs/local.json
```