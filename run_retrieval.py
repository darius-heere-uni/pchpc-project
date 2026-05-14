import argparse
import json

from mpi4py import MPI

from dense_retrieval.config import load_config
from dense_retrieval.data.storage import load_dataset
from dense_retrieval.paths import get_dataset_dir
from dense_retrieval.results import format_result_summary, save_run_outputs
from dense_retrieval.retrieval.mpi_centralized import run_mpi_centralized_retrieval


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run exact dense-vector retrieval with MPI."
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to the JSON config file, e.g. configs/local.json",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    config = load_config(args.config)
    dataset_dir = get_dataset_dir(config)

    if rank == 0:
        print(f"Loading dataset from: {dataset_dir}")

    vectors, queries, metadata = load_dataset(dataset_dir)

    if rank == 0:
        print("Dataset loaded.")
        print("Dataset metadata:")
        print(json.dumps(metadata, indent=2))

    search_cfg = config["search"]
    retrieval_cfg = config["retrieval"]

    if search_cfg["backend"] != "numpy":
        raise ValueError(
            f"Only the numpy backend is implemented for now, "
            f"got: {search_cfg['backend']}"
        )

    if retrieval_cfg["mode"] != "mpi_centralized":
        raise ValueError(
            f"Only mpi_centralized mode is implemented for now, "
            f"got: {retrieval_cfg['mode']}"
        )

    result = run_mpi_centralized_retrieval(
        vectors=vectors,
        queries=queries,
        top_k=search_cfg["top_k"],
        comm=comm,
    )

    if rank == 0:
        print()
        print(format_result_summary(result))

        saved_paths = save_run_outputs(
            config=config,
            result=result,
        )

        print()
        print("Saved result files:")
        for name, path in saved_paths.items():
            print(f"  {name}: {path}")


if __name__ == "__main__":
    main()
