import argparse
import json

from mpi4py import MPI

from dense_retrieval.config import load_config
from dense_retrieval.data.storage import load_dataset
from dense_retrieval.paths import get_dataset_dir
from dense_retrieval.results import format_result_summary, save_run_outputs
from dense_retrieval.retrieval.mpi_centralized import run_mpi_centralized_retrieval
from dense_retrieval.retrieval.mpi_tree import run_mpi_tree_retrieval


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


def run_selected_retrieval(
    mode: str,
    vectors,
    queries,
    top_k: int,
    comm: MPI.Comm,
    load_time_sec: float,
):
    if mode == "mpi_centralized":
        return run_mpi_centralized_retrieval(
            vectors=vectors,
            queries=queries,
            top_k=top_k,
            comm=comm,
            load_time_sec=load_time_sec,
        )

    if mode == "mpi_tree":
        return run_mpi_tree_retrieval(
            vectors=vectors,
            queries=queries,
            top_k=top_k,
            comm=comm,
            load_time_sec=load_time_sec,
        )

    raise ValueError(f"Unknown retrieval mode: {mode}")


def main() -> None:
    args = parse_args()

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    config = load_config(args.config)
    dataset_dir = get_dataset_dir(config)

    if rank == 0:
        print(f"Loading dataset from: {dataset_dir}")

    load_start = MPI.Wtime()
    vectors, queries, metadata = load_dataset(dataset_dir)
    load_end = MPI.Wtime()
    load_time_sec = load_end - load_start

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

    result = run_selected_retrieval(
        mode=retrieval_cfg["mode"],
        vectors=vectors,
        queries=queries,
        top_k=search_cfg["top_k"],
        comm=comm,
        load_time_sec=load_time_sec,
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
