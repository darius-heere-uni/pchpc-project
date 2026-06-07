import argparse
import json

from mpi4py import MPI

from dense_retrieval.config import load_config
from dense_retrieval.data.loading import load_dataset_shard
from dense_retrieval.paths import get_dataset_dir
from dense_retrieval.results import format_result_summary, save_run_outputs
from dense_retrieval.retrieval.mpi_centralized import run_mpi_centralized_retrieval
from dense_retrieval.retrieval.mpi_tree import run_mpi_tree_retrieval
from dense_retrieval.retrieval.mpi_tree_explicit import run_mpi_tree_explicit_retrieval


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
    local_vectors,
    queries,
    top_k: int,
    comm: MPI.Comm,
    shard_start_idx: int,
    num_global_vectors: int,
    load_time_sec: float,
    search_backend: str,
    faiss_num_threads: int | None,
):
    if mode == "mpi_centralized":
        return run_mpi_centralized_retrieval(
            local_vectors=local_vectors,
            queries=queries,
            top_k=top_k,
            comm=comm,
            shard_start_idx=shard_start_idx,
            num_global_vectors=num_global_vectors,
            load_time_sec=load_time_sec,
            search_backend=search_backend,
            faiss_num_threads=faiss_num_threads,
        )

    if mode == "mpi_tree":
        return run_mpi_tree_retrieval(
            local_vectors=local_vectors,
            queries=queries,
            top_k=top_k,
            comm=comm,
            shard_start_idx=shard_start_idx,
            num_global_vectors=num_global_vectors,
            load_time_sec=load_time_sec,
            search_backend=search_backend,
            faiss_num_threads=faiss_num_threads,
        )

    if mode == "mpi_tree_explicit":
        return run_mpi_tree_explicit_retrieval(
            local_vectors=local_vectors,
            queries=queries,
            top_k=top_k,
            comm=comm,
            shard_start_idx=shard_start_idx,
            num_global_vectors=num_global_vectors,
            load_time_sec=load_time_sec,
            search_backend=search_backend,
            faiss_num_threads=faiss_num_threads,
        )

    raise ValueError(f"Unknown retrieval mode: {mode}")


def main() -> None:
    args = parse_args()

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    world_size = comm.Get_size()

    config = load_config(args.config)
    dataset_dir = get_dataset_dir(config)

    if rank == 0:
        print(f"Loading dataset from: {dataset_dir}")
        print("Loading strategy: shard-aware loading from shared vectors.npy")

    load_start = MPI.Wtime()

    (
        local_vectors,
        queries,
        metadata,
        shard_start,
        shard_end,
        num_global_vectors,
    ) = load_dataset_shard(
        dataset_dir=dataset_dir,
        rank=rank,
        world_size=world_size,
    )

    load_end = MPI.Wtime()
    load_time_sec = load_end - load_start

    if rank == 0:
        print("Dataset shard loaded.")
        print("Dataset metadata:")
        print(json.dumps(metadata, indent=2))

    search_cfg = config["search"]
    retrieval_cfg = config["retrieval"]

    similarity = search_cfg.get("similarity", "dot")
    if similarity != "dot":
        raise ValueError(
            f"Only similarity='dot' is implemented for now, got: {similarity}"
        )

    search_backend = search_cfg["backend"]
    faiss_num_threads = search_cfg.get("faiss_num_threads")

    result = run_selected_retrieval(
        mode=retrieval_cfg["mode"],
        local_vectors=local_vectors,
        queries=queries,
        top_k=search_cfg["top_k"],
        comm=comm,
        shard_start_idx=shard_start,
        num_global_vectors=num_global_vectors,
        load_time_sec=load_time_sec,
        search_backend=search_backend,
        faiss_num_threads=faiss_num_threads,
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
