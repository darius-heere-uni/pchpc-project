import argparse

import numpy as np
from mpi4py import MPI

from dense_retrieval.config import load_config
from dense_retrieval.data.loading import (
    load_dataset_shard,
    load_full_dataset_for_baseline,
)
from dense_retrieval.paths import get_dataset_dir
from dense_retrieval.retrieval.mpi_centralized import run_mpi_centralized_retrieval
from dense_retrieval.retrieval.mpi_tree import run_mpi_tree_retrieval
from dense_retrieval.retrieval.sequential import run_sequential_retrieval
from dense_retrieval.retrieval.mpi_tree_explicit import run_mpi_tree_explicit_retrieval


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check MPI retrieval correctness against sequential baseline."
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to the JSON config file, e.g. configs/local.json",
    )
    parser.add_argument(
        "--atol",
        type=float,
        default=1e-5,
        help="Absolute tolerance for score comparison.",
    )
    parser.add_argument(
        "--rtol",
        type=float,
        default=1e-5,
        help="Relative tolerance for score comparison.",
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
            search_backend=search_backend,
            faiss_num_threads=faiss_num_threads,
        )

    raise ValueError(f"Unknown retrieval mode: {mode}")


def print_first_mismatch(
    mpi_indices: np.ndarray,
    seq_indices: np.ndarray,
    mpi_scores: np.ndarray,
    seq_scores: np.ndarray,
    rtol: float,
    atol: float,
) -> None:
    num_queries = seq_indices.shape[0]

    for query_id in range(num_queries):
        indices_match = np.array_equal(mpi_indices[query_id], seq_indices[query_id])
        scores_match = np.allclose(
            mpi_scores[query_id],
            seq_scores[query_id],
            rtol=rtol,
            atol=atol,
        )

        if not indices_match or not scores_match:
            print()
            print(f"First mismatch at query {query_id}:")
            print()
            print("Sequential indices:")
            print(seq_indices[query_id])
            print("MPI indices:")
            print(mpi_indices[query_id])
            print()
            print("Sequential scores:")
            print(seq_scores[query_id])
            print("MPI scores:")
            print(mpi_scores[query_id])
            return


def main() -> None:
    args = parse_args()

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    world_size = comm.Get_size()

    config = load_config(args.config)
    dataset_dir = get_dataset_dir(config)

    search_cfg = config["search"]
    retrieval_cfg = config["retrieval"]

    similarity = search_cfg.get("similarity", "dot")
    if similarity != "dot":
        raise ValueError(
            f"Only similarity='dot' is implemented for now, got: {similarity}"
        )

    search_backend = search_cfg["backend"]
    faiss_num_threads = search_cfg.get("faiss_num_threads")

    if rank == 0:
        print(f"Checking correctness with {world_size} MPI rank(s).")
        print(f"Retrieval mode: {retrieval_cfg['mode']}")
        print(f"Search backend: {search_backend}")
        if search_backend == "faiss":
            print(f"FAISS threads per rank: {faiss_num_threads}")
        print(f"Loading dataset from: {dataset_dir}")
        print("Loading strategy: shard-aware loading from shared vectors.npy")

    (
        local_vectors,
        queries,
        _metadata,
        shard_start,
        _shard_end,
        num_global_vectors,
    ) = load_dataset_shard(
        dataset_dir=dataset_dir,
        rank=rank,
        world_size=world_size,
    )

    top_k = search_cfg["top_k"]

    mpi_result = run_selected_retrieval(
        mode=retrieval_cfg["mode"],
        local_vectors=local_vectors,
        queries=queries,
        top_k=top_k,
        comm=comm,
        shard_start_idx=shard_start,
        num_global_vectors=num_global_vectors,
        search_backend=search_backend,
        faiss_num_threads=faiss_num_threads,
    )

    if rank == 0:
        print("Running sequential NumPy baseline on rank 0.")

        full_vectors, full_queries, _metadata = load_full_dataset_for_baseline(
            dataset_dir
        )

        seq_scores, seq_indices = run_sequential_retrieval(
            vectors=full_vectors,
            queries=full_queries,
            top_k=top_k,
        )

        mpi_scores = mpi_result["scores"]
        mpi_indices = mpi_result["indices"]

        shapes_match = (
            mpi_scores.shape == seq_scores.shape
            and mpi_indices.shape == seq_indices.shape
        )

        indices_match = np.array_equal(mpi_indices, seq_indices)
        scores_match = np.allclose(
            mpi_scores,
            seq_scores,
            rtol=args.rtol,
            atol=args.atol,
        )

        print()
        print("Correctness check summary:")
        print(f"  Shapes match:  {shapes_match}")
        print(f"  Indices match: {indices_match}")
        print(f"  Scores match:  {scores_match}")

        if shapes_match and indices_match and scores_match:
            print()
            print("Correctness check passed.")
        else:
            print()
            print("Correctness check failed.")
            print_first_mismatch(
                mpi_indices=mpi_indices,
                seq_indices=seq_indices,
                mpi_scores=mpi_scores,
                seq_scores=seq_scores,
                rtol=args.rtol,
                atol=args.atol,
            )
            raise SystemExit(1)


if __name__ == "__main__":
    main()
