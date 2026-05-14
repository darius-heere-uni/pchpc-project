import argparse

import numpy as np
from mpi4py import MPI

from dense_retrieval.config import load_config
from dense_retrieval.data.storage import load_dataset
from dense_retrieval.paths import get_dataset_dir
from dense_retrieval.retrieval.mpi_centralized import run_mpi_centralized_retrieval
from dense_retrieval.retrieval.sequential import run_sequential_retrieval


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


def print_first_mismatch(
    mpi_indices: np.ndarray,
    seq_indices: np.ndarray,
    mpi_scores: np.ndarray,
    seq_scores: np.ndarray,
) -> None:
    """
    Print a small diagnostic for the first query where results differ.
    """
    num_queries = seq_indices.shape[0]

    for query_id in range(num_queries):
        indices_match = np.array_equal(mpi_indices[query_id], seq_indices[query_id])
        scores_match = np.allclose(
            mpi_scores[query_id],
            seq_scores[query_id],
            rtol=1e-5,
            atol=1e-5,
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

    if rank == 0:
        print(f"Checking correctness with {world_size} MPI rank(s).")
        print(f"Loading dataset from: {dataset_dir}")

    vectors, queries, _metadata = load_dataset(dataset_dir)

    search_cfg = config["search"]
    retrieval_cfg = config["retrieval"]

    if search_cfg["backend"] != "numpy":
        raise ValueError(
            f"Correctness check currently supports only backend='numpy', "
            f"got: {search_cfg['backend']}"
        )

    if retrieval_cfg["mode"] != "mpi_centralized":
        raise ValueError(
            f"Correctness check currently supports only mode='mpi_centralized', "
            f"got: {retrieval_cfg['mode']}"
        )

    top_k = search_cfg["top_k"]

    mpi_result = run_mpi_centralized_retrieval(
        vectors=vectors,
        queries=queries,
        top_k=top_k,
        comm=comm,
    )

    if rank == 0:
        print("Running sequential baseline on rank 0.")

        seq_scores, seq_indices = run_sequential_retrieval(
            vectors=vectors,
            queries=queries,
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
            )
            raise SystemExit(1)


if __name__ == "__main__":
    main()
