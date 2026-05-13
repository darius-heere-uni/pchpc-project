import argparse
import json

from mpi4py import MPI

from dense_retrieval.config import load_config
from dense_retrieval.data.storage import load_dataset
from dense_retrieval.paths import get_dataset_dir
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


def print_result_summary(result: dict) -> None:
    metrics = result["metrics"]
    scores = result["scores"]
    indices = result["indices"]

    print()
    print("Retrieval finished.")
    print()
    print("Run summary:")
    print(f"  MPI ranks:      {metrics['world_size']}")
    print(f"  Vectors:        {metrics['num_vectors']}")
    print(f"  Queries:        {metrics['num_queries']}")
    print(f"  Dimension:      {metrics['dimension']}")
    print(f"  Top-k:          {metrics['top_k']}")
    print(f"  Total time:     {metrics['total_time_sec']:.6f} s")
    print(f"  Merge time:     {metrics['merge_time_sec']:.6f} s")

    print()
    print("Shard distribution:")
    for info in metrics["rank_info"]:
        print(
            f"  Rank {info['rank']:>2}: "
            f"[{info['shard_start']}, {info['shard_end']}) "
            f"({info['num_local_vectors']} vectors), "
            f"search={info['local_search_time_sec']:.6f}s, "
            f"comm={info['communication_time_sec']:.6f}s"
        )

    print()
    print("Top-k preview for first query:")
    first_query_indices = indices[0].tolist()
    first_query_scores = scores[0].tolist()

    for position, (idx, score) in enumerate(
        zip(first_query_indices, first_query_scores),
        start=1,
    ):
        print(f"  {position:>2}. vector_id={idx:>8}, score={score:.6f}")


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
        print_result_summary(result)


if __name__ == "__main__":
    main()
