from pathlib import Path
from typing import Any

import numpy as np

from dense_retrieval.data.sharding import get_shard_bounds
from dense_retrieval.data.storage import load_dataset


def load_dataset_shard(
    dataset_dir: str | Path,
    rank: int,
    world_size: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any], int, int, int]:
    """
    Load only the vector shard assigned to this MPI rank.

    The global vector matrix is stored once as vectors.npy. Each rank opens the
    file in memory-mapped mode, selects its own shard, and copies only that shard
    into local RAM.

    Queries are loaded fully by every rank, because all ranks need to evaluate
    the same query workload.

    Returns
    -------
    local_vectors:
        Vector shard assigned to this rank.
    queries:
        Full query matrix.
    metadata:
        Dataset metadata.
    shard_start:
        Global start index of this rank's shard.
    shard_end:
        Global end index of this rank's shard.
    num_global_vectors:
        Total number of vectors in the full dataset.
    """
    dataset_path = Path(dataset_dir)

    vectors_path = dataset_path / "vectors.npy"
    queries_path = dataset_path / "queries.npy"
    metadata_path = dataset_path / "metadata.json"

    if not vectors_path.exists():
        raise FileNotFoundError(f"Missing vectors file: {vectors_path}")
    if not queries_path.exists():
        raise FileNotFoundError(f"Missing queries file: {queries_path}")
    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing metadata file: {metadata_path}")

    # Open the global vector matrix without eagerly loading the full file.
    vectors_mmap = np.load(vectors_path, mmap_mode="r")
    num_global_vectors = vectors_mmap.shape[0]

    shard_start, shard_end = get_shard_bounds(
        num_items=num_global_vectors,
        rank=rank,
        world_size=world_size,
    )

    # Copy only this rank's shard into local memory.
    # This makes the local search operate on a normal NumPy array.
    local_vectors = np.asarray(vectors_mmap[shard_start:shard_end]).copy()

    # Queries are usually much smaller and are needed by every rank.
    queries = np.load(queries_path)

    # Reuse the existing metadata loader by loading only metadata through the
    # existing load_dataset function would also load arrays, so read directly here.
    import json

    with metadata_path.open("r", encoding="utf-8") as f:
        metadata = json.load(f)

    return (
        local_vectors,
        queries,
        metadata,
        shard_start,
        shard_end,
        num_global_vectors,
    )


def load_full_dataset_for_baseline(
    dataset_dir: str | Path,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """
    Load the full dataset.

    This is mainly used by correctness checks on rank 0 to compute the
    sequential reference result.
    """
    return load_dataset(dataset_dir)
