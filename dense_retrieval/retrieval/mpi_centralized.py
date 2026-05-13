from typing import Any

import numpy as np
from mpi4py import MPI

from dense_retrieval.data.sharding import get_shard_bounds
from dense_retrieval.merge import merge_topk
from dense_retrieval.search.numpy_flat import search_topk_numpy


def run_mpi_centralized_retrieval(
    vectors: np.ndarray,
    queries: np.ndarray,
    top_k: int,
    comm: MPI.Comm,
) -> dict[str, Any] | None:
    """
    MPI retrieval with centralized merging on rank 0.

    Each rank searches its local shard.
    Rank 0 gathers all local top-k results and merges them.
    """
    rank = comm.Get_rank()
    world_size = comm.Get_size()

    num_vectors = vectors.shape[0]
    start_idx, end_idx = get_shard_bounds(
        num_items=num_vectors,
        rank=rank,
        world_size=world_size,
    )

    local_vectors = vectors[start_idx:end_idx]

    comm.Barrier()
    total_start = MPI.Wtime()

    search_start = MPI.Wtime()

    local_scores, local_indices = search_topk_numpy(
        vectors=local_vectors,
        queries=queries,
        top_k=top_k,
    )

    search_end = MPI.Wtime()

    # Convert local shard indices to global vector indices.
    local_indices = local_indices + start_idx

    communication_start = MPI.Wtime()

    gathered_scores = comm.gather(local_scores, root=0)
    gathered_indices = comm.gather(local_indices, root=0)

    communication_end = MPI.Wtime()

    rank_info = {
        "rank": rank,
        "world_size": world_size,
        "shard_start": start_idx,
        "shard_end": end_idx,
        "num_local_vectors": end_idx - start_idx,
        "local_search_time_sec": search_end - search_start,
        "communication_time_sec": communication_end - communication_start,
    }

    gathered_rank_info = comm.gather(rank_info, root=0)

    if rank == 0:
        merge_start = MPI.Wtime()

        global_scores, global_indices = merge_topk(
            scores_list=gathered_scores,
            indices_list=gathered_indices,
            top_k=top_k,
        )

        merge_end = MPI.Wtime()

        total_end = MPI.Wtime()

        return {
            "scores": global_scores,
            "indices": global_indices,
            "metrics": {
                "world_size": world_size,
                "num_vectors": num_vectors,
                "num_queries": queries.shape[0],
                "dimension": vectors.shape[1],
                "top_k": top_k,
                "merge_time_sec": merge_end - merge_start,
                "total_time_sec": total_end - total_start,
                "rank_info": gathered_rank_info,
            },
        }

    return None
