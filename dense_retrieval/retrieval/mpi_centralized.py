from typing import Any

import numpy as np
from mpi4py import MPI

from dense_retrieval.data.sharding import get_shard_bounds
from dense_retrieval.merge import merge_topk
from dense_retrieval.search.backends import search_topk


def run_mpi_centralized_retrieval(
    vectors: np.ndarray,
    queries: np.ndarray,
    top_k: int,
    comm: MPI.Comm,
    load_time_sec: float | None = None,
    search_backend: str = "numpy",
    faiss_num_threads: int | None = None,
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
    mpi_total_start = MPI.Wtime()

    search_start = MPI.Wtime()

    local_scores, local_indices = search_topk(
        vectors=local_vectors,
        queries=queries,
        top_k=top_k,
        backend=search_backend,
        faiss_num_threads=faiss_num_threads,
    )

    search_end = MPI.Wtime()

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
        "load_time_sec": load_time_sec,
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
        mpi_total_end = MPI.Wtime()

        mpi_total_time_sec = mpi_total_end - mpi_total_start

        load_times = [
            info["load_time_sec"]
            for info in gathered_rank_info
            if info["load_time_sec"] is not None
        ]

        if load_times:
            load_time_sec_max = max(load_times)
            load_time_sec_mean = sum(load_times) / len(load_times)
            total_time_sec = load_time_sec_max + mpi_total_time_sec
        else:
            load_time_sec_max = None
            load_time_sec_mean = None
            total_time_sec = mpi_total_time_sec

        return {
            "scores": global_scores,
            "indices": global_indices,
            "metrics": {
                "retrieval_mode": "mpi_centralized",
                "search_backend": search_backend,
                "faiss_num_threads": faiss_num_threads,
                "world_size": world_size,
                "num_vectors": num_vectors,
                "num_queries": queries.shape[0],
                "dimension": vectors.shape[1],
                "top_k": top_k,
                "load_time_sec_max": load_time_sec_max,
                "load_time_sec_mean": load_time_sec_mean,
                "mpi_total_time_sec": mpi_total_time_sec,
                "merge_time_sec": merge_end - merge_start,
                "total_time_sec": total_time_sec,
                "rank_info": gathered_rank_info,
            },
        }

    return None
