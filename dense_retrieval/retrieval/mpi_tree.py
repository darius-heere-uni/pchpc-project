from typing import Any

import numpy as np
from mpi4py import MPI

from dense_retrieval.data.sharding import get_shard_bounds
from dense_retrieval.merge import merge_topk
from dense_retrieval.search.backends import search_topk


def run_mpi_tree_retrieval(
    vectors: np.ndarray,
    queries: np.ndarray,
    top_k: int,
    comm: MPI.Comm,
    load_time_sec: float | None = None,
    search_backend: str = "numpy",
    faiss_num_threads: int | None = None,
) -> dict[str, Any] | None:
    """
    MPI retrieval with tree-based merging.

    Each rank searches its local shard.
    Ranks then merge pairwise in a tree pattern until rank 0 holds the global top-k.
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

    current_scores = local_scores
    current_indices = local_indices + start_idx

    tree_communication_time_sec = 0.0
    local_merge_time_sec = 0.0

    reduction_start = MPI.Wtime()

    active = True
    step = 1

    while step < world_size:
        if active:
            if rank % (2 * step) == 0:
                partner = rank + step

                if partner < world_size:
                    communication_start = MPI.Wtime()

                    received_scores = comm.recv(
                        source=partner,
                        tag=1000 + step,
                    )
                    received_indices = comm.recv(
                        source=partner,
                        tag=2000 + step,
                    )

                    communication_end = MPI.Wtime()
                    tree_communication_time_sec += (
                        communication_end - communication_start
                    )

                    merge_start = MPI.Wtime()

                    current_scores, current_indices = merge_topk(
                        scores_list=[current_scores, received_scores],
                        indices_list=[current_indices, received_indices],
                        top_k=top_k,
                    )

                    merge_end = MPI.Wtime()
                    local_merge_time_sec += merge_end - merge_start

            else:
                target = rank - step

                communication_start = MPI.Wtime()

                comm.send(
                    current_scores,
                    dest=target,
                    tag=1000 + step,
                )
                comm.send(
                    current_indices,
                    dest=target,
                    tag=2000 + step,
                )

                communication_end = MPI.Wtime()
                tree_communication_time_sec += communication_end - communication_start

                active = False

        step *= 2

    reduction_end = MPI.Wtime()
    tree_reduction_time_sec = reduction_end - reduction_start

    rank_info = {
        "rank": rank,
        "world_size": world_size,
        "shard_start": start_idx,
        "shard_end": end_idx,
        "num_local_vectors": end_idx - start_idx,
        "load_time_sec": load_time_sec,
        "local_search_time_sec": search_end - search_start,
        "communication_time_sec": tree_communication_time_sec,
        "local_merge_time_sec": local_merge_time_sec,
    }

    gathered_rank_info = comm.gather(rank_info, root=0)

    if rank == 0:
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
            "scores": current_scores,
            "indices": current_indices,
            "metrics": {
                "retrieval_mode": "mpi_tree",
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
                "merge_time_sec": tree_reduction_time_sec,
                "tree_reduction_time_sec": tree_reduction_time_sec,
                "total_time_sec": total_time_sec,
                "rank_info": gathered_rank_info,
            },
        }

    return None
