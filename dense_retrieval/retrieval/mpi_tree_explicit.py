from typing import Any

import numpy as np
from mpi4py import MPI

from dense_retrieval.merge import merge_topk
from dense_retrieval.search.backends import search_topk
from dense_retrieval.search.local_index import LocalSearchIndex


def run_mpi_tree_explicit_retrieval(
    local_vectors: np.ndarray,
    queries: np.ndarray,
    top_k: int,
    comm: MPI.Comm,
    shard_start_idx: int,
    num_global_vectors: int,
    load_time_sec: float | None = None,
    search_backend: str = "numpy",
    faiss_num_threads: int | None = None,
    search_index: LocalSearchIndex | None = None,
) -> dict[str, Any] | None:
    """
    MPI retrieval with tree-based merging using explicit NumPy buffer communication.

    This variant is intentionally close to mpi_tree.py, but replaces mpi4py
    object communication via comm.send/comm.recv with buffer communication via
    comm.Send/comm.Recv for the large score/index arrays.
    """
    rank = comm.Get_rank()
    world_size = comm.Get_size()

    shard_end_idx = shard_start_idx + local_vectors.shape[0]

    comm.Barrier()
    mpi_total_start = MPI.Wtime()

    search_start = MPI.Wtime()

    if search_index is not None:
        local_scores, local_indices = search_index.search(
            queries=queries,
            top_k=top_k,
        )
    else:
        local_scores, local_indices = search_topk(
            vectors=local_vectors,
            queries=queries,
            top_k=top_k,
            backend=search_backend,
            faiss_num_threads=faiss_num_threads,
        )

    search_end = MPI.Wtime()

    # Buffer communication needs predictable contiguous NumPy buffers.
    current_scores = np.ascontiguousarray(local_scores, dtype=np.float32)
    current_indices = np.ascontiguousarray(
        local_indices + shard_start_idx,
        dtype=np.int64,
    )

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
                    received_scores = np.empty_like(current_scores)
                    received_indices = np.empty_like(current_indices)

                    communication_start = MPI.Wtime()

                    comm.Recv(
                        received_scores,
                        source=partner,
                        tag=1000 + step,
                    )
                    comm.Recv(
                        received_indices,
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

                    # Keep buffers contiguous and with fixed dtypes after merging.
                    current_scores = np.ascontiguousarray(
                        current_scores,
                        dtype=np.float32,
                    )
                    current_indices = np.ascontiguousarray(
                        current_indices,
                        dtype=np.int64,
                    )

                    merge_end = MPI.Wtime()
                    local_merge_time_sec += merge_end - merge_start

            else:
                target = rank - step

                current_scores = np.ascontiguousarray(
                    current_scores,
                    dtype=np.float32,
                )
                current_indices = np.ascontiguousarray(
                    current_indices,
                    dtype=np.int64,
                )

                communication_start = MPI.Wtime()

                comm.Send(
                    current_scores,
                    dest=target,
                    tag=1000 + step,
                )
                comm.Send(
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
        "shard_start": shard_start_idx,
        "shard_end": shard_end_idx,
        "num_local_vectors": local_vectors.shape[0],
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
                "retrieval_mode": "mpi_tree_explicit",
                "search_backend": search_backend,
                "faiss_num_threads": faiss_num_threads,
                "search_index_reuse": search_index is not None,
                "loading_strategy": "shard_aware_shared_npy",
                "vector_storage": "single_vectors_npy",
                "query_loading": "full_queries_on_each_rank",
                "world_size": world_size,
                "num_vectors": num_global_vectors,
                "num_queries": queries.shape[0],
                "dimension": local_vectors.shape[1],
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
