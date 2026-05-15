import numpy as np

from dense_retrieval.search.numpy_flat import search_topk_numpy


def search_topk(
    vectors: np.ndarray,
    queries: np.ndarray,
    top_k: int,
    backend: str,
    faiss_num_threads: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Dispatch exact top-k search to the selected local backend.
    """
    if backend == "numpy":
        return search_topk_numpy(
            vectors=vectors,
            queries=queries,
            top_k=top_k,
        )

    if backend == "faiss":
        from dense_retrieval.search.faiss_flat import search_topk_faiss

        return search_topk_faiss(
            vectors=vectors,
            queries=queries,
            top_k=top_k,
            num_threads=faiss_num_threads,
        )

    raise ValueError(f"Unknown search backend: {backend}")
