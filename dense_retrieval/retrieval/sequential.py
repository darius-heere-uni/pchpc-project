import numpy as np

from dense_retrieval.search.numpy_flat import search_topk_numpy


def run_sequential_retrieval(
    vectors: np.ndarray,
    queries: np.ndarray,
    top_k: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Run exact top-k retrieval on the full vector dataset in one process.
    """
    return search_topk_numpy(
        vectors=vectors,
        queries=queries,
        top_k=top_k,
    )
