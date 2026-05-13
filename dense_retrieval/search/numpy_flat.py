import numpy as np


def search_topk_numpy(
    vectors: np.ndarray,
    queries: np.ndarray,
    top_k: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Exact top-k search using NumPy dot products.

    Parameters
    ----------
    vectors:
        Matrix of shape (num_vectors, dimension).
    queries:
        Matrix of shape (num_queries, dimension).
    top_k:
        Number of nearest vectors to return per query.

    Returns
    -------
    scores:
        Array of shape (num_queries, top_k_eff), sorted descending.
    indices:
        Local vector indices of shape (num_queries, top_k_eff), sorted by score.
    """
    if vectors.ndim != 2:
        raise ValueError(f"vectors must be 2D, got shape {vectors.shape}")

    if queries.ndim != 2:
        raise ValueError(f"queries must be 2D, got shape {queries.shape}")

    if vectors.shape[1] != queries.shape[1]:
        raise ValueError(
            f"Dimension mismatch: vectors have dim {vectors.shape[1]}, "
            f"queries have dim {queries.shape[1]}"
        )

    num_vectors = vectors.shape[0]

    if num_vectors == 0:
        empty_scores = np.empty((queries.shape[0], 0), dtype=np.float32)
        empty_indices = np.empty((queries.shape[0], 0), dtype=np.int64)
        return empty_scores, empty_indices

    top_k_eff = min(top_k, num_vectors)

    # Dot product similarity: shape (num_queries, num_vectors)
    scores = queries @ vectors.T

    # Select top-k candidates without fully sorting all scores.
    candidate_indices = np.argpartition(
        -scores,
        kth=top_k_eff - 1,
        axis=1,
    )[:, :top_k_eff]

    candidate_scores = np.take_along_axis(scores, candidate_indices, axis=1)

    # Sort only the selected top-k candidates.
    order = np.argsort(-candidate_scores, axis=1)

    top_scores = np.take_along_axis(candidate_scores, order, axis=1)
    top_indices = np.take_along_axis(candidate_indices, order, axis=1)

    return top_scores.astype(np.float32), top_indices.astype(np.int64)
