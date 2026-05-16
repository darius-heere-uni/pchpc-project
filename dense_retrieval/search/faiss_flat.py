import numpy as np


def _as_float32_contiguous(array: np.ndarray) -> np.ndarray:
    """
    FAISS expects float32 C-contiguous arrays.
    """
    if array.dtype != np.float32 or not array.flags["C_CONTIGUOUS"]:
        return np.ascontiguousarray(array, dtype=np.float32)

    return array


def search_topk_faiss(
    vectors: np.ndarray,
    queries: np.ndarray,
    top_k: int,
    num_threads: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Exact top-k search using FAISS IndexFlatIP.

    IndexFlatIP computes inner products, which matches our current NumPy
    dot-product similarity.
    """
    try:
        import faiss
    except ImportError as exc:
        raise ImportError(
            "FAISS is not installed. Install it with: "
            "conda install -c conda-forge faiss-cpu"
        ) from exc

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

    if num_threads is not None:
        faiss.omp_set_num_threads(num_threads)

    top_k_eff = min(top_k, num_vectors)

    vectors_f32 = _as_float32_contiguous(vectors)
    queries_f32 = _as_float32_contiguous(queries)

    dimension = vectors_f32.shape[1]

    index = faiss.IndexFlatIP(dimension)
    index.add(vectors_f32)

    scores, indices = index.search(queries_f32, top_k_eff)

    return scores.astype(np.float32), indices.astype(np.int64)
