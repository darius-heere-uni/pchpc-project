from typing import Literal

import numpy as np

from dense_retrieval.search.numpy_flat import search_topk_numpy


SearchBackend = Literal["numpy", "faiss"]


def _as_float32_contiguous(array: np.ndarray) -> np.ndarray:
    """
    FAISS expects float32 C-contiguous arrays.
    """
    if array.dtype != np.float32 or not array.flags["C_CONTIGUOUS"]:
        return np.ascontiguousarray(array, dtype=np.float32)

    return array


class LocalSearchIndex:
    """
    Reusable local search index for hot retrieval benchmarks.

    For NumPy, this stores the local vector matrix.
    For FAISS, this builds IndexFlatIP once and reuses it for repeated searches.
    """

    def __init__(
        self,
        vectors: np.ndarray,
        backend: SearchBackend,
        faiss_num_threads: int | None = None,
    ) -> None:
        self.backend = backend
        self.faiss_num_threads = faiss_num_threads

        if backend == "numpy":
            self.vectors = vectors
            self.faiss_index = None
            return

        if backend == "faiss":
            try:
                import faiss
            except ImportError as exc:
                raise ImportError(
                    "FAISS is not installed. Install it with: "
                    "conda install -c conda-forge faiss-cpu"
                ) from exc

            if faiss_num_threads is not None:
                faiss.omp_set_num_threads(faiss_num_threads)

            vectors_f32 = _as_float32_contiguous(vectors)
            dimension = vectors_f32.shape[1]

            index = faiss.IndexFlatIP(dimension)
            index.add(vectors_f32)

            self.vectors = None
            self.faiss_index = index
            return

        raise ValueError(f"Unknown search backend: {backend}")

    def search(
        self,
        queries: np.ndarray,
        top_k: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        if self.backend == "numpy":
            return search_topk_numpy(
                vectors=self.vectors,
                queries=queries,
                top_k=top_k,
            )

        if self.backend == "faiss":
            queries_f32 = _as_float32_contiguous(queries)
            scores, indices = self.faiss_index.search(queries_f32, top_k)
            return scores.astype(np.float32), indices.astype(np.int64)

        raise ValueError(f"Unknown search backend: {self.backend}")
