import numpy as np


def generate_random_matrix(
    num_rows: int,
    dimension: int,
    dtype: str,
    seed: int,
    normalize: bool = False,
) -> np.ndarray:
    """
    Generate a reproducible random matrix.

    Each row represents one dense vector.
    """
    rng = np.random.default_rng(seed)

    matrix = rng.standard_normal(
        size=(num_rows, dimension),
        dtype=np.float32,
    )

    if dtype != "float32":
        matrix = matrix.astype(dtype)

    if normalize:
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms[norms == 0.0] = 1.0
        matrix = matrix / norms

    return matrix


def generate_dataset_arrays(
    num_vectors: int,
    num_queries: int,
    dimension: int,
    dtype: str,
    seed: int,
    normalize: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Generate vectors and queries with fixed random seeds.

    The query seed is derived from the vector seed to make both reproducible
    but not identical.
    """
    vectors = generate_random_matrix(
        num_rows=num_vectors,
        dimension=dimension,
        dtype=dtype,
        seed=seed,
        normalize=normalize,
    )

    queries = generate_random_matrix(
        num_rows=num_queries,
        dimension=dimension,
        dtype=dtype,
        seed=seed + 1,
        normalize=normalize,
    )

    return vectors, queries
