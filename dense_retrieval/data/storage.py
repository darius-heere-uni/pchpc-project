import json
from pathlib import Path
from typing import Any

import numpy as np


def save_dataset(
    dataset_dir: str | Path,
    vectors: np.ndarray,
    queries: np.ndarray,
    metadata: dict[str, Any],
) -> None:
    """
    Save vectors, queries, and metadata to a dataset directory.
    """
    dataset_path = Path(dataset_dir)
    dataset_path.mkdir(parents=True, exist_ok=True)

    np.save(dataset_path / "vectors.npy", vectors)
    np.save(dataset_path / "queries.npy", queries)

    with (dataset_path / "metadata.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)


def load_dataset(
    dataset_dir: str | Path,
    mmap_mode: str | None = None,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """
    Load vectors, queries, and metadata from a dataset directory.
    """
    dataset_path = Path(dataset_dir)

    vectors_path = dataset_path / "vectors.npy"
    queries_path = dataset_path / "queries.npy"
    metadata_path = dataset_path / "metadata.json"

    if not vectors_path.exists():
        raise FileNotFoundError(f"Missing vectors file: {vectors_path}")
    if not queries_path.exists():
        raise FileNotFoundError(f"Missing queries file: {queries_path}")
    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing metadata file: {metadata_path}")

    vectors = np.load(vectors_path, mmap_mode=mmap_mode)
    queries = np.load(queries_path, mmap_mode=mmap_mode)

    with metadata_path.open("r", encoding="utf-8") as f:
        metadata = json.load(f)

    return vectors, queries, metadata
