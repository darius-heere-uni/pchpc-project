import numpy as np


def merge_topk(
    scores_list: list[np.ndarray],
    indices_list: list[np.ndarray],
    top_k: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Merge several local top-k result lists into one global top-k result.

    Each item in scores_list has shape:
        (num_queries, local_k)

    Each item in indices_list has shape:
        (num_queries, local_k)

    The indices are expected to already be global vector IDs.
    """
    if len(scores_list) != len(indices_list):
        raise ValueError("scores_list and indices_list must have the same length")

    non_empty_scores = []
    non_empty_indices = []

    for scores, indices in zip(scores_list, indices_list):
        if scores.shape != indices.shape:
            raise ValueError(
                f"Score/index shape mismatch: {scores.shape} vs {indices.shape}"
            )

        if scores.shape[1] > 0:
            non_empty_scores.append(scores)
            non_empty_indices.append(indices)

    if not non_empty_scores:
        raise ValueError("Cannot merge empty top-k result lists")

    all_scores = np.concatenate(non_empty_scores, axis=1)
    all_indices = np.concatenate(non_empty_indices, axis=1)

    top_k_eff = min(top_k, all_scores.shape[1])

    candidate_indices = np.argpartition(
        -all_scores,
        kth=top_k_eff - 1,
        axis=1,
    )[:, :top_k_eff]

    candidate_scores = np.take_along_axis(all_scores, candidate_indices, axis=1)
    candidate_global_indices = np.take_along_axis(
        all_indices,
        candidate_indices,
        axis=1,
    )

    order = np.argsort(-candidate_scores, axis=1)

    merged_scores = np.take_along_axis(candidate_scores, order, axis=1)
    merged_indices = np.take_along_axis(candidate_global_indices, order, axis=1)

    return merged_scores.astype(np.float32), merged_indices.astype(np.int64)
