import json
from pathlib import Path
from typing import Any

import numpy as np

from dense_retrieval.paths import get_results_root


def get_run_dir(config: dict[str, Any]) -> Path:
    """
    Return the result directory for the current run_id.
    """
    run_id = config["project"]["run_id"]
    return get_results_root(config) / run_id


def make_json_serializable(value: Any) -> Any:
    """
    Convert NumPy values into plain Python values so they can be written as JSON.
    """
    if isinstance(value, np.ndarray):
        return value.tolist()

    if isinstance(value, np.integer):
        return int(value)

    if isinstance(value, np.floating):
        return float(value)

    if isinstance(value, dict):
        return {key: make_json_serializable(item) for key, item in value.items()}

    if isinstance(value, list):
        return [make_json_serializable(item) for item in value]

    if isinstance(value, tuple):
        return [make_json_serializable(item) for item in value]

    return value


def write_json(path: Path, data: Any) -> None:
    """
    Write data to a JSON file.
    """
    with path.open("w", encoding="utf-8") as f:
        json.dump(make_json_serializable(data), f, indent=2)


def build_topk_preview(
    result: dict[str, Any],
    num_preview_queries: int = 1,
) -> dict[str, Any]:
    """
    Build a small JSON-friendly preview of the top-k output.

    This avoids writing all scores and indices for larger benchmark runs.
    """
    scores = result["scores"]
    indices = result["indices"]

    num_queries = min(num_preview_queries, scores.shape[0])

    preview_queries = []

    for query_id in range(num_queries):
        query_results = []

        for position, (vector_id, score) in enumerate(
            zip(indices[query_id], scores[query_id]),
            start=1,
        ):
            query_results.append(
                {
                    "rank": position,
                    "vector_id": int(vector_id),
                    "score": float(score),
                }
            )

        preview_queries.append(
            {
                "query_id": query_id,
                "topk": query_results,
            }
        )

    return {
        "num_preview_queries": num_queries,
        "queries": preview_queries,
    }


def _format_optional_seconds(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.6f} s"


def format_result_summary(result: dict[str, Any]) -> str:
    """
    Create a human-readable text summary of one retrieval run.
    """
    metrics = result["metrics"]
    scores = result["scores"]
    indices = result["indices"]

    lines = []

    lines.append("Retrieval finished.")
    lines.append("")
    lines.append("Run summary:")

    if "retrieval_mode" in metrics:
        lines.append(f"  Retrieval mode:        {metrics['retrieval_mode']}")

    if "search_backend" in metrics:
        lines.append(f"  Search backend:        {metrics['search_backend']}")

    if metrics.get("search_backend") == "faiss":
        lines.append(f"  FAISS threads/rank:    {metrics.get('faiss_num_threads')}")

    lines.append(f"  MPI ranks:             {metrics['world_size']}")
    lines.append(f"  Vectors:               {metrics['num_vectors']}")
    lines.append(f"  Queries:               {metrics['num_queries']}")
    lines.append(f"  Dimension:             {metrics['dimension']}")
    lines.append(f"  Top-k:                 {metrics['top_k']}")

    lines.append("")
    lines.append("Timing summary:")
    lines.append(
        f"  Load time max:         "
        f"{_format_optional_seconds(metrics.get('load_time_sec_max'))}"
    )
    lines.append(
        f"  Load time mean:        "
        f"{_format_optional_seconds(metrics.get('load_time_sec_mean'))}"
    )
    lines.append(
        f"  MPI retrieval time:    "
        f"{_format_optional_seconds(metrics.get('mpi_total_time_sec'))}"
    )
    lines.append(
        f"  Merge time:            "
        f"{_format_optional_seconds(metrics.get('merge_time_sec'))}"
    )
    lines.append(
        f"  Total time approx.:    "
        f"{_format_optional_seconds(metrics.get('total_time_sec'))}"
    )

    lines.append("")
    lines.append("Shard distribution:")

    for info in metrics["rank_info"]:
        line = (
            f"  Rank {info['rank']:>2}: "
            f"[{info['shard_start']}, {info['shard_end']}) "
            f"({info['num_local_vectors']} vectors), "
            f"load={_format_optional_seconds(info.get('load_time_sec'))}, "
            f"search={info['local_search_time_sec']:.6f}s, "
            f"comm={info['communication_time_sec']:.6f}s"
        )

        if "local_merge_time_sec" in info:
            line += f", local_merge={info['local_merge_time_sec']:.6f}s"

        lines.append(line)

    lines.append("")
    lines.append("Top-k preview for first query:")

    first_query_indices = indices[0].tolist()
    first_query_scores = scores[0].tolist()

    for position, (vector_id, score) in enumerate(
        zip(first_query_indices, first_query_scores),
        start=1,
    ):
        lines.append(f"  {position:>2}. vector_id={vector_id:>8}, score={score:.6f}")

    return "\n".join(lines)


def save_run_outputs(
    config: dict[str, Any],
    result: dict[str, Any],
) -> dict[str, Path]:
    """
    Save result files for one retrieval run.

    Output structure:

        results/<run_id>/
        ├── result_summary.txt
        ├── metrics.json
        ├── topk_preview.json
        └── config_used.json
    """
    run_dir = get_run_dir(config)
    run_dir.mkdir(parents=True, exist_ok=True)

    summary_path = run_dir / "result_summary.txt"
    metrics_path = run_dir / "metrics.json"
    topk_preview_path = run_dir / "topk_preview.json"
    config_path = run_dir / "config_used.json"

    summary_text = format_result_summary(result)

    with summary_path.open("w", encoding="utf-8") as f:
        f.write(summary_text)
        f.write("\n")

    write_json(metrics_path, result["metrics"])
    write_json(topk_preview_path, build_topk_preview(result))
    write_json(config_path, config)

    return {
        "summary": summary_path,
        "metrics": metrics_path,
        "topk_preview": topk_preview_path,
        "config": config_path,
    }
