import argparse
import csv
import json
from pathlib import Path
from typing import Any
from datetime import datetime


TIMING_METRICS = [
    "mpi_total_time_sec",
    "merge_time_sec",
    "max_local_search_time_sec",
    "mean_local_search_time_sec",
    "max_communication_time_sec",
    "mean_communication_time_sec",
    "throughput_queries_per_sec",
]


SUMMARY_FIELDS = ["count", "mean", "median", "min", "max", "std"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect benchmark_metrics.json files into a flat CSV table."
    )
    parser.add_argument(
        "--results-root",
        default="results",
        help="Root directory containing benchmark result folders.",
    )
    parser.add_argument(
        "--output",
        default="analysis/benchmark_results.csv",
        help="Output CSV path.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite the output CSV if it already exists.",
    )
    return parser.parse_args()

def make_output_path_safe(path: Path, overwrite: bool) -> Path:
    """
    Avoid overwriting an existing CSV unless overwrite=True.
    """
    if overwrite or not path.exists():
        return path

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return path.with_name(f"{path.stem}_{timestamp}{path.suffix}")

def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def get_nested(data: dict[str, Any], keys: list[str], default: Any = None) -> Any:
    current: Any = data

    for key in keys:
        if not isinstance(current, dict):
            return default
        if key not in current:
            return default
        current = current[key]

    return current


def infer_output_info(
    metrics_path: Path,
    results_root: Path,
    metrics: dict[str, Any],
) -> dict[str, Any]:
    """
    Support both result layouts:

    Old:
        results/<run_id>/benchmark_metrics.json

    New:
        results/<experiment_id>/<unique_run_folder>/benchmark_metrics.json
    """
    run_dir = metrics_path.parent

    output_info = metrics.get("output", {})
    if output_info:
        return {
            "experiment_id": Path(output_info.get("experiment_dir", "")).name,
            "unique_run_name": output_info.get("unique_run_name"),
            "run_dir": output_info.get("run_dir"),
        }

    run_id = metrics.get("run_id", run_dir.name)

    try:
        relative_parts = run_dir.relative_to(results_root).parts
    except ValueError:
        relative_parts = run_dir.parts

    if len(relative_parts) >= 2:
        experiment_id = relative_parts[0]
        unique_run_name = relative_parts[-1]
    else:
        experiment_id = run_id
        unique_run_name = ""

    return {
        "experiment_id": experiment_id,
        "unique_run_name": unique_run_name,
        "run_dir": str(run_dir),
    }


def flatten_timing_summary(metrics: dict[str, Any]) -> dict[str, Any]:
    row: dict[str, Any] = {}
    timing_summary = metrics.get("timing_summary", {})

    for metric_name in TIMING_METRICS:
        summary = timing_summary.get(metric_name, {})

        for field in SUMMARY_FIELDS:
            column_name = f"{metric_name}_{field}"
            row[column_name] = summary.get(field)

    return row


def build_row(
    metrics_path: Path,
    results_root: Path,
) -> dict[str, Any]:
    metrics = load_json(metrics_path)

    config_path = metrics_path.parent / "config_used.json"
    if config_path.exists():
        config = load_json(config_path)
    else:
        config = {}

    output_info = infer_output_info(
        metrics_path=metrics_path,
        results_root=results_root,
        metrics=metrics,
    )

    runtime_metadata = metrics.get("runtime_metadata", {})
    slurm = runtime_metadata.get("slurm", {})

    row: dict[str, Any] = {
        # File/result location
        "experiment_id": output_info["experiment_id"],
        "unique_run_name": output_info["unique_run_name"],
        "run_dir": output_info["run_dir"],
        "metrics_path": str(metrics_path),

        # Runtime metadata
        "created_at": runtime_metadata.get("created_at"),
        "hostname": runtime_metadata.get("hostname"),
        "config_path": runtime_metadata.get("config_path"),

        # Slurm metadata, empty for local runs or older results
        "slurm_job_id": slurm.get("SLURM_JOB_ID"),
        "slurm_job_name": slurm.get("SLURM_JOB_NAME"),
        "slurm_partition": slurm.get("SLURM_JOB_PARTITION"),
        "slurm_nodes": slurm.get("SLURM_JOB_NUM_NODES"),
        "slurm_ntasks": slurm.get("SLURM_NTASKS"),
        "slurm_ntasks_per_node": slurm.get("SLURM_NTASKS_PER_NODE"),
        "slurm_cpus_per_task": slurm.get("SLURM_CPUS_PER_TASK"),
        "slurm_nodelist": slurm.get("SLURM_JOB_NODELIST"),

        # Main benchmark metadata
        "benchmark_mode": metrics.get("benchmark_mode"),
        "run_id": metrics.get("run_id"),
        "retrieval_mode": metrics.get("retrieval_mode"),
        "search_backend": metrics.get("search_backend"),
        "search_index_reuse": metrics.get("search_index_reuse"),
        "faiss_num_threads": metrics.get("faiss_num_threads"),
        "loading_strategy": metrics.get("loading_strategy"),
        "vector_storage": metrics.get("vector_storage"),
        "query_loading": metrics.get("query_loading"),

        # Workload
        "world_size": metrics.get("world_size"),
        "num_vectors": metrics.get("num_vectors"),
        "num_queries": metrics.get("num_queries"),
        "dimension": metrics.get("dimension"),
        "top_k": metrics.get("top_k"),

        # Benchmark configuration
        "warmup_runs": metrics.get("warmup_runs"),
        "measurement_runs": metrics.get("measurement_runs"),

        # Initial setup timings
        "initial_load_time_sec_max": metrics.get("initial_load_time_sec_max"),
        "initial_load_time_sec_mean": metrics.get("initial_load_time_sec_mean"),
        "initial_index_build_time_sec_max": metrics.get(
            "initial_index_build_time_sec_max"
        ),
        "initial_index_build_time_sec_mean": metrics.get(
            "initial_index_build_time_sec_mean"
        ),

        # Config-derived fields, useful when older metrics are missing metadata
        "config_project_run_id": get_nested(config, ["project", "run_id"]),
        "config_dataset_name": get_nested(config, ["dataset", "name"]),
        "config_dataset_seed": get_nested(config, ["dataset", "seed"]),
        "config_dataset_normalize": get_nested(config, ["dataset", "normalize"]),
    }

    row.update(flatten_timing_summary(metrics))

    return row


def sort_key(row: dict[str, Any]) -> tuple:
    return (
        str(row.get("experiment_id") or ""),
        int(row.get("num_vectors") or 0),
        int(row.get("num_queries") or 0),
        int(row.get("top_k") or 0),
        str(row.get("search_backend") or ""),
        str(row.get("retrieval_mode") or ""),
        int(row.get("world_size") or 0),
        str(row.get("created_at") or ""),
    )


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    if not rows:
        raise ValueError("No rows to write.")

    fieldnames: list[str] = []

    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()

    results_root = Path(args.results_root)
    output_path = make_output_path_safe(
        path=Path(args.output),
        overwrite=args.overwrite,
    )

    metrics_files = sorted(results_root.rglob("benchmark_metrics.json"))

    if not metrics_files:
        raise FileNotFoundError(
            f"No benchmark_metrics.json files found under {results_root}"
        )

    rows = [
        build_row(
            metrics_path=metrics_path,
            results_root=results_root,
        )
        for metrics_path in metrics_files
    ]

    rows.sort(key=sort_key)

    write_csv(output_path, rows)

    print(f"Collected benchmark runs: {len(rows)}")
    print(f"Wrote CSV to: {output_path}")


if __name__ == "__main__":
    main()
