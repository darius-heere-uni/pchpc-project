import argparse
import statistics
from typing import Any

from mpi4py import MPI

from dense_retrieval.config import load_config
from dense_retrieval.data.loading import load_dataset_shard
from dense_retrieval.paths import get_dataset_dir
from dense_retrieval.results import (
    collect_runtime_metadata,
    get_benchmark_run_dir,
    write_json,
)
from dense_retrieval.retrieval.mpi_centralized import run_mpi_centralized_retrieval
from dense_retrieval.retrieval.mpi_tree import run_mpi_tree_retrieval
from dense_retrieval.retrieval.mpi_tree_explicit import run_mpi_tree_explicit_retrieval
from dense_retrieval.search.local_index import LocalSearchIndex


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run repeated hot-style retrieval benchmarks with MPI."
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to the JSON config file, e.g. configs/local_benchmark.json",
    )
    return parser.parse_args()


def run_selected_retrieval(
    mode: str,
    local_vectors,
    queries,
    top_k: int,
    comm: MPI.Comm,
    shard_start_idx: int,
    num_global_vectors: int,
    search_backend: str,
    faiss_num_threads: int | None,
    search_index: LocalSearchIndex | None = None,
):
    if mode == "mpi_centralized":
        return run_mpi_centralized_retrieval(
            local_vectors=local_vectors,
            queries=queries,
            top_k=top_k,
            comm=comm,
            shard_start_idx=shard_start_idx,
            num_global_vectors=num_global_vectors,
            load_time_sec=None,
            search_backend=search_backend,
            faiss_num_threads=faiss_num_threads,
            search_index=search_index,
        )

    if mode == "mpi_tree":
        return run_mpi_tree_retrieval(
            local_vectors=local_vectors,
            queries=queries,
            top_k=top_k,
            comm=comm,
            shard_start_idx=shard_start_idx,
            num_global_vectors=num_global_vectors,
            load_time_sec=None,
            search_backend=search_backend,
            faiss_num_threads=faiss_num_threads,
            search_index=search_index,
        )

    if mode == "mpi_tree_explicit":
        return run_mpi_tree_explicit_retrieval(
            local_vectors=local_vectors,
            queries=queries,
            top_k=top_k,
            comm=comm,
            shard_start_idx=shard_start_idx,
            num_global_vectors=num_global_vectors,
            load_time_sec=None,
            search_backend=search_backend,
            faiss_num_threads=faiss_num_threads,
            search_index=search_index,
        )

    raise ValueError(f"Unknown retrieval mode: {mode}")


def summarize(values: list[float]) -> dict[str, float | int]:
    if not values:
        raise ValueError("Cannot summarize an empty list")

    if len(values) == 1:
        std_value = 0.0
    else:
        std_value = statistics.stdev(values)

    return {
        "count": len(values),
        "mean": statistics.mean(values),
        "median": statistics.median(values),
        "min": min(values),
        "max": max(values),
        "std": std_value,
    }


def extract_compact_run_metrics(
    run_index: int,
    metrics: dict[str, Any],
) -> dict[str, Any]:
    rank_info = metrics["rank_info"]

    local_search_times = [
        info["local_search_time_sec"]
        for info in rank_info
    ]
    communication_times = [
        info["communication_time_sec"]
        for info in rank_info
    ]

    local_merge_times = [
        info.get("local_merge_time_sec", 0.0)
        for info in rank_info
    ]

    num_queries = metrics["num_queries"]
    mpi_total_time_sec = metrics["mpi_total_time_sec"]

    if mpi_total_time_sec > 0:
        throughput_queries_per_sec = num_queries / mpi_total_time_sec
    else:
        throughput_queries_per_sec = None

    return {
        "run_index": run_index,
        "mpi_total_time_sec": mpi_total_time_sec,
        "merge_time_sec": metrics["merge_time_sec"],
        "throughput_queries_per_sec": throughput_queries_per_sec,
        "max_local_search_time_sec": max(local_search_times),
        "mean_local_search_time_sec": statistics.mean(local_search_times),
        "max_communication_time_sec": max(communication_times),
        "mean_communication_time_sec": statistics.mean(communication_times),
        "max_local_merge_time_sec": max(local_merge_times),
        "mean_local_merge_time_sec": statistics.mean(local_merge_times),
        "rank_info": rank_info,
    }


def build_benchmark_summary(
    config: dict[str, Any],
    benchmark_metrics: dict[str, Any],
) -> str:
    lines = []

    lines.append("Benchmark finished.")
    lines.append("")
    lines.append("Benchmark setup:")
    lines.append(f"  Run ID:             {config['project']['run_id']}")
    lines.append(f"  Retrieval mode:     {benchmark_metrics['retrieval_mode']}")
    lines.append(f"  Search backend:     {benchmark_metrics['search_backend']}")
    lines.append(f"  MPI ranks:          {benchmark_metrics['world_size']}")
    lines.append(f"  Vectors:            {benchmark_metrics['num_vectors']}")
    lines.append(f"  Queries:            {benchmark_metrics['num_queries']}")
    lines.append(f"  Dimension:          {benchmark_metrics['dimension']}")
    lines.append(f"  Top-k:              {benchmark_metrics['top_k']}")
    lines.append(f"  Warmup runs:        {benchmark_metrics['warmup_runs']}")
    lines.append(f"  Measurement runs:   {benchmark_metrics['measurement_runs']}")

    if benchmark_metrics["search_backend"] == "faiss":
        lines.append(
            f"  FAISS threads/rank: {benchmark_metrics['faiss_num_threads']}"
        )

    lines.append("")
    lines.append("Initial loading:")
    lines.append(
        f"  Load time max:      "
        f"{benchmark_metrics['initial_load_time_sec_max']:.6f} s"
    )
    lines.append(
        f"  Load time mean:     "
        f"{benchmark_metrics['initial_load_time_sec_mean']:.6f} s"
    )

    lines.append("")
    lines.append("Initial local index construction:")
    lines.append(
        f"  Index build max:    "
        f"{benchmark_metrics['initial_index_build_time_sec_max']:.6f} s"
    )
    lines.append(
        f"  Index build mean:   "
        f"{benchmark_metrics['initial_index_build_time_sec_mean']:.6f} s"
    )
    lines.append(
        f"  Index reuse:        "
        f"{benchmark_metrics['search_index_reuse']}"
    )

    lines.append("")
    lines.append("Measured retrieval timings:")
    for metric_name, label in [
        ("mpi_total_time_sec", "MPI retrieval time"),
        ("merge_time_sec", "Merge time"),
        ("max_local_search_time_sec", "Max local search time"),
        ("max_communication_time_sec", "Max communication time"),
        ("throughput_queries_per_sec", "Throughput queries/sec"),
    ]:
        summary = benchmark_metrics["timing_summary"][metric_name]
        lines.append(f"  {label}:")
        lines.append(f"    mean:   {summary['mean']:.6f}")
        lines.append(f"    median: {summary['median']:.6f}")
        lines.append(f"    min:    {summary['min']:.6f}")
        lines.append(f"    max:    {summary['max']:.6f}")
        lines.append(f"    std:    {summary['std']:.6f}")

    if benchmark_metrics["search_backend"] == "faiss":
        lines.append("")
        lines.append("Note:")
        lines.append(
            "  In benchmark mode, the local search index is built once per rank "
            "and reused across warmup and measured retrieval runs."
        )

    return "\n".join(lines)


def save_benchmark_outputs(
    config: dict[str, Any],
    benchmark_metrics: dict[str, Any],
    measured_runs: list[dict[str, Any]],
) -> dict[str, str]:
    run_dir = get_benchmark_run_dir(
        config=config,
        world_size=benchmark_metrics["world_size"],
    )
    run_dir.mkdir(parents=True, exist_ok=True)

    benchmark_metrics["output"] = {
        "run_dir": str(run_dir),
        "experiment_dir": str(run_dir.parent),
        "unique_run_name": run_dir.name,
    }

    summary_path = run_dir / "benchmark_summary.txt"
    metrics_path = run_dir / "benchmark_metrics.json"
    runs_path = run_dir / "benchmark_runs.json"
    config_path = run_dir / "config_used.json"

    summary_text = build_benchmark_summary(
        config=config,
        benchmark_metrics=benchmark_metrics,
    )

    with summary_path.open("w", encoding="utf-8") as f:
        f.write(summary_text)
        f.write("\n")

    write_json(metrics_path, benchmark_metrics)
    write_json(runs_path, measured_runs)
    write_json(config_path, config)

    return {
        "summary": str(summary_path),
        "metrics": str(metrics_path),
        "runs": str(runs_path),
        "config": str(config_path),
    }


def main() -> None:
    args = parse_args()

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    world_size = comm.Get_size()

    config = load_config(args.config)
    dataset_dir = get_dataset_dir(config)

    search_cfg = config["search"]
    retrieval_cfg = config["retrieval"]
    benchmark_cfg = config.get("benchmark", {})

    warmup_runs = benchmark_cfg.get("warmup_runs", 1)
    measurement_runs = benchmark_cfg.get("measurement_runs", 5)

    similarity = search_cfg.get("similarity", "dot")
    if similarity != "dot":
        raise ValueError(
            f"Only similarity='dot' is implemented for now, got: {similarity}"
        )

    search_backend = search_cfg["backend"]
    faiss_num_threads = search_cfg.get("faiss_num_threads")
    top_k = search_cfg["top_k"]
    retrieval_mode = retrieval_cfg["mode"]

    if rank == 0:
        print(f"Benchmark run_id: {config['project']['run_id']}")
        print(f"Loading dataset from: {dataset_dir}")
        print("Loading strategy: shard-aware loading from shared vectors.npy")
        print(f"Retrieval mode: {retrieval_mode}")
        print(f"Search backend: {search_backend}")
        print(f"Warmup runs: {warmup_runs}")
        print(f"Measurement runs: {measurement_runs}")

    load_start = MPI.Wtime()

    (
        local_vectors,
        queries,
        metadata,
        shard_start,
        shard_end,
        num_global_vectors,
    ) = load_dataset_shard(
        dataset_dir=dataset_dir,
        rank=rank,
        world_size=world_size,
    )

    load_end = MPI.Wtime()
    load_time_sec = load_end - load_start

    load_info = {
        "rank": rank,
        "shard_start": shard_start,
        "shard_end": shard_end,
        "num_local_vectors": local_vectors.shape[0],
        "load_time_sec": load_time_sec,
    }

    gathered_load_info = comm.gather(load_info, root=0)

    if rank == 0:
        print("Dataset shard loaded.")
        print(f"Global vectors: {num_global_vectors}")
        print(f"Queries: {queries.shape[0]}")
        print(f"Dimension: {queries.shape[1]}")

    if rank == 0:
        print("Building reusable local search index on each rank.")

    index_build_start = MPI.Wtime()

    local_search_index = LocalSearchIndex(
        vectors=local_vectors,
        backend=search_backend,
        faiss_num_threads=faiss_num_threads,
    )

    index_build_end = MPI.Wtime()
    index_build_time_sec = index_build_end - index_build_start

    index_build_info = {
        "rank": rank,
        "index_build_time_sec": index_build_time_sec,
    }

    gathered_index_build_info = comm.gather(index_build_info, root=0)

    if rank == 0:
        print("Reusable local search index built.")

    for warmup_index in range(warmup_runs):
        if rank == 0:
            print(f"Warmup run {warmup_index + 1}/{warmup_runs}")

        _ = run_selected_retrieval(
            mode=retrieval_mode,
            local_vectors=local_vectors,
            queries=queries,
            top_k=top_k,
            comm=comm,
            shard_start_idx=shard_start,
            num_global_vectors=num_global_vectors,
            search_backend=search_backend,
            faiss_num_threads=faiss_num_threads,
            search_index=local_search_index,
        )

    measured_runs: list[dict[str, Any]] = []

    for run_index in range(measurement_runs):
        if rank == 0:
            print(f"Measurement run {run_index + 1}/{measurement_runs}")

        result = run_selected_retrieval(
            mode=retrieval_mode,
            local_vectors=local_vectors,
            queries=queries,
            top_k=top_k,
            comm=comm,
            shard_start_idx=shard_start,
            num_global_vectors=num_global_vectors,
            search_backend=search_backend,
            faiss_num_threads=faiss_num_threads,
            search_index=local_search_index,
        )

        if rank == 0:
            compact_metrics = extract_compact_run_metrics(
                run_index=run_index,
                metrics=result["metrics"],
            )
            measured_runs.append(compact_metrics)

    if rank == 0:
        load_times = [
            info["load_time_sec"]
            for info in gathered_load_info
        ]

        index_build_times = [
            info["index_build_time_sec"]
            for info in gathered_index_build_info
        ]

        timing_summary = {
            "mpi_total_time_sec": summarize(
                [run["mpi_total_time_sec"] for run in measured_runs]
            ),
            "merge_time_sec": summarize(
                [run["merge_time_sec"] for run in measured_runs]
            ),
            "max_local_search_time_sec": summarize(
                [run["max_local_search_time_sec"] for run in measured_runs]
            ),
            "mean_local_search_time_sec": summarize(
                [run["mean_local_search_time_sec"] for run in measured_runs]
            ),
            "max_communication_time_sec": summarize(
                [run["max_communication_time_sec"] for run in measured_runs]
            ),
            "mean_communication_time_sec": summarize(
                [run["mean_communication_time_sec"] for run in measured_runs]
            ),
            "throughput_queries_per_sec": summarize(
                [
                    run["throughput_queries_per_sec"]
                    for run in measured_runs
                    if run["throughput_queries_per_sec"] is not None
                ]
            ),
        }

        benchmark_metrics = {
            "benchmark_mode": "hot_repeated_retrieval",
            "run_id": config["project"]["run_id"],
            "runtime_metadata": collect_runtime_metadata(args.config),
            "retrieval_mode": retrieval_mode,
            "search_backend": search_backend,
            "faiss_num_threads": faiss_num_threads,
            "loading_strategy": "shard_aware_shared_npy",
            "vector_storage": "single_vectors_npy",
            "query_loading": "full_queries_on_each_rank",
            "world_size": world_size,
            "num_vectors": num_global_vectors,
            "num_queries": queries.shape[0],
            "dimension": queries.shape[1],
            "top_k": top_k,
            "warmup_runs": warmup_runs,
            "measurement_runs": measurement_runs,
            "initial_load_time_sec_max": max(load_times),
            "initial_load_time_sec_mean": statistics.mean(load_times),
            "rank_load_info": gathered_load_info,
            "search_index_reuse": True,
            "initial_index_build_time_sec_max": max(index_build_times),
            "initial_index_build_time_sec_mean": statistics.mean(index_build_times),
            "rank_index_build_info": gathered_index_build_info,
            "dataset_metadata": metadata,
            "timing_summary": timing_summary,
        }

        saved_paths = save_benchmark_outputs(
            config=config,
            benchmark_metrics=benchmark_metrics,
            measured_runs=measured_runs,
        )

        print()
        print(build_benchmark_summary(config, benchmark_metrics))
        print()
        print("Saved benchmark files:")
        for name, path in saved_paths.items():
            print(f"  {name}: {path}")


if __name__ == "__main__":
    main()
