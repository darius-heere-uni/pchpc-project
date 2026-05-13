import argparse

from dense_retrieval.config import load_config
from dense_retrieval.paths import get_dataset_dir
from dense_retrieval.data.generation import generate_dataset_arrays
from dense_retrieval.data.storage import save_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a reproducible random dense-vector dataset."
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to the JSON config file, e.g. configs/local.json",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    dataset_cfg = config["dataset"]
    dataset_dir = get_dataset_dir(config)

    print(f"Generating dataset: {dataset_cfg['name']}")
    print(f"Output directory: {dataset_dir}")
    print(f"Number of vectors: {dataset_cfg['num_vectors']}")
    print(f"Number of queries: {dataset_cfg['num_queries']}")
    print(f"Dimension: {dataset_cfg['dimension']}")
    print(f"Dtype: {dataset_cfg['dtype']}")
    print(f"Seed: {dataset_cfg['seed']}")
    print(f"Normalize: {dataset_cfg['normalize']}")

    vectors, queries = generate_dataset_arrays(
        num_vectors=dataset_cfg["num_vectors"],
        num_queries=dataset_cfg["num_queries"],
        dimension=dataset_cfg["dimension"],
        dtype=dataset_cfg["dtype"],
        seed=dataset_cfg["seed"],
        normalize=dataset_cfg["normalize"],
    )

    metadata = {
        "name": dataset_cfg["name"],
        "num_vectors": dataset_cfg["num_vectors"],
        "num_queries": dataset_cfg["num_queries"],
        "dimension": dataset_cfg["dimension"],
        "dtype": dataset_cfg["dtype"],
        "seed": dataset_cfg["seed"],
        "normalize": dataset_cfg["normalize"],
        "vectors_shape": list(vectors.shape),
        "queries_shape": list(queries.shape),
    }

    save_dataset(
        dataset_dir=dataset_dir,
        vectors=vectors,
        queries=queries,
        metadata=metadata,
    )

    print("Dataset generation finished.")
    print(f"Saved vectors to: {dataset_dir / 'vectors.npy'}")
    print(f"Saved queries to: {dataset_dir / 'queries.npy'}")
    print(f"Saved metadata to: {dataset_dir / 'metadata.json'}")


if __name__ == "__main__":
    main()
