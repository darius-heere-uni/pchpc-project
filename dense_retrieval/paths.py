import os
from pathlib import Path
from typing import Any


def get_project_root() -> Path:
    """
    Return the project root directory.

    This assumes the package folder has the structure:

        pchpc-project/
        ├── dense_retrieval/
        │   └── paths.py
        ├── configs/
        ├── data/
        └── results/
    """
    return Path(__file__).resolve().parents[1]


def resolve_project_path(path_value: str | Path) -> Path:
    """
    Resolve a path from the config.

    Supported forms:
        data
        ./data
        ~/some/path
        $PROJECT_DIR/some/path
        ${PROJECT_DIR}/some/path
        /absolute/path

    Absolute paths stay unchanged.
    Relative paths are interpreted relative to the project root.
    """
    expanded = os.path.expandvars(str(path_value))
    path = Path(expanded).expanduser()

    if path.is_absolute():
        return path

    return get_project_root() / path


def get_data_root(config: dict[str, Any]) -> Path:
    return resolve_project_path(config["paths"]["data_root"])


def get_results_root(config: dict[str, Any]) -> Path:
    return resolve_project_path(config["paths"]["results_root"])


def get_dataset_dir(config: dict[str, Any]) -> Path:
    data_root = get_data_root(config)
    dataset_name = config["dataset"]["name"]
    return data_root / dataset_name
