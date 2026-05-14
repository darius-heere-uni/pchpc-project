import json
from pathlib import Path
from typing import Any


def load_config(config_path: str | Path) -> dict[str, Any]:
    """
    Load a JSON config file.

    Parameters
    ----------
    config_path:
        Path to a JSON config file.

    Returns
    -------
    dict
        Parsed config dictionary.
    """
    path = Path(config_path).expanduser()

    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        config = json.load(f)

    return config