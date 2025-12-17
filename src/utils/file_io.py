import os
import json
from omegaconf import DictConfig


def _resolve_path(path: str, config: DictConfig) -> str:
    """
    Resolve a path relative to the project root unless it's already absolute.
    """
    if os.path.isabs(path):
        return path
    return os.path.join(config["project"]["project_path"], path)


def create_dir(dir_path: str, config: DictConfig) -> str:
    """
    Create a directory, treating `dir_path` as either absolute or relative to the project root.
    """
    full_dir = _resolve_path(dir_path, config)
    os.makedirs(full_dir, exist_ok=True)
    return f"{full_dir} is created"


def save_json(data, output_path: str, file_name: str, config: DictConfig) -> None:
    """
    Save `data` as JSON to `output_path/file_name.json`.
    `output_path` can be absolute or relative to the project root.
    """
    base_dir = _resolve_path(output_path, config)
    os.makedirs(base_dir, exist_ok=True)
    file_path = os.path.join(base_dir, f"{file_name}.json")
    with open(file_path, "w") as f:
        json.dump(data, f, indent=1)