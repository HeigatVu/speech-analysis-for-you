import os
import json
from omegaconf import DictConfig


def __resolve_path(path: str, config: DictConfig) -> str:
    """ Consisting all path to absolute path
    Input:
        path: absolute or relative file path
        config: configuration to take project_path
    return:
        String of absolute path
    """
    if os.path.isabs(path):
        return path
    return os.path.join(config.project.project_path, path)


def create_dir(dir_path: str, config: DictConfig) -> str:
    """ Create directory
    Input:
        dir_path: absolute or relative path
        config: configuration to take project_path
    return:
        string to notify that directory created
    """
    full_dir = __resolve_path(dir_path, config)
    os.makedirs(full_dir, exist_ok=True)
    return f"{full_dir} is created"


def save_json(data, output_path: str, file_name: str, config: DictConfig) -> None:
    """ Create json file
    Input:
        data: data will save into json file
        output_path: place for saving json file
        file_name: name of json file
        config: configuration to take project_path
    Return:
        strong to notify that json file created
    """
    base_dir = __resolve_path(output_path, config)
    os.makedirs(base_dir, exist_ok=True)
    file_path = os.path.join(base_dir, f"{file_name}.json")
    with open(file_path, "w") as f:
        json.dump(data, f, indent=1)
    return f"{file_name}.json is created"

def load_json(json_path: str, config: DictConfig) -> None:
    """ Load json file
    Input:
        file_path: path of json file
    Return:
        data in json
    """
    full_dir_path_json = __resolve_path(json_path, config)
    with open(full_dir_path_json, 'r') as f:
        data = json.load(f)

    return data