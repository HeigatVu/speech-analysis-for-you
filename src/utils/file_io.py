import os
import json
from omegaconf import DictConfig

# create directory from project dir
def create_dir(dir_path:str, config:DictConfig)->str:
    os.makedirs(os.path.join(config["project"]["project_path"], dir_path), exist_ok=True)
    return f"{dir_path} is created"

# Save json
def save_json(input, output_path:str, file_name:str, config:DictConfig) -> None:
    with open(f"{config["project"]["project_path"]}/{output_path}/{file_name}.json", 'w') as f:
        json.dump(input, f, indent=1)