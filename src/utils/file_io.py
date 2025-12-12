import os
import json
import hydra
from omegaconf import DictConfig

# create directory from project dir
@hydra.main(config_path="../config", config_name="main", version_base=None)
def create_dir(dir_path:str, config:DictConfig)->str:
    os.makedir(os.path.join(config["project"]["project_path"], dir_path), exist_ok=True)
    return f"{dir_path} is created"

# Save json
@hydra.main(config_path="../config", config_name="main", version_base=None)
def save_json(input, output_path:str, file_name:str, config:DictConfig) -> None:
    with open(f"{config["project"]["project_path"]}/{output_path}/{file_name}.json", 'w') as f:
        json.dump(input, f, indent=1)