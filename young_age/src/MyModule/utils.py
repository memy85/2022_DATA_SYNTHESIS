from pathlib import Path
import os, sys
import yaml

def load_config():
    utils_path = Path(__file__).absolute().parents[2]

    with open(utils_path.joinpath("config/config.yaml")) as f:
        config = yaml.load(f, yaml.SafeLoader)

    return config


def get_path(path):
    '''
    insert a path to the data
    '''
    config = load_config()

    project_path = Path(config["project_path"])
    return project_path.joinpath(path)
