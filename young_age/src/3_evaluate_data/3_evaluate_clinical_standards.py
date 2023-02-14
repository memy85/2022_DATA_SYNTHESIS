import os
import sys
from pathlib import Path

# project_path = Path(__file__).absolute().parents[2]
project_path = Path().cwd()

os.sys.path.append(project_path.as_posix())

from src.MyModule.utils import *

print(f" this is project path : {project_path} ")

#%% settings

age = 50


#%% path settings

config = load_config()

input_path = get_path("data/processed/2_produce_data/synthetic_decoded")

synthetic_path = input_path.joinpath(f"Synthetic_data_epsilon10000_{age}.csv")

synthetic_data_path_list = [input_path.joinpath(
    f"Synthetic_data_epsilon{eps}_{age}.csv") for eps in config['epsilon']]

train_ori_path = get_path(f"data/processed/preprocess_1/train_ori_{age}.pkl")

testset_path = get_path(f"data/processed/preprocess_1/test_{age}.pkl")

output_path = get_path("data/processed/3_evaluate_data/")

if not output_path.exists():
    output_path.mkdir(parents=True)

#%%

class Tester:

    def __init__(self, test_criteria, information) :
        self.test_criteria
        self.information = information

    def do_test(self) :

