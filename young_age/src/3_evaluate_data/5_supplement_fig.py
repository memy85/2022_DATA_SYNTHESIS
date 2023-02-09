#%%

import scienceplots
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
from pathlib import Path

from src.MyModule.distribution_comparison import *
from src.MyModule.ml_function import *
from src.MyModule.utils import *

# project_path = Path(__file__).absolute().parents[2]
project_path = Path().cwd()

os.sys.path.append(project_path.as_posix())
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

synthetic_data_list = list(
    map(lambda x: pd.read_csv(x), synthetic_data_path_list))
train_ori = pd.read_pickle(train_ori_path)
test = pd.read_pickle(testset_path)

#%%

test.drop(["PT_SBST_NO"], axis=1, inplace=True)
train_ori.drop(["PT_SBST_NO"], axis=1, inplace=True)
train_ori = train_ori.rename(columns={"RLPS DIFF": "RLPS_DIFF"})
synthetic_data_list = list(map(lambda x: x.drop(
    ["PT_SBST_NO", "Unnamed: 0"], axis=1), synthetic_data_list))

#%%

def test_death_ratio(data):
    value_counts = data["DEAD"].value_counts()
    dead, alive = value_counts[1], value_counts[0]
    return dead, alive 


#%%

count_list = []
train_ratio = test_death_ratio(train_ori)
test_ratio = test_death_ratio(test)

count_list.append({"data_type":"train_ori", "death":train_ratio[0], "alive":train_ratio[1] })
count_list.append({"data_type":"test", "death":test_ratio[0], "alive":test_ratio[1] })

def make_for_all_epsilons(func):
    '''
    func : function for counting variables in a produced dataset
    Input a function
    '''
    pass

for idx, eps in enumerate(config['epsilon']):
    data = synthetic_data_list[idx]
    death, alive = test_death_ratio(data)

    count_list.append(
    {"data_type":"epsilon {}".format(eps),
     "death": death,
     "alive": alive})

#%%

#%%

df = pd.DataFrame(count_list).set_index("data_type")
df["death/alive"] = df.apply(lambda x : x["death"]/x["alive"], axis=1)

#%%
df


#%%

import pandas as pd


