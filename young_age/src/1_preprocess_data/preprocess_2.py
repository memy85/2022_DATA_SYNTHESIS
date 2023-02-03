
'''
This code creates train and test data for evaluating the data
'''

#%%
from pathlib import Path
import os, sys
import argparse

# project_path = Path(__file__).absolute().parents[2]
project_path = Path("/home/wonseok/projects/2022_DATA_SYNTHESIS/young_age")
print(f"this is project_path : {project_path.as_posix()}")
os.sys.path.append(project_path.as_posix())

from src.MyModule.utils import *
#%%
config = load_config()
project_path = Path(config["project_path"])
input_path = get_path("data/preprocess_1")
output_path = get_path("data/processed/preprocess_1")
if not output_path.exists() : 
    output_path.mkdir(parents=True)

import pandas as pd
import numpy as np
import random
import pickle
from sklearn.preprocessing import LabelEncoder

#%%
def argument_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--age", type = int, default = 50)
    args = parser.parse_args()
    return args


#%%
# encoded = pd.read_csv(output_path.joinpath(f'encoded_D0_{args.age}.csv'))
age = 50
train_idx = pd.read_pickle(input_path.joinpath("train_idx.pkl"))
encoded = pd.read_csv(output_path.joinpath(f'encoded_D0_{age}.csv'))
sampled = encoded.sample(frac= 0.8)
train_idx = sampled.index
valid = unmodified_D0.loc[~encoded.index.isin(train_idx)]

#%%
valid["RLPS DIFF"] = (data["RLPS_DIAG_YMD"] - data["BSPT_FRST_DIAG_YMD"]).dt.days
valid["BSPT_IDGN_AGE"]  = data["BSPT_IDGN_AGE"]
valid["DEAD_DIFF"] = (data["BSPT_DEAD_YMD"] - data["BSPT_FRST_DIAG_YMD"]).dt.days
valid["OVR_SURV"] = (data["CENTER_LAST_VST_YMD"]- data["BSPT_FRST_DIAG_YMD"]).dt.days
# valid["OPRT_SURV"] = (data["CENTER_LAST_VST_YMD"]- data["BSPT_FRST_DIAG_YMD"]).dt.days

for i in range(1,9):
    start = pd.to_datetime(data[f'TRTM_CASB_STRT_YMD{i}'], format = "%Y%m%d")
    end = pd.to_datetime(data[f'TRTM_CASB_CSTR_YMD2_{i}'], format = "%Y%m%d")

    monthly_diff = (end-start).dt.days
    start_diff = (start-data["BSPT_FRST_DIAG_YMD"]).dt.days
    valid[f"REGN_TIME_DIFF_{i}"] = monthly_diff
    valid[f"REGN_START_DIFF_{i}"] = start_diff

#%% 

sampled.to_csv(output_path.joinpath(f'encoded_D0_to_syn_{args.age}.csv'), index=False)
valid.to_csv(output_path.joinpath(f'encoded_D0_to_valid_{args.age}.csv'), index=False)





