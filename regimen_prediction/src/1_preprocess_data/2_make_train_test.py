
'''
This code creates train and test data for evaluating the data
'''

#%%
from pathlib import Path
import os, sys
import argparse

project_path = Path(__file__).absolute().parents[2]
os.sys.path.append(project_path.as_posix())

from src.MyModule.utils import *

#%%
config = load_config()
project_path = Path(config["project_path"])

import pandas as pd
import numpy as np
import random
import pickle
from sklearn.preprocessing import LabelEncoder

def restore_to_learnable(data, original_data):
    """
    restore's data into a learnable form. Through this function, the output data will be able to go into the machine learning model
    """

    data = data.copy()
    data["RLPS DIFF"] = (original_data["RLPS_DIAG_YMD"] - original_data["BSPT_FRST_DIAG_YMD"]).dt.days
    data["BSPT_IDGN_AGE"] = original_data["BSPT_IDGN_AGE"]
    data["DEAD_DIFF"] = (original_data["BSPT_DEAD_YMD"] - original_data["BSPT_FRST_DIAG_YMD"]).dt.days
    data["OVR_SURV"] = (original_data["CENTER_LAST_VST_YMD"] - original_data["BSPT_FRST_DIAG_YMD"]).dt.days

    for i in range(1,9):
        start = pd.to_datetime(original_data[f'TRTM_CASB_STRT_YMD{i}'], format = "%Y%m%d")
        end = pd.to_datetime(original_data[f'TRTM_CASB_CSTR_YMD2_{i}'], format = "%Y%m%d")

        monthly_diff = (end-start).dt.days
        start_diff = (start-original_data["BSPT_FRST_DIAG_YMD"]).dt.days
        data[f"REGN_TIME_DIFF_{i}"] = monthly_diff
        data[f"REGN_START_DIFF_{i}"] = start_diff

    return data

#%%
def argument_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--age", type = int, default = 50)
    parser.add_argument("--random_seed", type=int, default = 0)
    args = parser.parse_args()
    return args

#%%

def main():
    args = argument_parse() 
    age = args.age
    random_seed = args.random_seed

    input_path = get_path(f"data/processed/seed{random_seed}/1_preprocess")
    output_path = get_path(f"data/processed/seed{random_seed}/1_preprocess")

    if not output_path.exists() : 
        output_path.mkdir(parents=True)

    original = pd.read_pickle(input_path.joinpath(f"original_{age}.pkl"))
    unmodified_D0 = pd.read_pickle(input_path.joinpath(f"unmodified_D0_{age}.pkl"))
    train_idx = pd.read_pickle(input_path.joinpath(f"train_idx_{age}.pkl"))

    #%%
    encoded = pd.read_csv(output_path.joinpath(f'encoded_D0_{age}.csv'))

    #%%
    train = unmodified_D0.loc[encoded.index.isin(train_idx)].copy()
    test = unmodified_D0.loc[~encoded.index.isin(train_idx)].copy()

    train = restore_to_learnable(train, original)
    test = restore_to_learnable(test, original)

    #%%

    train.to_pickle(output_path.joinpath(f"train_ori_{age}.pkl"))
    test.to_pickle(output_path.joinpath(f"test_{age}.pkl"))

    #%% 

if __name__ == "__main__":
    main()



#%% test the above codes
#age = 50

#original = pd.read_pickle(input_path.joinpath(f"original_{age}.pkl"))
#unmodified_D0 = pd.read_pickle(input_path.joinpath(f"unmodified_D0_{age}.pkl"))
#train_idx = pd.read_pickle(input_path.joinpath(f"train_idx_{age}.pkl"))

##%%
#encoded = pd.read_csv(output_path.joinpath(f'encoded_D0_{age}.csv'))

##%%
#train = unmodified_D0.loc[encoded.index.isin(train_idx)].copy()
#test = unmodified_D0.loc[~encoded.index.isin(train_idx)].copy()

#train = restore_to_learnable(train, original)
#test = restore_to_learnable(test, original)

