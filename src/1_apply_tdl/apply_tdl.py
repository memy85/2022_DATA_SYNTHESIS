
#%%
import multiprocessing
import pandas as pd
import pickle
import argparse
from pathlib import Path
import os, sys
from sdv.timeseries import PAR

PROJ_PATH = Path(__file__).parents[2]
sys.path.append(PROJ_PATH.joinpath('src').as_posix())

from MyModule.utils import *
import random

config = load_config()

#%%

PROJ_PATH = Path(config['path_config']['project_path'])
INPUT_PATH = PROJ_PATH.joinpath('data/processed/0_preprocess')
OUTPUT_PATH = PROJ_PATH.joinpath('data/processed/1_apply_tdl/apply_tdl')

#%%

if not OUTPUT_PATH.exists():
    OUTPUT_PATH.mkdir(parents=True)

#%%
D0 = read_file(INPUT_PATH, 'D0.pkl')

#%%
import torch
import torch.nn as nn
import torch.utils.data as data

class CancerDataset(data.Dataset):
    
    def __init__(self, input_path, train, data_idx):
        self.data = read_file(input_path, 'D0.pkl')
        self.train = train 
        self.data_idx = data_idx
        
        max_count = self.data.groupby('PT_SBST_NO').size().max()
        n_pts = d0.PT_SBST_NO.nunique()
        n_cols = len(d0.columns)-1

        shape = torch.zeros(n_pts, max_count, n_cols)

        for idx, pt in enumerate(d0.PT_SBST_NO.unique().tolist()):
            
            td = torch.Tensor(d0[d0.PT_SBST_NO == pt].values[:,1:].astype('float'))
            timestamp = td.shape[0]
            shape[idx,0:timestamp,:] = td

        self.data = shape

        # select train or test
        self.data = torch.select(self.data, 0, self.data_idx)

    def __len__(self):
        
        return len(self.data)
    
    def __getitem__(self, idx):
        
        return self.data[idx]


d0.
#%%
train_idx = 
test_idx = 
#%%

train_data = CancerDataset(INPUT_PATH, )

data.DataLoader(CancerDataset, )
# %%
