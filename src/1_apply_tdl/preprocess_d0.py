#%%
import torch
import torch.nn as nn

import multiprocessing
import pandas as pd
import pickle
import argparse
from pathlib import Path
import os, sys
from sklearn.preprocessing import LabelEncoder

PROJ_PATH = Path(__file__).parents[2]
sys.path.append(PROJ_PATH.joinpath('src').as_posix())

from MyModule.utils import *
import random

config = load_config()
#%%



#%%

PROJ_PATH = Path(config['path_config']['project_path'])
INPUT_PATH = PROJ_PATH.joinpath('data/processed/0_preprocess')
OUTPUT_PATH = PROJ_PATH.joinpath('data/processed/1_apply_tdl/preprocess_d0')

#%%

if not OUTPUT_PATH.exists():
    OUTPUT_PATH.mkdir(parents=True)

#%%
d0 = read_file(INPUT_PATH, 'D0.pkl')
# %%

max_count = d0.groupby('PT_SBST_NO').size().max()
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