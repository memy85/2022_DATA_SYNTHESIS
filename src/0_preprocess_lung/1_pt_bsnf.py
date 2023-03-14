#%%
import yaml
import pandas as pd
import pickle
from pathlib import Path
import numpy as np
import os, sys

# project_path = Path(__file__).parents[2]
project_path = Path().cwd()
os.sys.path.append(project_path.as_posix())

from src.MyModule.utils import *
config = load_config()

table_name = 'lung_pt_bsnf'
file_name = config['lung_file_name'][table_name]
input_path = Path(config['path_config']['input_path'])
output_path = Path(config['path_config']['output_path'])
columns = config['lung_config']['']
prefix = config['lung_config']['prefix'][table_name.upper()]

if not output_path.joinpath('0_preprocess_lung').exists():
    output_path.joinpath('0_preprocess_lung').mkdir(parents=True)

output_path = output_path.joinpath('0_preprocess')

#%%
bsnf = read_file(input_path, file_name)

#%%
bsnf.columns

#%% filter only required column



#%%
bsnf_required = convert_dates(bsnf_required, config,table_name=table_name.upper())

#%%

def convert_dates_to_binary(data : pd.DataFrame, name):
    '''
    set date as index and value as 1 
    '''
    df = data[['PT_SBST_NO', name]].copy().dropna()
    df = df.rename(columns={name: 'TIME'})
    df = df.set_index(['PT_SBST_NO', 'TIME'])
    df[name] = 1
    return df

#%%
from functools import partial
from itertools import repeat

time_cols = [col for col in columns if columns[col] == 'datetime64[ns]']

date_data = pd.concat(list(map(convert_dates_to_binary, repeat(bsnf_required), time_cols))).fillna(0)


#%%

