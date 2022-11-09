#%%
import yaml
import pandas as pd
import pickle
from pathlib import Path
import numpy as np
import os, sys

with open('config.yaml')  as f:
    config = yaml.load(f, yaml.SafeLoader)
os.sys.path.append(config['path_config']['project_path'])

from src.MyModule.utils import *

table_name = 'clrc_pt_bsnf'
file_name = config['file_name'][table_name]
input_path = Path(config['path_config']['input_path'])
output_path = Path(config['path_config']['output_path'])
columns = config['data_config']['required'][table_name.upper()]
prefix = config['data_config']['prefix'][table_name.upper()]

if not output_path.joinpath('0_preprocess').exists():
    output_path.joinpath('0_preprocess').mkdir(parents=True)

output_path = output_path.joinpath('0_preprocess')

#%%
bsnf = read_file(input_path, file_name)

#%% 
bsnf_required = bsnf[columns.keys()]

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

bsnf_required = bsnf_required.rename(columns ={'BSPT_FRST_DIAG_YMD':"TIME"}).set_index(['PT_SBST_NO','TIME'])

#%%
bsnf_required = bsnf_required.reset_index()
date_data = date_data.reset_index()

#%%
bsnf_final = bsnf_required.merge(date_data)

duplicated_counts = bsnf_final[['PT_SBST_NO','TIME']].duplicated().sum()
print(f'duplicated_counts : {duplicated_counts}')


#%%
bsnf_final = bsnf_final.set_index(['PT_SBST_NO','TIME'])


#%%
bsnf_final = bsnf_final.add_prefix(prefix)

bsnf_final = remove_invalid_values(bsnf_final)
#%%

#%%
bsnf_final.to_pickle(output_path.joinpath('clrc_pt_bsnf.pkl'))