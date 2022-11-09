#%%
from matplotlib.pyplot import table
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

table_name = 'clrc_pth_srgc'
file_name = config['file_name'][table_name]
input_path = Path(config['path_config']['input_path'])
output_path = Path(config['path_config']['output_path'])
columns = config['data_config']['required'][table_name.upper()]
prefix = config['data_config']['prefix'][table_name.upper()]

if not output_path.joinpath('0_preprocess').exists():
    output_path.joinpath('0_preprocess').mkdir(parents=True)

output_path = output_path.joinpath('0_preprocess')

#%%
pth_srgc = read_file(input_path, file_name)

#%%
srgc_required = pth_srgc[columns.keys()]

#%%
srgc_required = convert_dates(srgc_required, config, table_name.upper())

#%% 
srgc_required = remove_invalid_values(srgc_required)

#%%
srgc_required = srgc_required.rename(columns = {'SGPT_ACPT_YMD':'TIME'})

duplicated_counts = srgc_required[['PT_SBST_NO','TIME']].duplicated().sum()
print(f'duplicated_counts : {duplicated_counts}')

#%%
srgc_final = srgc_required.set_index(['PT_SBST_NO','TIME'])

#%%
srgc_final = srgc_final.add_prefix(prefix)

#%%
srgc_final.to_pickle(output_path.joinpath(table_name + '.pkl'))