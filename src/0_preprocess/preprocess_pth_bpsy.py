
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

table_name = 'clrc_pth_bpsy'
file_name = config['file_name'][table_name]
input_path = Path(config['path_config']['input_path'])
output_path = Path(config['path_config']['output_path'])
columns = config['data_config']['required'][table_name.upper()]
prefix = config['data_config']['prefix'][table_name.upper()]

if not output_path.joinpath('0_preprocess').exists():
    output_path.joinpath('0_preprocess').mkdir(parents=True)

output_path = output_path.joinpath('0_preprocess')

#%%
pth_bpsy = read_file(input_path, file_name)

#%% 
bpsy_required = pth_bpsy[columns.keys()]

#%%
bpsy_required = convert_dates(bpsy_required, config, table_name.upper())

#%%
bpsy_required = remove_invalid_values(bpsy_required)

#%%
bpsy_required = bpsy_required.rename(columns = {'BPTH_ACPT_YMD':'TIME'})

duplicated_counts = bpsy_required[['PT_SBST_NO','TIME']].duplicated().sum()
print(f'duplicated_counts : {duplicated_counts}')
#%%

bpsy_final = bpsy_required.set_index(['PT_SBST_NO','TIME'])
bpsy_final = bpsy_final.add_prefix(prefix)

#%%
bpsy_final.to_pickle(output_path.joinpath(table_name + '.pkl'))
