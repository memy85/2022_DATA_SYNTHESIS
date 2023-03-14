#%%
from time import time_ns
import yaml
import pandas as pd
import pickle
from pathlib import Path
import numpy as np
import os, sys

project_path = Path(__file__).parents[2]
os.sys.path.append(project_path.as_posix())

from src.MyModule.utils import *
config = load_config()

table_name = 'clrc_pth_mlcr'
file_name = config['file_name'][table_name]
input_path = Path(config['path_config']['input_path'])
output_path = Path(config['path_config']['output_path'])
columns = config['data_config']['required'][table_name.upper()]
prefix = config['data_config']['prefix'][table_name.upper()]

if not output_path.joinpath('0_preprocess').exists():
    output_path.joinpath('0_preprocess').mkdir(parents=True)

output_path = output_path.joinpath('0_preprocess')

#%%
pth_mlcr = read_file(input_path, file_name)

#%% 
mlcr_required = pth_mlcr[columns.keys()]



#%%
mlcr_required = convert_dates(mlcr_required, config, table_name.upper())

#%%
mlcr_required = mlcr_required.rename(columns = {'MLPT_ACPT_YMD':'TIME'})

duplicated_counts = mlcr_required[['PT_SBST_NO','TIME']].duplicated().sum()
print(f'duplicated_counts : {duplicated_counts}')

#%%
mlcr_required = mlcr_required.set_index(['PT_SBST_NO','TIME'])

#%%
mlcr_final = mlcr_required.add_prefix(prefix)

#%%
mlcr_final.to_pickle(output_path.joinpath(table_name + '.pkl'))
# %%
