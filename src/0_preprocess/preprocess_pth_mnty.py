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

table_name = 'clrc_pth_mnty'
file_name = config['file_name'][table_name]
input_path = Path(config['path_config']['input_path'])
output_path = Path(config['path_config']['output_path'])
columns = config['data_config']['required'][table_name.upper()]
prefix = config['data_config']['prefix'][table_name.upper()]

if not output_path.joinpath('0_preprocess').exists():
    output_path.joinpath('0_preprocess').mkdir(parents=True)

output_path = output_path.joinpath('0_preprocess')

#%%
pth_mnty = read_file(input_path, file_name)

#%%
mnty_required = pth_mnty[columns.keys()]

#%%
mnty_required = convert_dates(mnty_required, config, table_name.upper())

#%%
mnty_required = mnty_required.rename(columns = {'IMPT_ACPT_YMD':'TIME'})

duplicated_counts = mnty_required[['PT_SBST_NO','TIME']].duplicated().sum()
print(f'duplicated_counts : {duplicated_counts}')

#%%
mnty_final = mnty_required.set_index(['PT_SBST_NO','TIME'])

#%%
mnty_final = mnty_final.add_prefix(prefix)

#%%

mnty_final = remove_invalid_values(mnty_final)

#%%
mnty_final.to_pickle(output_path.joinpath(table_name + '.pkl'))