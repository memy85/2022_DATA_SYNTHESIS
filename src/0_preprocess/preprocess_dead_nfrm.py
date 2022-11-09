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

table_name = 'clrc_dead_nfrm'
file_name = config['file_name'][table_name]
input_path = Path(config['path_config']['input_path'])
output_path = Path(config['path_config']['output_path'])
columns = config['data_config']['required'][table_name.upper()]
prefix = config['data_config']['prefix'][table_name.upper()]

if not output_path.joinpath('0_preprocess').exists():
    output_path.joinpath('0_preprocess').mkdir(parents=True)

output_path = output_path.joinpath('0_preprocess')

#%%
nfrm = read_file(input_path, file_name)

nfrm_required = nfrm[columns.keys()]
# %%
nfrm_required = convert_dates(nfrm_required, config, table_name.upper())


#%%
nfrm_required = nfrm_required.rename(columns = {'DEAD_YMD':'TIME'})

duplicated_counts = nfrm_required[['PT_SBST_NO','TIME']].duplicated().sum()
print(f'duplicated_counts : {duplicated_counts}')
#%%
nfrm_required = nfrm_required.set_index(['PT_SBST_NO','TIME'])

nfrm_required['DEAD'] = 1

#%%
nfrm_required = nfrm_required.astype({"DEAD":"object"})

#%%
nfrm_final = nfrm_required.copy()

nfrm_final = nfrm_final.add_prefix(prefix)

#%%
nfrm_final.to_pickle(output_path.joinpath('clrc_dead_nfrm.pkl'))
