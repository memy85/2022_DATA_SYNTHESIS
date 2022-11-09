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

table_name = 'clrc_dg_rcnf'
file_name = config['file_name'][table_name]
input_path = Path(config['path_config']['input_path'])
output_path = Path(config['path_config']['output_path'])
columns = config['data_config']['required'][table_name.upper()]
prefix = config['data_config']['prefix'][table_name.upper()]

if not output_path.joinpath('0_preprocess').exists():
    output_path.joinpath('0_preprocess').mkdir(parents=True)

output_path = output_path.joinpath('0_preprocess')

#%%
dg_rcnf = read_file(input_path, file_name)

#%%
rcnf_required = dg_rcnf[columns.keys()]


#%%
rcnf_required = convert_dates(rcnf_required, config, table_name.upper())

#%%
rcnf_required = rcnf_required.rename(columns = {'RLPS_DIAG_YMD':'TIME'})

#%%

duplicated_counts = rcnf_required[['PT_SBST_NO','TIME']].duplicated().sum()
print(f'duplicated_counts : {duplicated_counts}')

#%%
rcnf_final = rcnf_required.set_index(['PT_SBST_NO','TIME'])

rcnf_final['RLPS'] = 1

rcnf_final = rcnf_final.astype({'RLPS':'object'})

#%%
rcnf_final = rcnf_final.add_prefix(prefix)
rcnf_final.to_pickle(output_path.joinpath(table_name + '.pkl'))
# %%
