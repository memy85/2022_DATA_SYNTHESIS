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

table_name = 'clrc_trtm_rd'
file_name = config['file_name'][table_name]
input_path = Path(config['path_config']['input_path'])
output_path = Path(config['path_config']['output_path'])
columns = config['data_config']['required'][table_name.upper()]
prefix = config['data_config']['prefix'][table_name.upper()]

if not output_path.joinpath('0_preprocess').exists():
    output_path.joinpath('0_preprocess').mkdir(parents=True)

output_path = output_path.joinpath('0_preprocess')

#%%
trtm_rd = read_file(input_path, file_name)
#%%
rd_required = trtm_rd[columns.keys()]
#%%
rd_required = convert_dates(rd_required, config, table_name.upper())



# %%
def expand_row(row):
    dates = pd.date_range(row['RDT_STRT_YMD'], row['RDT_END_YMD'])
    coppied = row.copy()
    new_rows = []
    for date in dates:
        coppied = coppied.copy()
        coppied['TIME'] = date
        coppied['RDT'] = 1
        new_rows.append(coppied)
    return pd.DataFrame(new_rows)[['PT_SBST_NO','TIME','RDT']]


# %%
expanded = []
for _, row in rd_required.iterrows():
    expanded.append(expand_row(row))

# %%
rd_final = pd.concat(expanded)

#%%
rd_final = rd_final.drop_duplicates().reset_index(drop=True)

duplicated_counts = rd_final[['PT_SBST_NO','TIME']].duplicated().sum()
print(f'duplicated_counts : {duplicated_counts}')

#%%
rd_final = rd_final.set_index(['PT_SBST_NO','TIME'])
#%%
rd_final = remove_invalid_values(rd_final)  
rd_final = rd_final.add_prefix(prefix)

rd_final.to_pickle(output_path.joinpath(table_name + '.pkl'))