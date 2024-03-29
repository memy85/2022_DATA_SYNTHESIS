#%%
from matplotlib.pyplot import table
from datetime import timedelta
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

table_name = 'clrc_trtm_casb'
file_name = config['file_name'][table_name]
input_path = Path(config['path_config']['input_path'])
output_path = Path(config['path_config']['output_path'])
columns = config['data_config']['required'][table_name.upper()]
prefix = config['data_config']['prefix'][table_name.upper()]

if not output_path.joinpath('0_preprocess').exists():
    output_path.joinpath('0_preprocess').mkdir(parents=True)

output_path = output_path.joinpath('0_preprocess')

#%%
trtm_casb = read_file(input_path, file_name)
#%%
casb_required = trtm_casb[columns.keys()]

#%%
casb_required = convert_dates(casb_required, config, table_name.upper())

#%%
not_date_columns = [col for col in columns if columns[col] != 'datetime64[ns]'] + ['TIME']

#%%

casb_required = casb_required.drop(columns="CSTR_END_YMD").copy()
#%%

# def expand_row(row, cols):
#     dates = pd.date_range(row['CSTR_STRT_YMD'], row['CSTR_END_YMD'] - timedelta(1))
#     coppied = row.copy()
#     new_rows = []
#     for date in dates:
#         coppied = coppied.copy()
#         coppied['TIME'] = date
#         new_rows.append(coppied)
#     return pd.DataFrame(new_rows)


# # %%
# expanded = []
# for _, row in casb_required.iterrows():
#     expanded.append(expand_row(row, cols=not_date_columns))

# #%%
# casb_final = pd.concat(expanded)[not_date_columns]

#%%

casb_required = casb_required.rename(columns={'CSTR_STRT_YMD':'TIME'})
casb_final = casb_required.copy()

#%%
duplicated_counts = casb_final[['PT_SBST_NO','TIME']].duplicated().sum()
print(f'duplicated_counts : {duplicated_counts}')


#%%
casb_final = casb_final.set_index(['PT_SBST_NO','TIME'])

#%%
casb_final = casb_final.add_prefix(prefix)

casb_final.to_pickle(output_path.joinpath(table_name + '.pkl'))
# %%
