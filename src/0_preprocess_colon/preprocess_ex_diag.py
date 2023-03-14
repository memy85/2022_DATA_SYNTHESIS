
#%%
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

table_name = 'clrc_ex_diag'
file_name = config['file_name'][table_name]
input_path = Path(config['path_config']['input_path'])
output_path = Path(config['path_config']['output_path'])
columns = config['data_config']['required'][table_name.upper()]
prefix = config['data_config']['prefix'][table_name.upper()]

if not output_path.joinpath('0_preprocess').exists():
    output_path.joinpath('0_preprocess').mkdir(parents=True)

output_path = output_path.joinpath('0_preprocess')
#%% load data
ex_diag = read_file(input_path, file_name)

#%%
ex_diag['CEXM_NM'] = ex_diag.CEXM_NM.str.strip()

#%%
ex_diag = ex_diag.query("CEXM_NM == 'CEA' ")

#%%  read data
ex_diag_required = ex_diag[columns.keys()].copy()

ex_diag_required['CEXM_RSLT_CONT'] = ex_diag_required.CEXM_RSLT_CONT.replace('>|<','', regex=True)

ex_diag_required = ex_diag_required.replace('+++++', np.nan)
ex_diag_required = ex_diag_required.replace(':::::',np.nan)
ex_diag_required['CEXM_RSLT_CONT'] = ex_diag_required.CEXM_RSLT_CONT.replace('.', np.nan)

ex_diag_required['CEXM_RSLT_CONT'] = ex_diag_required['CEXM_RSLT_CONT'].astype('float32')


#%% convert dates
ex_diag_required = convert_dates(ex_diag_required, config=config, table_name=table_name.upper())
ex_diag_required = ex_diag_required.drop_duplicates()

#%%
ex_diag_required = ex_diag_required.groupby(['PT_SBST_NO','TIME','CEXM_NM'], as_index=False).mean()


#%%
ex_diag_required = ex_diag_required.rename(columns = {'CEXM_YMD':"TIME"})

duplicated_counts = ex_diag_required[['PT_SBST_NO','TIME']].duplicated().sum()
print(f'duplicated_counts : {duplicated_counts}')

#%% convert to time index 

pivoted_ex_diag = pd.pivot_table(ex_diag_required, index=['PT_SBST_NO','TIME'], columns= 'CEXM_NM', values='CEXM_RSLT_CONT')
pivoted_ex_diag = pivoted_ex_diag.add_prefix(prefix)


#%%
pivoted_ex_diag.to_pickle(output_path.joinpath('clrc_ex_diag.pkl'))
