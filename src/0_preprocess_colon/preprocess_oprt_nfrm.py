
#%%
import yaml
import pandas as pd
import pickle
from pathlib import Path
import numpy as np
import os, sys
from datetime import datetime

project_path = Path(__file__).parents[2]
os.sys.path.append(project_path.as_posix())

from src.MyModule.utils import *
config = load_config()

table_name = 'clrc_oprt_nfrm'
file_name = config['file_name'][table_name]
input_path = Path(config['path_config']['input_path'])
output_path = Path(config['path_config']['output_path'])
columns = config['data_config']['required'][table_name.upper()]
prefix = config['data_config']['prefix'][table_name.upper()]

if not output_path.joinpath('0_preprocess').exists():
    output_path.joinpath('0_preprocess').mkdir(parents=True)

output_path = output_path.joinpath('0_preprocess')

#%%
oprt_nfrm = read_file(input_path, file_name)

#%% 
oprt_required = oprt_nfrm[columns.keys()]

#%%
oprt_required = convert_dates(oprt_required, config, table_name.upper())

#%%

oprt_required = oprt_required.rename(columns = {'OPRT_YMD':'TIME'})
#%%

idx = oprt_required.duplicated(['PT_SBST_NO','TIME'], False)
print(f"the duplicated rows are {len(oprt_required[idx])}")

#%%
oprt_required = oprt_required.drop_duplicates(subset=['PT_SBST_NO','TIME'])



#%%

# -> 중복된 날에 다른 수술을 받은 기록이 있다. 그런데 패턴이 있는 것 같고, 케이스가 많지도 않다
# -> 이런 경우, 하나의 패턴으로 묶어주는 작업을 진행해야 할 거 같다.
# 
# def check_duplicated(data):
#     if data.duplicated(['PT_SBST_NO','TIME']).sum() != 0 :
#         return 1
#     else : 
#         return 0

# def check_correct_code(patient_id):
#     table_name = 'clrc_pt_bsnf'
#     file_name = config['file_name'][table_name]
#     bsnf = read_file(input_path, file_name)
#     dx = bsnf[bsnf.PT_SBST_NO == patient_id]['BSPT_FRST_DIAG_NM'].values
    
#     if dx == 'colon':
#         return 11
#     elif dx == 'rectum':
#         return 1
#     else :
#         return 11


# def convert_data_based_on_pt_bsnf(data):
#     result = check_duplicated(data)
#     if result == 1 :
#         msk = data.duplicated(['PT_SBST_NO','TIME'], keep=False)
#         duplicated_dates = data[msk][['PT_SBST_NO','TIME']]
#         patients = list(duplicated_dates.PT_SBST_NO.unique())
        
#         indices = []
#         for pt in patients : 
#             code = check_correct_code(pt)
#             time = pd.to_datetime(duplicated_dates[duplicated_dates.PT_SBST_NO == pt]['TIME'].unique()[0])
#             index = data[(data.PT_SBST_NO == pt) & (data.TIME == time) & (data.OPRT_CLCN_OPRT_KIND_CD != code)].index
            
#             index = index.values
#             indices.extend(index)
            
#         data = data.drop(index=indices)
#         return data.reset_index(drop=True)
#     else :
#         return data

# oprt_required = convert_data_based_on_pt_bsnf(oprt_required)

#%%
duplicated_counts = oprt_required[['PT_SBST_NO','TIME']].duplicated().sum()
print(f'duplicated_counts : {duplicated_counts}')

#%%

oprt_final = oprt_required.set_index(['PT_SBST_NO','TIME'])

oprt_final = oprt_final.add_prefix(prefix)

oprt_final = remove_invalid_values(oprt_final)


#%%
oprt_final.to_pickle(output_path.joinpath(table_name + '.pkl'))
