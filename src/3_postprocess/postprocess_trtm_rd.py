#%%
import pandas as pd
import numpy as np
import pickle
import os, sys
from pathlib import Path
import argparse
import numpy as np
import math

PROJ_PATH = Path(__file__).parents[2]
sys.path.append(PROJ_PATH.joinpath('src').as_posix())

from MyModule.utils import *
config = load_config()

PROJ_PATH = Path(config['path_config']['project_path'])
INPUT_PATH = PROJ_PATH.joinpath('data/processed/2_restore/restore_to_db_form')
OUTPUT_PATH = PROJ_PATH.joinpath('data/processed/3_postprocess/')

if not OUTPUT_PATH.exists() :
    OUTPUT_PATH.mkdir(parents=True)

file_name = "CLRC_TRTM_RD"

# %%
def load_file_epsilon(epsilon):
    return read_file(INPUT_PATH, f'{file_name}_{epsilon}.pkl')



#%%
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epsilon', '--e', help="epsilon values")
    args = parser.parse_args()
#%%
    data = load_file_epsilon(args.epsilon)

    data = load_file_epsilon(0.1)
    
    data = data.dropna()

#%%
    data = data.rename(columns ={'TIME':"RDT_STRT_YMD"})
    data['RDT_END_YMD'] = data['RDT_STRT_YMD'] + pd.to_timedelta(25, 'days')

#%%
    original_path = PROJ_PATH.joinpath(f'data/raw/{file_name}.xlsx')
    head = pd.read_excel(original_path, nrows=0)
    data = pd.concat([head, data])

    data = data.rename(columns = {'TIME':'CSTR_STRT_YMD'})
    data['CSTR_END_YMD'] = data['CSTR_STRT_YMD'] + pd.to_timedelta(28, 'days')

    #%%
    original_path = PROJ_PATH.joinpath(f'data/raw/{file_name}.xlsx')
    head = pd.read_excel(original_path, nrows=0)

    data = pd.concat([head, data])
    
    data['CENTER_CD'] = 00000
    data['IRB_APRV_NO'] = '2-2222-02-22'
    data['CRTN_DT'] = '2022-02-22'
    data['RDT_RT_NO'] = 1
    data['RDT_SITE_CD'] = "no codes"
    data['RDT_SITE_CD'] = "no name"
    
    dtypes = head.dtypes.to_dict()
    #%%
    data['RDT_STRT_YMD']= data['RDT_STRT_YMD'].astype('datetime64[ns]')
    data['RDT_END_YMD']= data['RDT_END_YMD'].astype('datetime64[ns]')
    data = data.astype(dtypes)

    data.to_excel(OUTPUT_PATH.joinpath(f'{file_name}_{args.epsilon}'))
    pass

if __name__ == "__main__":
    main()