#%%
import pandas as pd
import numpy as np
import pickle
import os, sys
from pathlib import Path
import argparse

PROJ_PATH = Path(__file__).parents[2]
sys.path.append(PROJ_PATH.joinpath('src').as_posix())

from MyModule.utils import *
config = load_config()

PROJ_PATH = Path(config['path_config']['project_path'])
INPUT_PATH = PROJ_PATH.joinpath('data/processed/2_restore/restore_to_db_form')
OUTPUT_PATH = PROJ_PATH.joinpath('data/processed/3_postprocess/')

if not OUTPUT_PATH.exists() :
    OUTPUT_PATH.mkdir(parents=True)

file_name = "CLRC_PTH_BPSY"

# %%
def load_file_epsilon(epsilon):
    return read_file(INPUT_PATH, f'{file_name}_{epsilon}.pkl')



#%%

#%%
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epsilon', '--e', help="epsilon values")
    args = parser.parse_args()
#%%
    data = load_file_epsilon(args.epsilon)

#%%
    data = data.rename(columns = {'TIME':'BPTH_ACPT_YMD'})

#%%
    data = data.rename(columns = {'BRSLT_CONT':'BPTH_BPSY_RSLT_CONT'})

#%%
    data = data.dropna(subset=['BPTH_BPSY_RSLT_CONT','BPTH_CELL_DIFF_CD'], how='any')
    data = data.drop_duplicates(subset=['BPTH_BPSY_RSLT_CONT','BPTH_CELL_DIFF_CD'])
    #%%
    data['BPTH_READ_YMD'] = data['BPTH_ACPT_YMD'] + pd.to_timedelta(3,'days')
    #%%

    original_path = PROJ_PATH.joinpath(f'data/raw/{file_name}.xlsx')
    original_data = pd.read_excel(original_path)
    head = pd.read_excel(original_path, nrows=0)

    #%%
    reference1 = original_data[['BPTH_CELL_DIFF_CD','BPTH_CELL_DIFF_NM']].drop_duplicates()
    #%%
    data = data.merge(reference1, how='left')
    #%%
    data = pd.concat([head, data])

    #%%
    data['CENTER_CD'] = 00000
    data['IRB_APRV_NO'] = '2-2222-02-22'
    data['CRTN_DT'] = '0200.0'
    data['BPTH_SEQ'] = 1
    data['BPTH_SITE_CONT'] = 'language not supported'
    
    dtypes = head.dtypes.to_dict()
    #%%
    data['BPTH_ACPT_YMD']= data['BPTH_ACPT_YMD'].astype('datetime64[ns]')
    data['BPTH_READ_YMD']= data['BPTH_READ_YMD'].astype('datetime64[ns]')
    data = data.astype(dtypes)

    data.to_excel(OUTPUT_PATH.joinpath(f'{file_name}_{args.epsilon}'))
    pass

if __name__ == "__main__":
    main()