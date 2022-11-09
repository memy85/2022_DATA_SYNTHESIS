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

file_name = "CLRC_PTH_MNTY"

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

 #%%
    def encode_this(x):
        if math.isnan(x) :
            return np.nan
        if x == 2:
            return 'intact'
        else :
            return 'loss'

    data = data.dropna(subset=['IMPT_HM1E_RSLT_CD','IMPT_HP2E_RSLT_CD','IMPT_HS2E_RSLT_CD','IMPT_HS6E_RSLT_CD'], how='all')

    #%%
    data = data.groupby(['PT_SBST_NO'], as_index=False).min()

    #%%

    data['IMPT_HM1E_RSLT_NM']= data['IMPT_HM1E_RSLT_CD'].apply(encode_this)
    data['IMPT_HP2E_RSLT_NM']= data['IMPT_HP2E_RSLT_CD'].apply(encode_this)
    data['IMPT_HS2E_RSLT_NM']= data['IMPT_HS2E_RSLT_CD'].apply(encode_this)
    data['IMPT_HS6E_RSLT_NM']= data['IMPT_HS6E_RSLT_CD'].apply(encode_this)

    #%%

    original_path = PROJ_PATH.joinpath(f'data/raw/{file_name}.xlsx')
    head = pd.read_excel(original_path, nrows=0)

    #%%

    data = data.rename(columns={'TIME':'IMPT_ACPT_YMD'})
    #%%
    data['IMPT_READ_YMD'] = data['IMPT_ACPT_YMD'] + pd.to_timedelta(3, 'days')

    #%%
    data = pd.concat([head, data])
    
    data['CENTER_CD'] = 00000
    data['IRB_APRV_NO'] = '2-2222-02-22'
    data['CRTN_DT'] = '0200.0'
    data['IMPT_SEQ'] = 1
    
    dtypes = head.dtypes.to_dict()
    #%%
    data['IMPT_ACPT_YMD']= data['IMPT_ACPT_YMD'].astype('datetime64[ns]')
    data['IMPT_READ_YMD']= data['IMPT_READ_YMD'].astype('datetime64[ns]')
    data = data.astype(dtypes)

    data.to_excel(OUTPUT_PATH.joinpath(f'{file_name}_{args.epsilon}'))
    pass

if __name__ == "__main__":
    main()