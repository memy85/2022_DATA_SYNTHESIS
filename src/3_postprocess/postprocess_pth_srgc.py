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

file_name = "CLRC_PTH_SRGC"

# %%
def load_file_epsilon(epsilon):
    return read_file(INPUT_PATH, f'{file_name}_{epsilon}.pkl')

#%%
# 여기서 데이터를 조금 정제해야 함

#%%
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epsilon', '--e', help="epsilon values")
    args = parser.parse_args()
#%%
    data = load_file_epsilon(args.epsilon)

    data = data.rename(columns={"TIME":"SGPT_ACPT_YMD"})
    #%%
    data["SGPT_READ_YMD"] = data["SGPT_ACPT_YMD"] + pd.to_timedelta(3, 'days')


    #%%
    original_path = PROJ_PATH.joinpath(f'data/raw/{file_name}.xlsx')
    head = pd.read_excel(original_path, nrows=0)
    original_data = pd.read_excel(original_path)
    #%%
    reference1 = original_data[['SGPT_CELL_DIFF_CD','SGPT_CELL_DIFF_NM']].drop_duplicates()

    #%%
    data = data.merge(reference1, how='left')
    #%%
    def encoding1(x):
        if math.isnan(x):
            return np.nan
        if x == 2 :
            return "uninvovled"
        elif x == 1 :
            return "involved"
        elif x == 3 :
            return "interminate"
        elif x == 9 :
            return 'Other'

    def encoding2(x):
        if math.isnan(x):
            return np.nan
        if x == 1 :
            return "present"
        elif x == 2 :
            return "absent"
        elif x == 3 :
            return "non identified"
        elif x == 4 :
            return "no record"
        else :
            return "Other"

    #%%
    data['SGPT_SRMG_PCTS_STAT_NM'] = data['SGPT_SRMG_PCTS_STAT_CD'].apply(encoding1)
    data['SGPT_SRMG_DCTS_STAT_NM'] = data['SGPT_SRMG_DCTS_STAT_CD'].apply(encoding1)
    data['SGPT_SRMG_RCTS_STAT_NM'] = data['SGPT_SRMG_RCTS_STAT_CD'].apply(encoding1)

    data['SGPT_NERV_PREX_NM'] = data['SGPT_NERV_PREX_CD'].apply(encoding2)
    data['SGPT_VNIN_NM'] = data['SGPT_VNIN_CD'].apply(encoding2)
    data['SGPT_ANIN_NM'] = data['SGPT_ANIN_CD'].apply(encoding2)
    data['SGPT_TUMR_BUDD_NM'] = data['SGPT_TUMR_BUDD_CD'].apply(encoding2)


#%%
    data = pd.concat([head, data])
    
    data['CENTER_CD'] = 00000
    data['IRB_APRV_NO'] = '2-2222-02-22'
    data['CRTN_DT'] = '2022-02-22'
    data['IMPT_SEQ'] = 1
    
    dtypes = head.dtypes.to_dict()
    #%%
    data['SGPT_ACPT_YMD']= data['SGPT_ACPT_YMD'].astype('datetime64[ns]')
    data['SGPT_READ_YMD']= data['SGPT_READ_YMD'].astype('datetime64[ns]')
    data = data.astype(dtypes)

    data.to_excel(OUTPUT_PATH.joinpath(f'{file_name}_{args.epsilon}'))
    pass

if __name__ == "__main__":
    main()