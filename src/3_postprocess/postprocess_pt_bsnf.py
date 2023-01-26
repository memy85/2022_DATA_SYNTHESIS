#%%
import pandas as pd
import numpy as np
import pickle
import os, sys
from pathlib import Path
import argparse

from requests import head

PROJ_PATH = Path(__file__).parents[2]
sys.path.append(PROJ_PATH.joinpath('src').as_posix())

from MyModule.utils import *
config = load_config()

PROJ_PATH = Path(config['path_config']['project_path'])
INPUT_PATH = PROJ_PATH.joinpath('data/processed/2_restore/restore_to_db_form')
OUTPUT_PATH = PROJ_PATH.joinpath('data/processed/3_postprocess/')

if not OUTPUT_PATH.exists() :
    OUTPUT_PATH.mkdir(parents=True)

file_name = "CLRC_PT_BSNF"

# %%
def load_file_epsilon(epsilon):
    return read_file(INPUT_PATH, f'{file_name}_{epsilon}.pkl')

data = load_file_epsilon(0.1)

#%%
def load_s1(epsilon):
    return read_file(PROJ_PATH.joinpath(f'data/processed/2_restore/restore_to_s1'), f'S1_{epsilon}.pkl')

#%%
original_data =pd.read_excel(PROJ_PATH.joinpath('data/raw/CLRC_PT_BSNF.xlsx'))
s1 = load_s1(0.1)

#%%
diag_name = original_data[['BSPT_FRST_DIAG_CD','BSPT_FRST_DIAG_NM']].drop_duplicates()

data = data.merge(diag_name, how='left')


#%% first diag ymd 만들기
first_diag = data.groupby(['PT_SBST_NO','BSPT_FRST_DIAG_YMD'],as_index=False).TIME.min()
first_diag  = first_diag.drop(columns='BSPT_FRST_DIAG_YMD').rename(columns={'TIME':"BSPT_FRST_DIAG_YMD"})
data = data.drop(columns=['BSPT_FRST_DIAG_YMD','TIME'])
data = data.merge(first_diag, how='left')

#%%
data =  data.drop_duplicates()
num_dup = data.duplicated('PT_SBST_NO').sum()
print(f'Number of duplicated patients : {num_dup}')


#%%
operation_data = s1.filter(regex='OPRT|TIME|PT_SBST_NO').dropna(subset=['OPRT_NFRM_OPRT_CLCN_OPRT_KIND_CD','OPRT_NFRM_OPRT_CURA_RSCT_CD'], how='all')

first_oprt = operation_data.groupby(['PT_SBST_NO'], as_index=False)['TIME'].min()
first_oprt = first_oprt.rename(columns= {'TIME':'BSPT_FRST_OPRT_YMD'})
data = data.merge(first_oprt, how='left')

#%%
num_dup = data.duplicated('PT_SBST_NO').sum()
print(f'Number of duplicated patients : {num_dup}')

#%%
# data['BSPT_FRST_ANCN_TRTM_STRT_YMD']
# #%%
# trtm_data = s1.filter(regex='TRTM|TIME|PT_SBST_NO').dropna(subset=['TRTM_CASB_CSTR_NT','TRTM_CASB_CSTR_PRPS_CD','TRTM_CASB_CSTR_REGN_CD','TRTM_RD_RDT'], how='all')
# trtm_data


# #%%

# first_oprt = operation_data.groupby(['PT_SBST_NO'], as_index=False)['TIME'].min()
# first_oprt = first_oprt.rename(columns= {'TIME':'BSPT_FRST_OPRT_YMD'})
# data = data.merge(first_oprt, how='left')

#%%
rd_data = s1.filter(regex='TRTM_RD|TIME|PT_SBST_NO').dropna(subset=['TRTM_RD_RDT'], how='all')
first_rd = rd_data.groupby(['PT_SBST_NO'], as_index=False)['TIME'].min()

first_rd = first_rd.rename(columns = {'TIME':'TRTM_RD_RDT'})

data = data.merge(first_rd, how='left')

#%% deatj
death_data = s1.filter(regex='DEAD|PT_SBST_NO|TIME')

death_data = death_data.query('DEAD_NFRM_DEAD == 1')

death_date = death_data.groupby(['PT_SBST_NO'], as_index=False)['TIME'].max()
death_date = death_date.rename(columns = {'TIME':'BSPT_DEAD_YMD'})

data = data.merge(death_date, how='left')

#%% relapse
relapse_data = s1.filter(regex='RLPS|PT_SBST_NO|TIME')

relapse_data = relapse_data.query('DG_RCNF_RLPS == 1')

relapse_date = relapse_data.groupby(['PT_SBST_NO'], as_index=False)['TIME'].max()
relapse_date = relapse_date.rename(columns = {'TIME':'RLPS_DIAG_YMD'})

data = data.merge(death_date, how='left')



#%%
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epsilon', '--e', help="epsilon values")
    args = parser.parse_args()
#%%
    data = load_file_epsilon(args.epsilon)

    data = data.dropna(subset=['OPRT_CLCN_OPRT_KIND_CD','OPRT_CURA_RSCT_CD'])

#%%
    data = data.rename(columns={"TIME":"OPRT_YMD"})

    #%%
    original_path = PROJ_PATH.joinpath(f'data/raw/{file_name}.xlsx')
    original_data = pd.read_excel(original_path)

    #%%
    reference1 = original_data[['OPRT_CLCN_OPRT_KIND_CD','OPRT_CLCN_OPRT_KIND_NM']].drop_duplicates().reset_index(drop=True)
    reference2 = original_data[['OPRT_CURA_RSCT_CD','OPRT_CURA_RSCT_NM']].drop_duplicates().reset_index(drop=True)
    reference2 = reference2.dropna()
    #%%
    head = pd.read_excel(original_path, nrows = 0)

    head = head.drop(columns=["OPRT_EDI_CD","OPRT_NM"])

    data = pd.concat([head,data])

    #%%
    data = data.drop(columns=['OPRT_CLCN_OPRT_KIND_NM','OPRT_CURA_RSCT_NM'])

    #%%
    data = data.merge(reference1, how='left', on='OPRT_CLCN_OPRT_KIND_CD')
    data = data.merge(reference2, how='right', on='OPRT_CURA_RSCT_CD')
    #%%
    data['CENTER_CD'] = 00000
    data['IRB_APRV_NO'] = '2-2222-02-22'
    data['CRTN_DT'] = '0200.0'
    data['OPRT_SEQ'] = 1
    dtypes = head.dtypes.to_dict()

#%%
    data = data[head.columns]
    data['OPRT_YMD']= data['OPRT_YMD'].astype('datetime64[ns]')
    data = data.astype(dtypes)

    data.to_excel(OUTPUT_PATH.joinpath(f'{file_name}_{args.epsilon}'))
    pass

if __name__ == "__main__":
    main()