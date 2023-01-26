#%%
import pandas as pd
import numpy as np
import pickle
import os, sys
from pathlib import Path
import argparse


PROJ_PATH = Path(__file__).resolve().parents[2]
sys.path.append(PROJ_PATH.joinpath('src').as_posix())

from MyModule.utils import *
config = load_config()

PROJ_PATH = Path(config['path_config']['project_path'])
INPUT_PATH = PROJ_PATH.joinpath('/mnt/synthetic_data/data/processed/2_restore/restore_to_db_form/D0')
OUTPUT_PATH = PROJ_PATH.joinpath('/mnt/synthetic_data/data/processed/4_evaluate/make_whole_data/D0')

if not OUTPUT_PATH.exists() :
    OUTPUT_PATH.mkdir(parents=True)

file_name = "CLRC_PT_BSNF"

parser = argparse.ArgumentParser()
parser.add_argument('--epsilon', '--e', help="epsilon values")
args = parser.parse_args()

print(f'epsilon is {args.epsilon}')

# %%
def load_file_epsilon(epsilon):
    return read_file(INPUT_PATH, f'{file_name}_{epsilon}.pkl')

def load_s1(epsilon):
    return read_file(PROJ_PATH.joinpath(f'/mnt/synthetic_data/data/processed/2_restore/restore_to_s1/D0'), f'S1_{epsilon}.pkl')

#%%

# data = load_file_epsilon(0.1)
# original_data =pd.read_excel(PROJ_PATH.joinpath('data/raw/CLRC_PT_BSNF.xlsx'))
# s1 = load_s1(0.1)

data = load_file_epsilon(args.epsilon)
original_data =pd.read_excel(PROJ_PATH.joinpath('/mnt/synthetic_data/data/raw/CLRC_PT_BSNF.xlsx'))
s1 = load_s1(args.epsilon)

#%%
diag_name = original_data[['BSPT_FRST_DIAG_CD','BSPT_FRST_DIAG_NM']].drop_duplicates()
#data = data.astype("object")
#data = pd.merge(data,diag_name, how='left')
#data = pd.concat([data, diag_name], axis=1, join_axes=[data.index])

print(data)

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

first_rd = first_rd.rename(columns = {'TIME':'BSPT_FRST_RDT_STRT_YMD'})

data = data.merge(first_rd, how='left')

#%% deatj
death_data = s1.filter(regex='DEAD|PT_SBST_NO|TIME')

death_data = death_data.query('DEAD_NFRM_DEAD == 1')

death_date = death_data.groupby(['PT_SBST_NO'], as_index=False)['TIME'].max()
death_date = death_date.rename(columns = {'TIME':'BSPT_DEAD_YMD'})

data = data.merge(death_date, how='left')

#%%
death_data = s1.filter(regex='DEAD|PT_SBST_NO|TIME')
alive_patients = death_data.groupby('PT_SBST_NO', as_index=False)['DEAD_NFRM_DEAD'].max().query('DEAD_NFRM_DEAD != 1').PT_SBST_NO

alive = death_data[death_data.PT_SBST_NO.isin(alive_patients)]
alive = alive.groupby('PT_SBST_NO',as_index=False).TIME.max()
alive = alive.rename(columns = {'TIME':'CENTER_LAST_VST_YMD'})

data = data.merge(alive, how='left')

#%% relapse
relapse_data = s1.filter(regex='RLPS|PT_SBST_NO|TIME')

relapse_data = relapse_data.query('DG_RCNF_RLPS == 1')

relapse_date = relapse_data.groupby(['PT_SBST_NO'], as_index=False)['TIME'].max()
relapse_date = relapse_date.rename(columns = {'TIME':'RLPS_DIAG_YMD'})

data = data.merge(relapse_date, how='left')

#%%
data['BSPT_OPRT'] = ~data['BSPT_FRST_OPRT_YMD'].isna()*1

#%%
data['TRTM_RD_RDT'] = ~data['BSPT_FRST_RDT_STRT_YMD'].isna()*1

#%%
data['CENTER_LAST_VST_YMD'] = data['CENTER_LAST_VST_YMD'].fillna(data['BSPT_DEAD_YMD'])
data['CENTER_LAST_VST_YMD'] = data['CENTER_LAST_VST_YMD'].astype('datetime64[ns]')
data['BSPT_DEAD_YMD'] = data['BSPT_DEAD_YMD'].astype('datetime64[ns]')
data['OVRL_DAYS'] = (data.CENTER_LAST_VST_YMD - data.BSPT_FRST_DIAG_YMD).dt.days

data['DEAD'] = ~data['BSPT_DEAD_YMD'].isna()*1
data['RLPS'] = ~data['RLPS_DIAG_YMD'].isna()*1

# %%
comparison_data = data[['PT_SBST_NO','BSPT_SEX_CD','BSPT_FRST_DIAG_CD','BSPT_IDGN_AGE','BSPT_STAG_VL','BSPT_T_STAG_VL','BSPT_N_STAG_VL','BSPT_M_STAG_VL','BSPT_OPRT','TRTM_RD_RDT','OVRL_DAYS','DEAD']].copy()

comparison_data.to_csv(OUTPUT_PATH.joinpath(f'S0_comparsion_data_{args.epsilon}.csv'), index=False)


#%%
ori = original_data[['PT_SBST_NO','BSPT_SEX_CD','BSPT_FRST_DIAG_CD','BSPT_IDGN_AGE','BSPT_DEAD_YMD','CENTER_LAST_VST_YMD','OVRL_SRVL_DTRN_DCNT','BSPT_STAG_VL','BSPT_T_STAG_VL','BSPT_N_STAG_VL','BSPT_M_STAG_VL','BSPT_FRST_RDT_STRT_YMD','BSPT_FRST_OPRT_YMD']]
#%%
# dg_rcnf = pd.read_excel(PROJ_PATH.joinpath('data/raw/CLRC_DG_RCNF.xlsx'))
# # %%

# dg_rcnf

ori['DEAD'] = ~ori['BSPT_DEAD_YMD'].isna()*1
ori['TRTM_RD_RDT'] = ~ori['BSPT_FRST_RDT_STRT_YMD'].isna()*1
ori['BSPT_OPRT'] = ~ori['BSPT_FRST_OPRT_YMD'].isna()*1

#%%
ori = ori.rename(columns = {"OVRL_SRVL_DTRN_DCNT":"OVRL_DAYS"})

ori = ori[['PT_SBST_NO','BSPT_SEX_CD','BSPT_FRST_DIAG_CD','BSPT_IDGN_AGE','BSPT_STAG_VL','BSPT_T_STAG_VL','BSPT_N_STAG_VL','BSPT_M_STAG_VL','BSPT_OPRT','TRTM_RD_RDT','OVRL_DAYS','DEAD']].copy()

ori.to_csv(OUTPUT_PATH.joinpath('original.csv'), index=False)
#%%