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
OUTPUT_PATH = PROJ_PATH.joinpath('data/processed/4_evaluate/make_death')

if not OUTPUT_PATH.exists() :
    OUTPUT_PATH.mkdir(parents=True)


# %%
def load_file_epsilon(epsilon):
    file_name = "CLRC_PT_BSNF"
    return read_file(INPUT_PATH, f'{file_name}_{epsilon}.csv')

def load_s1(epsilon):
    return read_file(PROJ_PATH.joinpath(f'data/processed/2_restore/restore_to_s1'), f'S1_{epsilon}.pkl')

#%%

data = load_file_epsilon(1)
original_data =pd.read_excel(PROJ_PATH.joinpath('data/raw/CLRC_PT_BSNF.xlsx'))
s1 = load_s1(1)


#%%


#%%
def main():
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--epsilon', '--e', help="epsilon values")
    args = parser.parse_args()
    print(f'epsilon is {args.epsilon}')
    
    data = load_file_epsilon(args.epsilon)
    original_data =pd.read_excel(PROJ_PATH.joinpath('data/raw/CLRC_PT_BSNF.xlsx'))
    s1 = load_s1(args.epsilon)
    
    #%% first diag ymd 만들기
    frst_diag_yes = data[['PT_SBST_NO','TIME','BSPT_FRST_DIAG_YMD']].dropna().drop(columns = "BSPT_FRST_DIAG_YMD")
    frst_diag_yes= frst_diag_yes.groupby('PT_SBST_NO', as_index=False)['TIME'].min()
    frst_diag_yes = frst_diag_yes.rename(columns = {'TIME':'BSPT_FRST_DIAG_YMD'})

    yes_patients = frst_diag_yes.PT_SBST_NO

    frst_diag_no = data[['PT_SBST_NO','TIME','BSPT_FRST_DIAG_YMD']].copy()
    frst_diag_no = frst_diag_no[~frst_diag_no.PT_SBST_NO.isin(yes_patients)]
    frst_diag_no = frst_diag_no.groupby('PT_SBST_NO', as_index=False)['TIME'].min()
    frst_diag_no = frst_diag_no.rename(columns = {'TIME':'BSPT_FRST_DIAG_YMD'})

    frst_diag_data = pd.concat([frst_diag_yes, frst_diag_no])
    data = data.drop(columns = ['TIME','BSPT_FRST_DIAG_YMD']).drop_duplicates()
    data = data.sort_values(by='PT_SBST_NO')
    data = data.dropna(subset=['BSPT_SEX_CD','BSPT_FRST_DIAG_CD']).copy()
    data = data.merge(frst_diag_data)

    diag_name = original_data[['BSPT_FRST_DIAG_CD','BSPT_FRST_DIAG_NM']].drop_duplicates()
    data = data.merge(diag_name, how='left')
    
    data =  data.drop_duplicates()
    num_dup = data.duplicated('PT_SBST_NO').sum()
    print(f'Number of duplicated patients : {num_dup}')

    #%% operation data
    operation_data = s1.filter(regex='OPRT|TIME|PT_SBST_NO').dropna(subset=['OPRT_NFRM_OPRT_CLCN_OPRT_KIND_CD','OPRT_NFRM_OPRT_CURA_RSCT_CD'], how='all')

    first_oprt = operation_data.groupby(['PT_SBST_NO'], as_index=False)['TIME'].min()
    first_oprt = first_oprt.rename(columns= {'TIME':'BSPT_FRST_OPRT_YMD'})
    data = data.merge(first_oprt, how='left')

    #%%
    num_dup = data.duplicated('PT_SBST_NO').sum()
    print(f'Number of duplicated patients : {num_dup}')


    #%%
    rd_data = s1.filter(regex='TRTM_RD|TIME|PT_SBST_NO').dropna(subset=['TRTM_RD_RDT'], how='all')
    first_rd = rd_data.groupby(['PT_SBST_NO'], as_index=False)['TIME'].min()
    first_rd = first_rd.rename(columns = {'TIME':'BSPT_FRST_RDT_STRT_YMD'})
    data = data.merge(first_rd, how='left')

    #%% death
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

    #%%
    data['BSPT_OPRT'] = ~data['BSPT_FRST_OPRT_YMD'].isna()*1

    #%%
    data['TRTM_RD_RDT'] = ~data['BSPT_FRST_RDT_STRT_YMD'].isna()*1

    #%%
    data['CENTER_LAST_VST_YMD'] = data['CENTER_LAST_VST_YMD'].fillna(data['BSPT_DEAD_YMD'])
    data['OVRL_DAYS'] = (data.CENTER_LAST_VST_YMD - data.BSPT_FRST_DIAG_YMD).dt.days

    data['DEAD'] = ~data['BSPT_DEAD_YMD'].isna()*1


    # %%
    comparison_data = data[['PT_SBST_NO','BSPT_SEX_CD','BSPT_FRST_DIAG_CD','BSPT_IDGN_AGE','BSPT_STAG_VL','BSPT_T_STAG_VL','BSPT_N_STAG_VL','BSPT_M_STAG_VL','BSPT_OPRT','TRTM_RD_RDT','OVRL_DAYS','DEAD']].copy()

    comparison_data.to_csv(OUTPUT_PATH.joinpath(f'comparison_data_{args.epsilon}.csv'), index=False)


    #%%
    ori = original_data[['PT_SBST_NO','BSPT_SEX_CD','BSPT_FRST_DIAG_CD','BSPT_IDGN_AGE','BSPT_DEAD_YMD','CENTER_LAST_VST_YMD','OVRL_SRVL_DTRN_DCNT','BSPT_STAG_VL','BSPT_T_STAG_VL','BSPT_N_STAG_VL','BSPT_M_STAG_VL','BSPT_FRST_RDT_STRT_YMD','BSPT_FRST_OPRT_YMD']]
    #%%

    ori['DEAD'] = ~ori['BSPT_DEAD_YMD'].isna()*1
    ori['TRTM_RD_RDT'] = ~ori['BSPT_FRST_RDT_STRT_YMD'].isna()*1
    ori['BSPT_OPRT'] = ~ori['BSPT_FRST_OPRT_YMD'].isna()*1

    #%%
    ori = ori.rename(columns = {"OVRL_SRVL_DTRN_DCNT":"OVRL_DAYS"})

    ori = ori[['PT_SBST_NO','BSPT_SEX_CD','BSPT_FRST_DIAG_CD','BSPT_IDGN_AGE','BSPT_STAG_VL','BSPT_T_STAG_VL','BSPT_N_STAG_VL','BSPT_M_STAG_VL','BSPT_OPRT','TRTM_RD_RDT','OVRL_DAYS','DEAD']].copy()

    ori.to_csv(OUTPUT_PATH.joinpath('original.csv'), index=False)
#%%

if __name__ == "__main__" : 
    main()
    