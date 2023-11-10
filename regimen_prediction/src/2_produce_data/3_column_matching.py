
#%%

import pandas as pd
import numpy as np
from pathlib import Path
import os
import pickle
import argparse

project_path = Path(__file__).absolute().parents[2]
# project_path = Path().cwd()

os.sys.path.append(project_path.as_posix())

from src.MyModule.utils import *
config = load_config()

project_path = get_path('')
data_dir = project_path.joinpath('data')

#%%

def parse_argument():
    parser = argparse.ArgumentParser()
    parser.add_argument('--age', default = 50, type = int)
    parser.add_argument('--random_seed', default = 0, type = int)
    args = parser.parse_args()
    return args

def count_adjuvant(data) : 
    '''
    count if a patient had adjuvant chemotherapy
    '''
    first_check = data["CASB_CSTR_PRPS_NM_1"].apply(lambda x : x == "['Adjuvant']")
    second_check = data[['CASB_CSTR_PRPS_NM_1', 'CASB_CSTR_PRPS_NM_2']].apply(
        lambda x : (x["CASB_CSTR_PRPS_NM_1"] == "['Neo-adjuvant']") & (x["CASB_CSTR_PRPS_NM_2"] == "['Adjuvant']"), axis=1)
    return first_check + second_check
#%%


if __name__ == "__main__" :

    args = parse_argument()
    age = args.age
    random_seed = args.random_seed
    #%%
    # age = 50
    # random_seed = 0

    org_dir = data_dir.joinpath(f'processed/seed{random_seed}/1_preprocess')
    syn_dir = data_dir.joinpath(f'processed/seed{random_seed}/2_produce_data/synthetic_restore')
# syn_dir = data_dir.joinpath('processed/no_bind/restored/seed0/synthetic_restore')
    output_path = data_dir.joinpath(f'processed/seed{random_seed}/3_evaluate_data')
# output_path = data_dir.joinpath('processed/no_bind/matching')
    if not output_path.exists() : 
        output_path.mkdir(parents = True)

#%%
    sel_col = ['BSPT_SEX_CD','BSPT_IDGN_AGE','BSPT_FRST_DIAG_NM','BSPT_STAG_VL',
               'RLPS','DEAD','BPTH_SITE_CONT','BPTH_CELL_DIFF_NM','OPRT_CLCN_OPRT_KIND_NM']

    org = pd.read_pickle(org_dir.joinpath(f'original_{args.age}.pkl'))
    # org = pd.read_pickle(org_dir.joinpath(f'original_{age}.pkl'))
    # org = pd.read_csv('/home/wonseok/projects/2022_DATA_SYNTHESIS/young_age/data/processed/no_bind/encoded_D0_to_syn_50.csv')

#%%
    stdl_org = org.loc[:,sel_col + ['OVRL_SRVL_DTRN_DCNT']]

    binded_org = org[org.columns[list(org.columns).index('SGPT_PATL_STAG_VL'):]]
    # binded_org=binded_org.drop('MLPT_ACPT_YMD',axis=1)

    matched_org = pd.concat([stdl_org,binded_org],axis=1)

    matched_org = matched_org.rename(columns = {"OVRL_SRVL_DTRN_DCNT" : "OVRL_SURV"})
    matched_org['LNE_CHEMO'] = 8 - matched_org.filter(like = 'CSTR_REGN').isna().sum(axis=1)

    matched_org['SGPT_PATL_T_STAG_VL'] = matched_org['SGPT_PATL_T_STAG_VL'].replace(999, np.nan)
    matched_org['SGPT_PATL_T_STAG_VL'] = matched_org['SGPT_PATL_T_STAG_VL'].replace('999',np.nan)

#%%
    
    epsilons = config['epsilon']

#%%
    for epsilon in epsilons :
        syn = pd.read_csv(syn_dir.joinpath('Synthetic_data_epsilon{}_{}.csv'.format(epsilon, args.age)),encoding='cp949',index_col = 0)
        # syn = pd.read_csv(syn_dir.joinpath('Synthetic_data_epsilon{}_{}.csv'.format(epsilon, 50)),encoding='cp949',index_col = 0)
        # break
#%%
        stdl_syn = syn.loc[:,sel_col + ['OVR_SURV']]
        stdl_syn.rename(columns = {"OVR_SURV": "OVRL_SURV"})
        #
        syn_col = list(syn.columns)
        new_syn_col =[]
    
        for col in syn_col:
            if 'TIME_DIFF' not in col:
                new_syn_col.append(col)

#%%
        # binded_syn = syn[(new_syn_col[new_syn_col.index('OVR_SURV') + 1:])]
        binded_syn = syn.loc[:,new_syn_col].copy()
        binded_syn = binded_syn.loc[:, "SGPT_PATL_STAG_VL":]
        matched_syn = pd.concat([stdl_syn, binded_syn], axis=1)

        matched_syn['LNE_CHEMO'] = 8 - matched_syn.filter(like = 'CSTR_REGN').isna().sum(axis=1)
        matched_syn = matched_syn.reset_index()

        print(len(matched_syn.columns.tolist()), len(matched_org.columns.tolist()))
#%%
        matched_syn.columns = matched_org.columns
        matched_syn['ADJ_CNT'] = count_adjuvant(matched_syn)

        with open(output_path.joinpath(f'matched_syn_{epsilon}_{args.age}.pkl'), 'wb') as f:
            pickle.dump(matched_syn, f, pickle.HIGHEST_PROTOCOL)
    
    matched_org['ADJ_CNT'] = count_adjuvant(matched_org)

    with open(output_path.joinpath(f'matched_org_{args.age}.pkl'), 'wb') as f:
        pickle.dump(matched_org, f, pickle.HIGHEST_PROTOCOL)
