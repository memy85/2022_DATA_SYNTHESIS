import pandas as pd
from pathlib import Path
import os
import pickle
import argparse

cur_file = Path(__file__).absolute()
working_dir = cur_file.parent
parent_dir = working_dir.parent.parent
data_dir = parent_dir.joinpath('data')

os.sys.path.append(parent_dir.as_posix())

from src.MyModule.utils import *
config = load_config()

#%%

org_dir = data_dir.joinpath('processed/preprocess_1')
syn_dir = data_dir.joinpath('processed/2_produce_data/synthetic_restore')
output_path = data_dir.joinpath('processed/3_evaluate_data')

def parse_argument():
    parser = argparse.ArgumentParser()
    parser.add_argument('--age', default = 50, type = int)
    args = parser.parse_args()
    return args

if __name__ == "__main__" :

    args = parse_argument()
    
    sel_col = ['BSPT_SEX_CD','BSPT_IDGN_AGE','BSPT_FRST_DIAG_NM','BSPT_STAG_VL','RLPS','DEAD','BPTH_SITE_CONT','BPTH_CELL_DIFF_NM','OPRT_CLCN_OPRT_KIND_NM']


    org = pd.read_pickle(org_dir.joinpath('original_50.pkl'))
    stdl_org = org.loc[:,sel_col]

    binded_org = org[org.columns[list(org.columns).index('SGPT_PATL_STAG_VL'):]]
    binded_org=binded_org.drop('MLPT_ACPT_YMD',axis=1)

    matched_org = pd.concat([stdl_org,binded_org],axis=1)

    epsilons = config['epsilon']

    for epsilon in epsilons :
        syn = pd.read_csv(syn_dir.joinpath('Synthetic_data_epsilon{}_{}.csv'.format(epsilon, args.age)),encoding='cp949',index_col = 0)

        stdl_syn = syn.loc[:,sel_col]

        syn_col = list(syn.columns)
        new_syn_col =[]
        for col in syn_col:
            if 'TIME_DIFF' not in col:
                new_syn_col.append(col)

        binded_syn = syn[(new_syn_col[new_syn_col.index('SGPT_PATL_STAG_VL'):])]
        matched_syn = pd.concat([stdl_syn,binded_syn],axis=1)
        matched_syn.columns = matched_org.columns

        with open(output_path.joinpath(f'matched_syn_{epsilon}_{args.age}.pkl'), 'wb') as f:
            pickle.dump(matched_org, f, pickle.HIGHEST_PROTOCOL)
    
    with open(output_path.joinpath(f'matched_org_{args.age}.pkl'), 'wb') as f:
        pickle.dump(matched_org, f, pickle.HIGHEST_PROTOCOL)