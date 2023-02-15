import pandas as pd
from pathlib import Path
import os
import pickle

cur_file = Path(__file__).absolute()
print(cur_file)
working_dir = cur_file.parent
print('working directory {}'.format(working_dir))
parent_dir = working_dir.parent.parent
data_dir = parent_dir.joinpath('data')

org_dir = data_dir.joinpath('processed/preprocess_1')
syn_dir = data_dir.joinpath('processed/restored')

output_path = data_dir.joinpath('processed/3_evaluate_data')

org = pd.read_pickle(org_dir.joinpath('original_50.pkl'))
syn = pd.read_csv(syn_dir.joinpath('Synthetic_data_epsilon10000.csv'),encoding='cp949',index_col = 0)

sel_col = ['BSPT_SEX_CD','BSPT_IDGN_AGE','BSPT_FRST_DIAG_NM','BSPT_STAG_VL','RLPS','DEAD',
          'BPTH_SITE_CONT','BPTH_CELL_DIFF_NM','OPRT_CLCN_OPRT_KIND_NM']

stdl_org = org.loc[:,sel_col]
stdl_syn = syn.loc[:,sel_col]

syn_col = list(syn.columns)
new_syn_col =[]
for col in syn_col:
    if 'TIME_DIFF' not in col:
        new_syn_col.append(col)

binded_syn = syn[(new_syn_col[new_syn_col.index('SGPT_PATL_STAG_VL'):])]
binded_org = org[org.columns[list(org.columns).index('SGPT_PATL_STAG_VL'):]]
binded_org=binded_org.drop('MLPT_ACPT_YMD',axis=1)

matched_org = pd.concat([stdl_org,binded_org],axis=1)
matched_syn = pd.concat([stdl_syn,binded_syn],axis=1)
matched_syn.columns = matched_org.columns

with open(output_path.joinpath('matched_org.pickle'), 'wb') as f:
    pickle.dump(matched_org, f, pickle.HIGHEST_PROTOCOL)
with open(output_path.joinpath('matched_syn.pickle'), 'wb') as f:
    pickle.dump(matched_org, f, pickle.HIGHEST_PROTOCOL)