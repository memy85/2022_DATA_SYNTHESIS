#%%
from pathlib import Path
import os, sys
import argparse

# project_path = Path(__file__).absolute().parents[2]
project_path = Path().cwd()
os.sys.path.append(project_path.as_posix())

#%%
from src.MyModule.utils import *
import pandas as pd
import numpy as np

config = load_config()
input_path = get_path("data/processed/2_produce_data/synthetic_decoded/")
output_path = get_path("data/processed/4_results/")
figure_path = get_path("figures/")
age = 50

cols = [
    "BSPT_IDGN_AGE",
    "BSPT_SEX_CD",
    "BSPT_FRST_DIAG_NM",
    "SGPT_PATL_T_STAG_VL",
    "SGPT_PATL_N_STAG_VL",
    "BSPT_STAG_VL",
    "OPRT_CLCN_OPRT_KIND_NM",
    "MLPT_KRES_RSLT_NM",
    "IMPT_HM1E_RSLT_NM",
    "IMPT_HS2E_RSLT_NM",
    "IMPT_HS6E_RSLT_NM",
    "IMPT_HP2E_RSLT_NM",
    "ADJ_CNT",
    "LNE_CHEMO",
    "DEAD",
    "OVRL_SURV",
    "ADJ_CNT"
]

bin_dict = {
    "BSPT_IDGN_AGE" : [0,20,30,40,50],
    "OVRL_SURV" : np.arange(0,365*10,1000),
}

original_path = get_path(f'data/processed/3_evaluate_data/matched_org_{age}.pkl')

synthetic_data_path_list = []
for epsilon in config['epsilon'] : 
    synthetic_path = get_path(f'data/processed/3_evaluate_data/matched_syn_{epsilon}_{age}.pkl')
    synthetic_data_path_list.append(synthetic_path)

original = pd.read_pickle(original_path) 
original = original[cols].copy()

def load_pickle(path, col=None) :
    df = pd.read_pickle(path) 
    # preprocess some information
    if col is not None :
        return df[col].copy()
    return df.copy()

synthetic_data_list = list(map(lambda x : load_pickle(x, col = cols), synthetic_data_path_list))

#%%
synthetic = synthetic_data_list[-1]

#%%

