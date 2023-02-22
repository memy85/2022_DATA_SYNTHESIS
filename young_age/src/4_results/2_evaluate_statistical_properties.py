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
from tableone import TableOne
from lifelines import KaplanMeierFitter
from lifelines.utils import median_survival_times
from matplotlib.pyplot as plt

config = load_config()
input_path = get_path("data/processed/2_produce_data/synthetic_decoded/")
output_path = get_path("data/processed/4_results/")
figure_path = get_path("figures/")

def convert_to_5years(data) :
    new_data = data.copy()
    msk = (data['OVRL_SURV'] > 12*5) & (data['DEAD'] == 1)

    new_data.loc[msk,"DEAD"] = 0
    msk = (data['OVRL_SURV'] > 12*5)
    new_data.loc[msk,"OVRL_SURV"] = 12*5

    return new_data

def argument_parse() :
    parser = argparse.ArgumentParser()
    parser.add_argument("--age", default = 50, type = int)
    args = parser.parse_args()
    return args

def main() :
    args = argument_parse()
    # age = args.age
    age = 50

    cols = [
        "BSPT_IDGN_AGE",
        "BSPT_SEX_CD",
        "BSPT_FRST_DIAG_NM",
        "SGPT_PATL_T_STAG_VL",
        "SGPT_PATL_N_STAG_VL",
        "BSPT_STAG_VL",
        "OPRT",
        "MLPT_KRES_RSLT_NM",
        "IMPT_HM1E_RSLT_NM",
        "IMPT_HS2E_RSLT_NM",
        "IMPT_HS6E_RSLT_NM",
        "IMPT_HP2E_RSLT_NM",
        "DEAD",
        "OVRL_SURV",
        "LNE_CHEMO",
        "ADJ_CNT"
    ]

    original_path = get_path(f'data/processed/3_evaluate_data/matched_org_{age}.pkl')

    synthetic_data_path_list = []
    for epsilon in config['epsilon'] : 
        synthetic_path = get_path(f'data/processed/3_evaluate_data/matched_syn_{epsilon}_{age}.pkl')
        synthetic_data_path_list.append(synthetic_path)

    original = pd.read_pickle(original_path) 
    original['OPRT'] = ~original.filter(like = "OPRT").isna()

    original = original[cols].copy()

    def load_pickle(path, col=None) :
        df = pd.read_pickle(path) 
        # preprocess some information
        if col is not None :
            df['OPRT'] = ~df.filter(like = "OPRT").isna()
            return df[col].copy()
        return df.copy()

    synthetic_data_list = list(map(lambda x : load_pickle(x, col = cols), synthetic_data_path_list))

    synthetic = synthetic_data_list[-1]

    synthetic['BSPT_IDGN_AGE'] = pd.cut(synthetic['BSPT_IDGN_AGE'], bins = [0,20,30,40,50], right = False, include_lowest = True)
    synthetic['LNE_CHEMO'] = pd.cut(synthetic['LNE_CHEMO'], bins = [0,1,2,3,4,9], right = False, include_lowest = True)


    categorical = ["BSPT_SEX_CD", "BSPT_FRST_DIAG_NM", "SGPT_PATL_T_STAG_VL", "SGPT_PATL_N_STAG_VL", 
                   "BSPT_STAG_VL", "MLPT_KRES_RSLT_NM", "IMPT_HM1E_RSLT_NM", "IMPT_HS2E_RSLT_NM", 
                   "IMPT_HS6E_RSLT_NM", "IMPT_HP2E_RSLT_NM", "DEAD","LNE_CHEMO",
                   "ADJ_CNT", "BSPT_IDGN_AGE", "OPRT"]

    nonnormal = ['OVRL_SURV']

    mytable = TableOne(synthetic, cols, categorical, None, nonnormal)

    # mytable.to_html(output_path.joinpath('mytable.html'))
    return synthetic, original

def save_overall_survival(data) :
    data = data[['BSPT_STAG_VL','OVRL_SURV','DEAD']].copy()
    data = data.dropna(subset = 'OVRL_SURV')

    # convert to months
    data['OVRL_SURV'] = round(data['OVRL_SURV'] / 30)

    data = convert_to_5years(data)

    ovr_srv_list = []
    for stage in data['BSPT_STAG_VL'].unique() :
        df = data[data.BSPT_STAG_VL == stage].copy()

        kmf = KaplanMeierFitter()
        kmf.fit(df['OVRL_SURV'], df['DEAD'])

        ci = median_survival_times(kmf.confidence_interval_)
        median = kmf.median_survival_time_
        book = {"stage" : stage, 'median os' : median, 'CI' : ci}
        ovr_srv_list.append(book)

    df = pd.DataFrame(ovr_srv_list)
    df.to_csv(output_path.joinpath("survival.csv"), index=False)

if __name__  == "__main__" : 
    synthetic, original = main()
    save_overall_survival(synthetic)

