#%%

from pathlib import Path
import os, sys

# project_path = Path(__file__).absolute().parents[2]
project_path = Path().cwd()
os.sys.path.append(project_path.as_posix())
#%%
from src.MyModule.utils import *

config = load_config()
input_path = get_path("data/processed/3_evaluate_data")
figure_path = get_path("figures/")
ouput_path = get_path("data/processed/3_evaluate_data/")

#%%
import pandas as pd
import numpy as np
import argparse

import matplotlib.pyplot as plt
import matplotlib as mpl
import scienceplots

#%%

class CorrelationChecker:

    def __init__(self, data1, data2):

        self.data1 = data1.copy()
        self.data2 = data2.copy()

        assert self.check(), "Two columns must be the same"


    def check(self) :
        self.data1columns = set(self.data1.columns.tolist())
        self.data2columns = set(self.data1.columns.tolist())

        if len(self.data1columns - self.data2columns) < 1 :
            return True
        else :
            return False
    
    def factorize(self, column, dtype) :
        if dtype == 'object' :
            codes, _  = pd.factorize(column)
            return pd.Series(codes)
        else :
            return column
    
    def process4correlation(self):
        self.data1["origin"] = 'data1'
        self.data2["origin"] = 'data2'

        new_data = pd.concat([self.data1, self.data2], axis=0, ignore_index=True)
        new_data = new_data.apply(lambda x : self.factorize(x, x.dtype.__str__()))

        self.data1_factorized = new_data[new_data.origin == 0].drop(columns = 'origin')
        self.data2_factorized = new_data[new_data.origin == 1].drop(columns = 'origin')
        return self.data1_factorized, self.data2_factorized

    def calculate_correlation_diff(self) :

        self.process4correlation()
        corrmatrix1 = self.data1_factorized.corr().values
        corrmatrix2 = self.data2_factorized.corr().values
        
        self.difference = abs(corrmatrix1 - corrmatrix2)
        return self.difference
         
#%%

cols = [
    "BSPT_IDGN_AGE",
    "BSPT_SEX_CD",
    "BSPT_FRST_DIAG_NM",
    "SGPT_PATL_T_STAG_VL",
    "SGPT_PATL_N_STAG_VL",
    "SGPT_PATL_STAG_VL",
    "BSPT_STAG_VL",
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

def load_pickle(path) :
    df = pd.read_pickle(path) 
    return df[cols].copy()

def argument_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--age', default = 50, type = int)
    args = parser.parse_args()
    return args
#%%
def main() : 
    args = argument_parse()

    original_path = get_path(f'data/processed/3_evaluate_data/matched_org_{args.age}.pkl')

    synthetic_data_path_list = []
    for epsilon in config['epsilon'] : 
        synthetic_path = get_path(f'data/processed/3_evaluate_data/matched_syn_{epsilon}_{args.age}.pkl')
        synthetic_data_path_list.append(synthetic_path)

    original = pd.read_pickle(original_path) 
    original = original[cols].copy()

    synthetic_data_list = list(map(load_pickle, synthetic_data_path_list))

    for idx, epsilon in enumerate(config['epsilon']) :

        processor = CorrelationChecker(original, synthetic_data_list[idx]) 
        plot_correlation(processor, epsilon, args.age)

    
            

#%%

def plot_correlation(processor, epsilon, age) :
    diff = processor.calculate_correlation_diff()

    cols = list(processor.data1columns)
    fig, ax = plt.subplots(figsize = (12,12))

    # im = ax.imshow(diff, cmap='YlGn')
    plt.pcolor(diff, cmap='YlGn', vmin = 0, vmax=0.8)
    # cbar = ax.figure.colorbar(im, ax = ax, cmap='YlGn')
    plt.colorbar( ax = ax )

    plt.xticks(np.arange(diff.shape[1]), labels = cols, rotation=90)
    plt.yticks(np.arange(diff.shape[1]), labels = cols)
    plt.tick_params(axis = 'both', labelsize = 7)

    plt.title("Correlation Difference, $\epsilon =$ {}".format(epsilon))
    plt.savefig(figure_path.joinpath(f"correlation_{epsilon}_{age}.png"), dpi=300)
    plt.show()

#%%
if __name__ == "__main__" :

    main() 
    
#%%

#age = 50

#original_path = get_path(f'data/processed/3_evaluate_data/matched_org_{age}.pkl')

#synthetic_data_path_list = []
#for epsilon in config['epsilon'] : 
#    synthetic_path = get_path(f'data/processed/3_evaluate_data/matched_syn_{epsilon}_{age}.pkl')
#    synthetic_data_path_list.append(synthetic_path)

#original = pd.read_pickle(original_path) 
#original = original[cols].copy()

#synthetic_data_list = list(map(load_pickle, synthetic_data_path_list))

#for idx, epsilon in enumerate(config['epsilon']) :
#    if epsilon != 10000 :
#        continue

#    processor = CorrelationChecker(original, synthetic_data_list[idx]) 
#    plot_correlation(processor, epsilon, age)
##%%
#pd.DataFrame(processor.calculate_correlation_diff()).values()
