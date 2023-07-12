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
        self.data2columns = set(self.data2.columns.tolist())

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

def decode(whole_encoded_df, tables, bind_data_columns):
    
    np_encoded = np.array(whole_encoded_df)
    np_encoded = np_encoded.astype(str)
    restored = pd.DataFrame()

    for k in range(len(np_encoded.transpose())):
        temp1 = []
        for i in np_encoded.transpose()[k]:
            temp2=[]
            a =''
            for j in range(len(i)):
                a+=i[j]
                if((j+1)%3==0):
                    temp2.append(a)
                    if(len(a)!=3):
                        print('error')
                    a=''
            
            temp1.append(temp2)
        sep_df = pd.DataFrame(temp1)
        restored = pd.concat([restored,sep_df],axis=1)
        
    cols = []
    for head in tables:
        columns = list(filter(lambda x : head in x, bind_data_columns))
        for col in columns:
            cols.append(col)

    return restored
         
def prepare_original_data(random_seed, age) :

    original_data_path = get_path(f"data/processed/seed{random_seed}/1_preprocess/encoded_D0_{age}.csv")
    data = pd.read_csv(original_data_path)

    bind_columns = pd.read_pickle(project_path.joinpath(f"data/processed/seed{random_seed}/1_preprocess/bind_columns_{age}.pkl"))

    tables= []
    for col in bind_columns:
            tables.append('_'.join(col.split('_')[0:1]))
    try:
        data = data.drop('Unnamed: 0', axis=1)
    except:
        pass
    data = data.astype(str)

    # for col in data.iloc[:,11:]:
    #     data[col] = data[col].str.replace('r','')
    #     
    # decoded = decode(data.iloc[:,11:], tables, bind_columns)
    # decoded.columns = bind_columns

    data.reset_index(drop=True, inplace=True)
    
    # data = pd.concat([data.iloc[:,:11],decoded],axis=1)
    data = data.rename(columns = {'RLPS DIFF' : 'RLPS_DIFF'})
    data = data.drop(columns = "PT_SBST_NO")
    data['BSPT_STAG_VL'] = data['BSPT_STAG_VL'].astype('float').astype('object')

    return data

def prepare_synthetic_data(random_seed, age) :

    epsilons = config['epsilon']
    synthetic_data_list = []

    bind_columns = pd.read_pickle(project_path.joinpath(f"data/processed/seed{random_seed}/1_preprocess/bind_columns_{age}.pkl"))

    tables= []
    for col in bind_columns:
            tables.append('_'.join(col.split('_')[0:1]))

    synthetic_path = get_path(f"data/processed/seed{random_seed}/2_produce_data")
    for epsilon in epsilons:
        syn = pd.read_csv(synthetic_path.joinpath(f'S0_mult_encoded_{epsilon}_{age}.csv'))

        try:
            syn = syn.drop('Unnamed: 0', axis=1)
        except:
            pass
        syn = syn.astype(str)

        for col in syn.iloc[:,11:]:
            syn[col] =syn[col].str.replace('r','')
            
        decoded = decode(syn.iloc[:,11:], tables, bind_columns)
        decoded.columns = bind_columns
        
        syn = pd.concat([syn.iloc[:,:11],decoded],axis=1)
        syn = syn.rename(columns = {'RLPS DIFF' : 'RLPS_DIFF'})
        syn = syn.drop(columns = 'PT_SBST_NO')
        syn['BSPT_STAG_VL'] = syn['BSPT_STAG_VL'].astype('float').astype('object')

        synthetic_data_list.append(syn)
    return synthetic_data_list

def calculate_correlation_diff_for_all_variables() :
    args = argument_parse()

    original = prepare_original_data(args.random_seed, args.age)
    synthetic_data_list = prepare_synthetic_data(args.random_seed, args.age)
    
    for idx, epsilon in enumerate(config['epsilon']) :
        processor = CorrelationChecker(original, synthetic_data_list[idx]) 
        plot_correlation_for_all_variables(processor, epsilon, args.age)

def calculate_correlation_diff_for_all_variables_no_bind() :
    args = argument_parse()

    no_bind_path = project_path.joinpath("data/processed/no_bind")
    original = pd.read_csv(no_bind_path.joinpath('encoded_D0_to_syn_50.csv'))
    original = original.drop(columns = 'PT_SBST_NO')

    synthetic_data_list = [pd.read_csv(no_bind_path.joinpath(f'seed0/S0_mult_encoded_{epsilon}_{50}.csv'))
                           for epsilon in config['epsilon']]

    def drop_column(data, columns):
        data = data.drop(columns = columns)
        return data

    synthetic_data_list = list(map(lambda x : drop_column(x, 'PT_SBST_NO'), synthetic_data_list))

    for idx, epsilon in enumerate(config['epsilon']) :
        processor = CorrelationChecker(original, synthetic_data_list[idx]) 
        figure_name = f"correlation_all_{epsilon}_{args.age}_no_bind.png"
        plot_correlation_for_all_variables(processor, epsilon, args.age, figure_name)



def plot_correlation_for_all_variables(processor, epsilon, age, figure_name = None) :
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

    if figure_name is None : 
        figure_name = f"correlation_all_{epsilon}_{age}.png"

    plt.savefig(figure_path.joinpath(figure_name), dpi=300)
    plt.show()

#%%

def load_pickle(path) :
    df = pd.read_pickle(path) 
    return df[cols].copy()

def argument_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--age', default = 50, type = int)
    parser.add_argument('--random_seed', default = 0, type = int)
    args = parser.parse_args()
    return args

#%%
def main() : 
    args = argument_parse()

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


    original_path = get_path(f'data/processed/seed{args.random_seed}/3_evaluate_data/matched_org_{args.age}.pkl')

    synthetic_data_path_list = []
    for epsilon in config['epsilon'] : 
        synthetic_path = get_path(f'data/processed/seed{args.random_seed}/3_evaluate_data/matched_syn_{epsilon}_{args.age}.pkl')
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

    # main() 
    calculate_correlation_diff_for_all_variables()
    # calculate_correlation_diff_for_all_variables_no_bind()

    
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
