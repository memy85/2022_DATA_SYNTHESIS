#%%

import scienceplots
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
from pathlib import Path

from src.MyModule.distribution_comparison import *
from src.MyModule.ml_function import *
from src.MyModule.utils import *

# project_path = Path(__file__).absolute().parents[2]
project_path = Path().cwd()

os.sys.path.append(project_path.as_posix())
print(f" this is project path : {project_path} ")

#%% settings

age = 50

#%% path settings

config = load_config()

input_path = get_path("data/processed/2_produce_data/synthetic_decoded")

synthetic_path = input_path.joinpath(f"Synthetic_data_epsilon10000_{age}.csv")

# synthetic_data_path_list = [input_path.joinpath(
#     f"Synthetic_data_EPSILON{EPS}_{AGE}.CSV") for eps in config['epsilon']]

ori_path = get_path(f"data/processed/preprocess_1/original_{age}.pkl")

# testset_path = get_path(f"data/processed/preprocess_1/test_{age}.pkl")

output_path = get_path("data/processed/3_evaluate_data/")
figure_path = project_path.joinpath('figures/')

if not output_path.exists():
    output_path.mkdir(parents=True)

#%% matched_data

matched_syn_path = output_path.joinpath('matched_syn_10000_50.pkl')
matched_ori_path = output_path.joinpath('matched_org_50.pkl')

original = pd.read_pickle(matched_ori_path)
synthetic = pd.read_pickle(matched_syn_path)


#%%

# synthetic_data_list = list(
#     map(lambda x: pd.read_csv(x), synthetic_data_path_list))
# train_ori = pd.read_pickle(train_ori_path)
# test = pd.read_pickle(testset_path)

#%%

# test.drop(["PT_SBST_NO"], axis=1, inplace=True)
# train_ori.drop(["PT_SBST_NO"], axis=1, inplace=True)
# train_ori = train_ori.rename(columns={"RLPS DIFF": "RLPS_DIFF"})
# synthetic_data_list = list(map(lambda x: x.drop(
#     ["PT_SBST_NO", "Unnamed: 0"], axis=1), synthetic_data_list))

#%%

# def test_death_ratio(data):
#     value_counts = data["DEAD"].value_counts()
#     dead, alive = value_counts[1], value_counts[0]
#     return dead, alive 


#%% load data

original = pd.read_pickle(ori_path)
synthetic = pd.read_csv(synthetic_path)


#%% compare marginal 

#%% compare death, relapse, stage, age
import matplotlib.pyplot as plt
import matplotlib as mpl

plt.style.use('science')


def compare_two_columns(column_name) :

    fig, ax = plt.subplots(nrows=1, ncols=2, figsize = (4,2))

    tempdf_ori = original[[column_name]].copy()
    tempdf_syn = synthetic[[column_name]].copy()


    tempdf_ori.hist( ax= ax[0], label = 'original')
    ax[0].set_title('original')
    tempdf_syn.hist( ax = ax[1], label = 'synthetic')
    ax[1].set_title('synthetic')
    fig.suptitle(column_name)
    plt.tight_layout()
    plt.savefig(figure_path.joinpath(f'{column_name}_marginal.png'), dpi = 300)
    plt.show()

#%%

compare_two_columns('BSPT_STAG_VL')
#%%
compare_two_columns("DEAD")

#%%
compare_two_columns("RLPS")

#%%
compare_two_columns("BSPT_IDGN_AGE")
#%%
synthetic.filter(like ='SGPT')

#%%
compare_two_columns('IMPT_HS6E_RSLT_NM')

#%%
original['SGPT_PATL_T_STAG_VL'].unique()

#%%
synthetic['SGPT_PATL_T_STAG_VL'].unique()

#%%
original["IMPT_HS6E_RSLT_NM"].value_counts()

#%%
plt.show()

#%%
synthetic["IMPT_HS6E_RSLT_NM"].hist()
plt.show()

#%%

