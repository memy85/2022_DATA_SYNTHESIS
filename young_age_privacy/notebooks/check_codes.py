#%%
import os, sys
from pathlib import Path

project_path = Path().cwd()
os.sys.path.append(project_path.as_posix())

import pandas as pd 
import pickle
import matplotlib.pyplot as plt

from src.MyModule.utils import *
#%% check original data before input to bayesian network

# path = get_path("data/processed/preprocess_1/unmodified_D0_50.pkl")
# df = pd.read_pickle(path)


#%%
mylist = []
for cols in df.columns:
    mylist.append(len(df[cols].unique()))

#%%

mylist = pd.Series(mylist)


#%%
mylist.hist(bins=100)
plt.show()

#mylist > 200%%

mylist[mylist > 200]
# the maximum categoricals should be 200

#%% check bayesian network input data


path = get_path("data/processed/preprocess_1/encoded_D0_to_syn_50.csv")
df = pd.read_csv(path)

#%%
len(df.columns)


#%%

df



#%%




config = load_config()
project_path = Path(config["project_path"])
restore_path = get_path("data/processed/2_produce_data/synthetic_restore")
decode_path = get_path("data/processed/2_produce_data/synthetic_decoded")

#%%

pd.read_csv(restore_path.joinpath("Synthetic_data_epsilon10000.csv"))



#%%
