#%%
import os, sys
from pathlib import Path

project_path = Path().cwd()
os.sys.path.append(project_path.as_posix())

import pandas as pd 
import pickle

#%%

from src.MyModule.utils import *

config = load_config()
project_path = Path(config["project_path"])
restore_path = get_path("data/processed/2_produce_data/synthetic_restore")
decode_path = get_path("data/processed/2_produce_data/synthetic_decoded")

#%%

pd.read_csv(restore_path.joinpath("Synthetic_data_epsilon10000.csv"))



#%%
