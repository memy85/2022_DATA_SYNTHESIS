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
output_path = get_path("data/processed/3_evaluate_data/")
figure_path = get_path("figures/")

original_path = get_path("data/raw/original.xlsx")

#%%

original = pd.read_excel(original_path)
#%%

original.head()

#%%

original.shape

#%%

