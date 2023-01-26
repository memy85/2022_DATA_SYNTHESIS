#%%
import pandas as pd
from pathlib import Path
import os, sys
from lifelines.utils import add_covariate_to_timeline

PROJECT_PATH = Path(__file__).parents[2].as_posix()
os.sys.path.append(PROJECT_PATH)
# %%
from src.MyModule.utils import *

config = load_config()
# %%
PROJECT_PATH = Path(config['path_config']['project_path'])
INPUT_PATH = PROJECT_PATH.joinpath('data/processed/0_preprocess')
OUTPUT_PATH = PROJECT_PATH.joinpath('data/processed/notebooks/')
# %%

d0 = pd.read_pickle(INPUT_PATH.joinpath('D0.pkl'))

idx = d0.PT_BSNF_BSPT_IDGN_AGE < 50
d1 = d0[idx].copy()
youngsters = d1.PT_SBST_NO.unique().tolist()
# %%
d1 = d0[d0.PT_SBST_NO.isin(youngsters)].copy()

# %% forward filling
patient_info = d1.PT_SBST_NO
d1 = d1.groupby(['PT_SBST_NO'], as_index=False).ffill()

#%%
d1['PT_SBST_NO'] = patient_info
# %%
d1
