#%%
import pandas as pd
from pathlib import Path

project_path = Path().cwd()
trtm_path = project_path.joinpath("../data/raw/CLRC_TRTM_CASB.xlsx")

trtm = pd.read_excel(trtm_path)

#%%
trtm.columns 

#%%
trtm.CSTR_CYCL_VL.unique()


