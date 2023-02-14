#%%


from pathlib import Path
import os, sys

# project_path = Path(__file__).absolute().parents[2]
project_path = Path().cwd()
os.sys.path.append(project_path.as_posix())
#%%
from src.MyModule.utils import *

config = load_config()
input_path = get_path("data/processed/2_produce_data/synthetic_decoded/")
figure_path = get_path("figures/")
ouput_path = get_path("data/processed/3_evaluate_data/")
#%%
import pandas as pd
import numpy as np

#%%
original_path = get_path('data/raw/D0_Handmade_ver1.1.xlsx')
synthetic_path = get_path('data/processed/2_produce_data/synthetic_restore/Synthetic_data_epsilon10000_50.csv')

original = pd.read_excel(original_path) 
synthetic = pd.read_csv(synthetic_path, encoding = 'cp949')

#%% define variable

columns1 = ["BSPT_SEX_CD", "BSPT_IDGN_AGE", "BSPT_FRST_DIAG_NM", "BSPT_STAG_CLSF_CD", "BSPT_STAG_VL",
           "RLPS","RLPS_DTRN_DCNT","OVRL_SRVL_DTRN_DCNT", "DEAD"]
columns2 = ["BSPT_SEX_CD", "BSPT_IDGN_AGE", "BSPT_FRST_DIAG_NM", "BSPT_STAG_CLSF_CD", "BSPT_STAG_VL",
           "RLPS","RLPS_DIFF","OVR_SURV", "DEAD"]

ori = original[columns1].copy()
syn = synthetic[columns2].copy()

syn = syn.rename(columns = {"RLPS_DIFF" : "RLPS_DTRN_DCNT", "OVR_SURV" : "OVRL_SRVL_DTRN_DCNT"})

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

processor = CorrelationChecker(ori, syn)        

#%%
diff = processor.calculate_correlation_diff()

#%%
diff

#%%
import matplotlib.pyplot as plt
import matplotlib as mpl
import scienceplots

cols = list(processor.data1columns)

fig, ax = plt.subplots(figsize = (12,12))

im = ax.imshow(diff, cmap='YlGn')
cbar = ax.figure.colorbar(im, ax = ax, cmap='YlGn')

ax.set_xticks(np.arange(diff.shape[1]), labels = cols, rotation=90)
ax.set_yticks(np.arange(diff.shape[1]), labels = cols)
ax.tick_params(axis = 'both', labelsize = 7)

plt.title("Correlation Difference")
plt.savefig(figure_path.joinpath("correlation.png"), dpi=300)
plt.show()

