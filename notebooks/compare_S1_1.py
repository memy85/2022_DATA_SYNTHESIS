
import pandas as pd
from pathlib import Path
import pickle

PROJECT_PATH = Path().cwd().parents[0]
d0_path = PROJECT_PATH.joinpath('data/processed/0_preprocess/D0.pkl')
s1_path = PROJECT_PATH.joinpath('data/processed/2_restore/restore_to_s1/S1_1.pkl')

# |%%--%%| <xMiDjZWs13|5IGWIJLqmK>

if not PROJECT_PATH.joinpath('data/processed/notebooks').exists():
    PROJECT_PATH.joinpath('data/processed/notebooks').mkdir(parents=True)
    
OUTPUT_PATH = PROJECT_PATH.joinpath('data/processed/notebooks')

# |%%--%%| <5IGWIJLqmK|IE1K95B0XU>

D0 = pd.read_pickle(d0_path)
S1 = pd.read_pickle(s1_path)

# |%%--%%| <IE1K95B0XU|ueeGPIzqsI>

D0.to_csv('D0.csv', index=False)

# |%%--%%| <ueeGPIzqsI|ZkPyO0hadI>

S1 = S1.head(10).copy()

# |%%--%%| <ZkPyO0hadI|wupwHbJMmo>

S1 = S1.drop(columns = "Unnamed: 0")

S1.to_csv('s1.csv', index=False)

# |%%--%%| <wupwHbJMmo|ipQxuMzn2C>

S1.drop(columns="Unnamed: 0")

# |%%--%%| <ipQxuMzn2C|kPeLovwrjR>

D0.groupby('PT_SBST_NO')['TIME'].max().mean()
D0.groupby('PT_SBST_NO')['TIME'].max().std()

# |%%--%%| <kPeLovwrjR|GgKLLkF4hM>

D0.query('DEAD_NFRM_DEAD == 1')

# |%%--%%| <GgKLLkF4hM|1RzPbOkJrk>

import os, sys
from pathlib import Path

files = os.listdir('/home/wonseok/2022_DATA_SYNTHESIS/data/processed/1_apply_bayesian/apply_bayesian/out/epsilon0.1/')
def filter_out_descriptions(file_name):
    return 'description' in file_name

files = sorted(list(filter(filter_out_descriptions, files)))



# |%%--%%| <1RzPbOkJrk|FIyJr4GZuM>

with open(PROJECT_PATH.joinpath('data/processed/0_preprocess/encoding.pkl'), 'rb') as f:
    encoding = pickle.load(f)
    
def reverse_encoding(book):
    return {v:k for k, v in book.items()}

book = {col : reverse_encoding(encoding[col]) for col in encoding.keys() if col != "PT_SBST_NO"}

# |%%--%%| <FIyJr4GZuM|fefgUVPKOO>

book

# |%%--%%| <fefgUVPKOO|e86xFpAuhi>

D0.PT_BSNF_BSPT_SEX_CD

# |%%--%%| <e86xFpAuhi|JEX4mhX1GW>

D0 = D0.replace(book)
S1 = S1.replace(book)

# |%%--%%| <JEX4mhX1GW|y38RoiurBF>

D0.to_csv(OUTPUT_PATH.joinpath('D0.csv'), index=False)
S1.to_csv(OUTPUT_PATH.joinpath('S1.csv'), index=False)

# |%%--%%| <y38RoiurBF|vGyV2vAIaA>

import pandas as pd

df = pd.read_pickle('/home/wonseok/2022_DATA_SYNTHESIS/data/processed/2_restore/restore_to_s1/S1_0.1.pkl')

# |%%--%%| <vGyV2vAIaA|QEQ0p65ueJ>

df.to_csv('synthetic_data_epsilon_0.1.csv', index=False)
