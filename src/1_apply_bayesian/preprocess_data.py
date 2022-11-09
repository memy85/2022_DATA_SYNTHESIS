#%%
import pandas as pd
import pickle
import argparse
from pathlib import Path
import os, sys
import multiprocessing
import yaml
from itertools import repeat

PROJECT_PATH = Path(__file__).parents[2]

#%%
with open(PROJECT_PATH.joinpath('config/config.yaml')) as f:
    config = yaml.load(f, yaml.SafeLoader)
os.sys.path.append(config['path_config']['project_path'])

from src.MyModule.utils import *

#%%
PROJ_PATH = Path(config['path_config']['project_path'])
INPUT_PATH = PROJ_PATH.joinpath('data/processed/0_preprocess')
OUTPUT_PATH = PROJ_PATH.joinpath('data/processed/1_apply_bayesian/preprocess_data')

if not OUTPUT_PATH.exists() :
    OUTPUT_PATH.mkdir(exist_ok=True, parents=True)

#%%
def load_input_data():
    path = INPUT_PATH.joinpath('D0.pkl')
    assert path.exists(), 'D0.pkl does not exist. Start from 0_preprocess'
    
    with open(path, 'rb') as f:
        return pickle.load(f)

def load_encodings():
    '''
    loads the encodings
    '''
    path = INPUT_PATH.joinpath('encoding.pkl')
    with open(path, 'rb') as f:
        return pickle.load(f)
    
def chunk_data(D0, patient_id):
    '''
    splits D0 by patient to patient and saves it into a data format
    input : D0 -> the original data
            patient_id -> patient id, PT_SBST_NO
    output : data for the corresponding patient_id
    '''
    return D0[D0.PT_SBST_NO == patient_id].copy()

def save_data(data_of_single_patient, patient_id):
    '''
    saves the input data
    '''
    path = OUTPUT_PATH.joinpath(f'pt_{patient_id}.csv')
    data_of_single_patient.to_csv(path, index=False)
    return

#%%
D0 = load_input_data()
categorical_columns = D0.select_dtypes('object').columns.tolist()

#%%
def preprocess_data(D0, patient_id):
    data = chunk_data(D0, patient_id)
    data = data.loc[:,~data.isna().all()].copy()
    
    data = data[data.columns[1:]]
    
    save_data(data, patient_id)
    return 

with open(OUTPUT_PATH.joinpath('categorical_columns.pkl'),'wb') as f:
    pickle.dump(categorical_columns, f)

#%%
patients = D0.PT_SBST_NO.unique().tolist()

#%% 
with multiprocessing.Pool(8) as p:
    p.starmap(preprocess_data, zip(repeat(D0), patients))
