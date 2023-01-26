
#%%
from pkgutil import get_data
import pandas as pd
import numpy as np
import pickle
import os, sys
from pathlib import Path
import argparse

PROJ_PATH = Path(__file__).parents[2]
sys.path.append(PROJ_PATH.joinpath('src').as_posix())

from MyModule.utils import *
config = load_config()

PROJ_PATH = Path(config['path_config']['project_path'])
INPUT_PATH = PROJ_PATH.joinpath('data/processed/2_restore/restore_to_s1')
OUTPUT_PATH = PROJ_PATH.joinpath('data/processed/2_restore/restore_to_db_form')

if not OUTPUT_PATH.exists() :
    OUTPUT_PATH.mkdir(parents=True)
    
#%%

def load_s1_pickle(epsilon):
    s1_path = INPUT_PATH.joinpath(f'S1_{epsilon}.pkl')
    return pd.read_pickle(s1_path).reset_index(drop=True)

all_required_keys = config['data_config']['required'].keys()
prefix_dictionary = config.get('data_config').get('prefix')

#%%
def get_data_by_table_name(s1, table_name):
    prefix = prefix_dictionary[table_name]
    data = s1.iloc[:, s1.columns.str.startswith(prefix)].copy()
    data.columns = data.columns.str.replace(prefix, '')
    id_and_time = s1[['PT_SBST_NO','TIME']]
    return pd.concat([id_and_time, data], axis=1)

def save_table(table, table_name, epsilon):
    
    table.to_csv(OUTPUT_PATH.joinpath(f'{table_name}_{epsilon}.csv'), index=False)
    
    pass

#%%
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epsilon','--e',help='epsilon values')
    args = parser.parse_args()
    
    all_tables = prefix_dictionary.keys()
    
    s1 = load_s1_pickle(args.epsilon)
    
    for table in all_tables:
        data = get_data_by_table_name(s1, table)
        save_table(data, table, args.epsilon)

if __name__=="__main__" : 
    main()