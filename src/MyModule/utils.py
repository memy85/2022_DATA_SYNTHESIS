
from fileinput import filename
import pandas as pd
from pandas import DataFrame
import numpy as np
from pathlib import Path
import yaml
import pickle

def convert_dates(data, config, table_name):
    '''
    convert dates that are in object format to pandas dates
    '''
    data_new = data.copy()
    def give_ymd_data(required_columns:dict):
        '''
        filter ymd data
        '''
        return [k for k, v in required_columns.items() if v == 'datetime64[ns]']
    
    date_cols = give_ymd_data(config['data_config']['required'][table_name])
    for col in date_cols:
        data_new[col] = pd.to_datetime(data_new[col], format='%Y%m%d')
    return data_new

def remove_invalid_values(data : DataFrame):
    data = data.replace('x',np.nan)
    data = data.replace('nan',np.nan)
    return data

def read_file(path : Path, file_name):
    extension = file_name.split('.')[1]
    if extension == "xlsx":
        return pd.read_excel(path.joinpath(file_name))
    elif extension == 'csv' :
        return pd.read_csv(path.joinpath(file_name))
    else :
        return pd.read_pickle(path.joinpath(file_name))

def load_config():
    project_dir = Path(__file__).parents[2]
    conf_dir = project_dir.joinpath('config')
    with open(conf_dir.joinpath('config.yaml')) as f:
        return yaml.load(f, yaml.SafeLoader)
    

