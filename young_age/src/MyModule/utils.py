from pathlib import Path
import os, sys
import yaml

import numpy as np
import pickle
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler
import warnings


def load_config():
    utils_path = Path(__file__).absolute().parents[2]

    with open(utils_path.joinpath("config/config.yaml")) as f:
        config = yaml.load(f, yaml.SafeLoader)

    return config


def get_path(path):
    '''
    insert a path to the data
    '''
    config = load_config()

    project_path = Path(config["project_path"])
    return project_path.joinpath(path)

def read_csv(path, epsilon):
    '''
    path : path inside a project
    '''
    input_path = get_path(path)
    epsilon = str(epsilon)
    csv_name = 'S0_'+epsilon+'.csv'
    data = pd.read_csv(input_path.joinpath(csv_name))
    return data

def scale(data):
    scaler = MinMaxScaler()
    data = scaler.fit_transform(data)
    
    return data

def cross_entropy(pred_prob, y):
    delta = 1e-7
    return -np.sum((y*np.log(pred_prob+delta)))


def get_syn_data(sampling = False):
    syn_data = []
    epsilons = [0.1,1,10,100,1000,10000]
    for epsilon in epsilons:
        if sampling == False:
            temp_syn = (read_csv(epsilon))
            temp_syn = temp_syn.rename(columns={'DEAD_NFRM_DEAD':'DEAD'})
            temp_syn.drop(['Unnamed: 0'], axis=1, inplace = True)
            temp_syn.drop(['PT_SBST_NO'], axis=1, inplace = True)
            syn_data.append(temp_syn)
        # under sampling synthetic data for comparsion
        elif sampling == True:
            temp_syn = (read_csv(epsilon))
            
            dead_syn = temp_syn.loc[temp_syn.DEAD == 1]
            dead_sample = int(len(dead_syn)/5)

            dead_syn = dead_syn.sample(n=dead_sample,replace=True)

            survive_syn = temp_syn.loc[temp_syn.DEAD == 0].sample(n=int(dead_sample*11.613), replace = True)

            syn_data.append(pd.concat([dead_syn, survive_syn]))
            
    return syn_data