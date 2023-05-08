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

def get_machine_learning_data(project_path, age, random_seed) :

    # load synthetic data
    synthetic_path = project_path.joinpath(f"data/processed/seed{random_seed}/2_produce_data/synthetic_decoded/")
    config = load_config()

    synthtic_data_dict = {}
    original_data_dict = {}

    for epsilon in config['epsilon'] : 
        df = pd.read_csv(synthetic_path.joinpath(f"Synthetic_data_epsilon{epsilon}_{age}.csv"))
        synthtic_data_dict[f'epsilon{epsilon}'] = df

    # load original with train and test
    original_path = project_path.joinpath(f"data/processed/seed{random_seed}/")

    ## train & test path
    train_path = original_path.joinpath(f"1_preprocess/train_ori_{age}.pkl")
    test_path = original_path.joinpath(f"1_preprocess/test_{age}.pkl")

    original_data_dict['train'] = pd.read_pickle(train_path)
    original_data_dict['test'] = pd.read_pickle(test_path)
    
    ## preprocess for machine learning
    preprocessedDict = preprocess_for_machine_learning(original = original_data_dict['train'],
                                    holdout = original_data_dict['test'],
                                    synthetic_dict = synthtic_data_dict)
    

    return preprocessedDict


def preprocess_for_machine_learning(*args, **kwargs):
    '''
    preprocess the machine learning datas
    '''
     
    for key in kwargs.keys():
        if key in ['original', 'holdout', 'synthetic_dict'] :
            if key == "original" :
                original = kwargs['original'] 
                original = original.rename(columns = {"RLPS DIFF":"RLPS_DIFF"})
                original = original.drop(columns = "PT_SBST_NO")
                original = original.fillna(999)
                continue 
                
            elif key == 'holdout' :                
                holdout = kwargs['holdout']
                holdout = holdout.rename(columns = {"RLPS DIFF":"RLPS_DIFF"})
                holdout = holdout.drop(columns = "PT_SBST_NO")
                holdout = holdout.fillna(999)
            
            elif key == 'synthetic_dict' :
                syntheticDict = kwargs['synthetic_dict']

                processed_SyntheticDict = {}

                for key, syntheticData in syntheticDict.items():
                    syntheticData = syntheticData.copy()
                    syntheticData = syntheticData.drop(columns = ["Unnamed: 0","PT_SBST_NO"])
                    syntheticData = syntheticData.fillna(999)
                    processed_SyntheticDict[key] = syntheticData


    
    return {"original" : original,
            "holdout": holdout,
            "synthetic_dict" : processed_SyntheticDict}
    

