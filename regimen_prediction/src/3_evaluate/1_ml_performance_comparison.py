#%%
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
import pandas as pd
import numpy as np
import os
import sys 
from pathlib import Path
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import train_test_split
import random
import argparse

import shap

project_path = Path(__file__).absolute().parents[2]
# project_path = Path().cwd()

os.sys.path.append(project_path.as_posix())

from src.MyModule.distribution_comparison import *
from src.MyModule.ml_function import *
from src.MyModule.utils import *

print(f" this is project path : {project_path} ")

#%% set seed and arguments

config = load_config()
age_cut = config['age_cut']

def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--random_seed", default = 0, type = int)
    args = parser.parse_args()
    return args

args = arguments()
age = 50

input_path = get_path(f"data/processed/seed{args.random_seed}/2_produce_data/synthetic_decoded")
preprocess_1_path = get_path(f'data/processed/seed{args.random_seed}/1_preprocess/')


synthetic_data_path_list100 = [input_path.joinpath(
    f"Synthetic_data_epsilon100_{age}.csv")] 

synthetic_data_path_list1000 = [input_path.joinpath(
    f"Synthetic_data_epsilon1000_{age}.csv")] 

synthetic_data_path_list10000 = [input_path.joinpath(
    f"Synthetic_data_epsilon10000_{age}.csv")] 

#%%

train_ori_data_path_list = [preprocess_1_path.joinpath(f'train_ori_{age}.pkl')]

test_ori_data_path_list = [preprocess_1_path.joinpath(f'test_{age}.pkl')]

output_path = get_path(f"data/processed/seed{args.random_seed}/4_results/")

if not output_path.exists():
    output_path.mkdir(parents=True)


#%% import models

synthetic_data_list100 = list(
    map(lambda x: pd.read_csv(x), synthetic_data_path_list100))

synthetic_data_list1000 = list(
    map(lambda x: pd.read_csv(x), synthetic_data_path_list1000))

synthetic_data_list10000 = list(
    map(lambda x: pd.read_csv(x), synthetic_data_path_list10000))


synthetic_data_list = [synthetic_data_list100, synthetic_data_list1000, synthetic_data_list10000]

train_ori_list = list(
    map(lambda x: pd.read_pickle(x), train_ori_data_path_list))

test_ori_list = list(
    map(lambda x: pd.read_pickle(x), test_ori_data_path_list))


#%% preprocess data

def preprocess_original(data) :
    data = data.copy()
    data.drop(['PT_SBST_NO',  'REGN_CSTR_PRPS_NT_1', 'REGN_CASB_CSTR_PRPS_NM_1', 
               'REGN_TIME_DIFF_1', 'REGN_START_DIFF_1', 'REGN_CSTR_CYCL_VL_END_1'], axis=1, inplace=True)
    data = data.rename(columns = {"RLPS_DIFF" : "RLPS DIFF"})
    return data

def preprocess_synthetic(data) :
    data = data.copy()
    data.drop(['PT_SBST_NO',  'REGN_CSTR_PRPS_NT_1', 'REGN_CASB_CSTR_PRPS_NM_1', 
               'REGN_TIME_DIFF_1', 'REGN_START_DIFF_1',  'REGN_CSTR_CYCL_VL_END_1'], axis=1, inplace=True)
    data = data.rename(columns = {"RLPS_DIFF" : "RLPS DIFF"})
    return data

#%%

train_data_dict = { age_cut[idx] : preprocess_original(data) for idx, data in
                   enumerate(train_ori_list) }

test_data_dict = { age_cut[idx] : preprocess_original(data) for idx, data in
                  enumerate(test_ori_list) }

synthetic_data_dict = {

        epsilon : {
            age_cut[j] : preprocess_synthetic(data) for j, data in 
            enumerate(synthetic_data_list[i])
            } for i, epsilon in enumerate(config['epsilon'])

        }

#%%

def process_for_training(data, outcome, columns):
    new_data = data.drop(outcome, axis=1).copy()
    new_data = new_data[columns]
    new_data = new_data.fillna(999)
    return new_data

def make_columns(data, drop_columns: list):
    col_list = data.columns.tolist()
    
    for cols in drop_columns:
        col_list.remove(cols) 

    return col_list

#%% set all the outcome and needless variables
################################################################# machine learning
################################################################# machine learning
################################################################# machine learning

outcome = "REGN_CSTR_REGN_NM_1"
drop_columns = [outcome] 
columns = make_columns(train_data_dict[50], drop_columns)

#%%
def transform_label(y) :
    num = np.unique(y, axis=0).shape[0]
    label = np.eye(num)[y]
    return label

def make_train_valid_dataset(data, synthetic=True, test=False) :
    data = data.copy()

    x = process_for_training(data, outcome, columns)
    y = data[outcome]

    # split x and y
    if test :
        return x, y

    train_x, valid_x, train_y, valid_y = train_test_split(x, y, test_size=0.2,
                                                          random_state=0,
                                                          stratify=y)
    return (train_x, train_y), (valid_x, valid_y)

def save_shap_values(shap_value, path) :

    with open(path, 'wb') as f :
        pickle.dump(shap_value, f)

#%% prepare dataset

ori_train_dict = {age : make_train_valid_dataset(data, False, test=False) for age, data
                 in train_data_dict.items()}

test_dict = {age : make_train_valid_dataset(data, False, test=True) for age, data
             in test_data_dict.items()}

syn_train_dict = {
        epsilon : {
            age : make_train_valid_dataset(data, synthetic=True, test=False) for age, data
            in synthetic_data_dict[epsilon].items()
            } for epsilon in config['epsilon'] 
        }

#%%

#### make train test here!

#%%

def main() :

    model_names = [
        "DecisionTree", 
        "RandomForest", 
        "XGBoost"
    ]

    model_path = output_path.joinpath("best_models_age")
    shap_path = output_path.joinpath("shap_values")

    if not model_path.exists() :
        model_path.mkdir(parents = True)

    if not shap_path.exists() :
        shap_path.mkdir(parents = True)

    scores = []
    for i, model in enumerate(model_names):

        for age in config['age_cut'] :

            age_model_path = model_path.joinpath(f'age{age}')

            print(f'processing ...' + str(model))

            test_x, test_y = test_dict[age]
            (ori_train_x, ori_train_y), (ori_valid_x, ori_valid_y) = ori_train_dict[age]

            with open(output_path.joinpath("testX.pkl"), 'wb') as f:
                pickle.dump(test_x, f)

            with open(output_path.joinpath("testY.pkl"), 'wb') as f:
                pickle.dump(test_y, f)

            # original 
            accuracy, auc, f1, shap_values = train_and_test(model,
                                    train_x = ori_train_x ,
                                    train_y = ori_train_y ,
                                    valid_x = ori_valid_x ,
                                    valid_y = ori_valid_y ,
                                    test_x = test_x,
                                    test_y = test_y,
                                    model_path = age_model_path,
                                    scaler = None)

            result = {"model": model, "type":"original", "age":age, 
                      "f1_score": f1, "auroc": auc, "accuracy": accuracy}

            path = shap_path.joinpath(f"{model}_{age}_original.pkl")
            save_shap_values(shap_values, path)

            scores.append(result)

            for eps in config['epsilon'] :

                syn_book = syn_train_dict[eps]
                age_model_path = model_path.joinpath(f'age{age}_eps{eps}')

                (syn_train_x, syn_train_y), (syn_valid_x, syn_valid_y) = syn_book[age]

                # original 
                accuracy, auc, f1, shap_values = train_and_test(model,
                                        train_x = syn_train_x ,
                                        train_y = syn_train_y ,
                                        valid_x = ori_valid_x ,
                                        valid_y = ori_valid_y ,
                                        test_x = test_x,
                                        test_y = test_y,
                                        model_path = age_model_path,
                                        scaler = None)

                result = {"model": model, "type":f"epsilon_{eps}", "age":age,
                          "f1_score": f1, "auroc": auc, "accuracy": accuracy}
                          

                path = shap_path.joinpath(f"{model}_{age}_{eps}.pkl")
                save_shap_values(shap_values, path)

                scores.append(result)

            print("ended evaluating for {}...".format(age))
        print("finished calculating the outcomes...!")

    scoreDF = pd.DataFrame(scores)
    scoreDF.to_pickle(output_path.joinpath('extreme_case.pkl'))
    print(f"saved file to {output_path.as_posix()}")
    return scoreDF

#%%


if __name__ == "__main__" :
    main()

