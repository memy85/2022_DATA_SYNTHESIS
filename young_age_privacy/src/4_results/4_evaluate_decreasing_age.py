#%%
import scienceplots

from xgboost import XGBClassifier
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

import shap

# project_path = Path(__file__).absolute().parents[2]
project_path = Path().cwd()

os.sys.path.append(project_path.as_posix())

from src.MyModule.distribution_comparison import *
from src.MyModule.ml_function import *
from src.MyModule.utils import *

print(f" this is project path : {project_path} ")

#%%

random.seed(0)
config = load_config()
age_cut = config['age_cut']

input_path = get_path("data/processed/2_produce_data/synthetic_decoded")
preprocess_1_path = get_path('data/processed/preprocess_1/')


synthetic_data_path_list10000 = [input_path.joinpath(
    f"Synthetic_data_epsilon10000_{age}.csv") for age in age_cut] 

synthetic_data_path_list1000 = [input_path.joinpath(
    f"Synthetic_data_epsilon1000_{age}.csv") for age in age_cut] 

synthetic_data_path_list100 = [input_path.joinpath(
    f"Synthetic_data_epsilon100_{age}.csv") for age in age_cut] 

synthetic_data_path_list10 = [input_path.joinpath(
    f"Synthetic_data_epsilon10_{age}.csv") for age in age_cut] 

synthetic_data_path_list1 = [input_path.joinpath(
    f"Synthetic_data_epsilon1_{age}.csv") for age in age_cut] 

synthetic_data_path_list01 = [input_path.joinpath(
    f"Synthetic_data_epsilon0.1_{age}.csv") for age in age_cut] 

synthetic_data_path_list0 = [input_path.joinpath(
    f"Synthetic_data_epsilon0_{age}.csv") for age in age_cut] 

#%%

train_ori_data_path_list = [preprocess_1_path.joinpath(f'train_ori_{age}.pkl') for age in age_cut]

test_ori_data_path_list = [preprocess_1_path.joinpath(f'test_{age}.pkl') for age in age_cut]

output_path = get_path("data/processed/4_results/")

if not output_path.exists():
    output_path.mkdir(parents=True)


#%% import models
# synthetic = pd.read_csv(synthetic_path)
synthetic_data_list10000 = list(
    map(lambda x: pd.read_csv(x), synthetic_data_path_list10000))

synthetic_data_list1000 = list(
    map(lambda x: pd.read_csv(x), synthetic_data_path_list1000))

synthetic_data_list100 = list(
    map(lambda x: pd.read_csv(x), synthetic_data_path_list100))

synthetic_data_list10 = list(
    map(lambda x: pd.read_csv(x), synthetic_data_path_list10))

synthetic_data_list1 = list(
    map(lambda x: pd.read_csv(x), synthetic_data_path_list1))

synthetic_data_list01 = list(
    map(lambda x: pd.read_csv(x), synthetic_data_path_list01))

synthetic_data_list0 = list(
    map(lambda x: pd.read_csv(x), synthetic_data_path_list0))

synthetic_data_list = [
        synthetic_data_list0,
        synthetic_data_list01,
        synthetic_data_list1,
        synthetic_data_list10,
        synthetic_data_list100,
        synthetic_data_list1000,
        synthetic_data_list10000
                       ]

train_ori_list = list(
    map(lambda x: pd.read_pickle(x), train_ori_data_path_list))

test_ori_list = list(
    map(lambda x: pd.read_pickle(x), test_ori_data_path_list))


#%% preprocess data

def preprocess_original(data) :
    data = data.copy()
    data.drop(['PT_SBST_NO'], axis=1, inplace=True)
    data = data.rename(columns = {"RLPS_DIFF" : "RLPS DIFF"})
    return data

def preprocess_synthetic(data) :
    data= data.drop(columns = ['PT_SBST_NO', "Unnamed: 0"])
    data = data.rename(columns = {"RLPS_DIFF" : "RLPS DIFF"})
    return data

#%%

train_data_dict = {age_cut[idx] : preprocess_original(data) for idx, data in
                   enumerate(train_ori_list) }

test_data_dict = {age_cut[idx] : preprocess_original(data) for idx, data in
                  enumerate(test_ori_list) }

# synthetic_data_dict = {age_cut[idx] : preprocess_synthetic(data) for idx, data in
#                        enumerate(synthetic_data_list) }

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

outcome = "DEAD"
drop_columns = [outcome, "DEAD_DIFF", "OVR_SURV"] 
columns = make_columns(train_data_dict[50], drop_columns)

#%%

def make_train_valid_dataset(data, synthetic=True, test=False) :
    data = data.copy()

    x = process_for_training(data, outcome, columns)
    y = data[outcome]

    if synthetic : 
        rus = RandomUnderSampler(sampling_strategy = 0.5, random_state = 0)
        x, y = rus.fit_resample(x, y)

    # split x and y
    if test :
        return x, y

    train_x, valid_x, train_y, valid_y = train_test_split(x, y, test_size=0.2,
                                                          random_state=0,
                                                          stratify=y)

    return (train_x, train_y) , (valid_x, valid_y)

#%%

ori_train_dict = {age : make_train_valid_dataset(data, False, test=False) for age, data
                 in train_data_dict.items()}

test_dict = {age : make_train_valid_dataset(data, False, test=True) for age, data
             in test_data_dict.items()}

syn_train_dict = {
        epsilon : {
            age : make_train_valid_dataset(data, synthetic=True, test=False) for age, data
            in synthetic_data_dict[epsilon].items()
            } for epsilon in config['epsilon'] }
#%%
def save_shap_values(shap_value, path) :
    with open(path, 'wb') as f :
        pickle.dump(shap_value,f)


#%%

# make train test here!

#%%

def main() :

    model_names = [
        "DecisionTree", "RandomForest", "XGBoost"
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

            # original 
            accuracy, auc, f1, auprc, shap_values = train_and_test(model,
                                    train_x = ori_train_x ,
                                    train_y = ori_train_y ,
                                    valid_x = ori_valid_x ,
                                    valid_y = ori_valid_y ,
                                    test_x = test_x,
                                    test_y = test_y,
                                    model_path = age_model_path,
                                    scaler = None)

            result = {"model": model, "type":"original", "age":age, 
                      "f1_score": f1, "auroc": auc, "accuracy": accuracy, 
                      "auprc" : auprc}

            path = shap_path.joinpath(f"{model}_{age}_original.pkl")
            save_shap_values(shap_values, path)

            scores.append(result)

            for eps in config['epsilon'] :

                syn_book = syn_train_dict[eps]

                (syn_train_x, syn_train_y), (syn_valid_x, syn_valid_y) = syn_book[age]

                # original 
                accuracy, auc, f1, auprc, shap_values = train_and_test(model,
                                        train_x = syn_train_x ,
                                        train_y = syn_train_y ,
                                        valid_x = ori_valid_x ,
                                        valid_y = ori_valid_y ,
                                        test_x = test_x,
                                        test_y = test_y,
                                        model_path = age_model_path,
                                        scaler = None)

                result = {"model": model, "type":f"epsilon_{eps}", "age":age,
                          "f1_score": f1, "auroc": auc, "accuracy": accuracy,
                          "auprc" : auprc} 

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

