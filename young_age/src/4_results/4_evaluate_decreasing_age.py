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

# project_path = Path(__file__).absolute().parents[2]
project_path = Path().cwd()

os.sys.path.append(project_path.as_posix())

from src.MyModule.distribution_comparison import *
from src.MyModule.ml_function import *
# from src.MyModule.age_functions import *
from src.MyModule.utils import *

print(f" this is project path : {project_path} ")

#%%

config = load_config()
age_cut = config['age_cut']

input_path = get_path("data/processed/2_produce_data/synthetic_decoded")
preprocess_1_path = get_path('data/processed/preprocess_1/')


synthetic_data_path_list = [input_path.joinpath(
    f"Synthetic_data_epsilon10000_{age}.csv") for age in age_cut] 

train_ori_data_path_list = [preprocess_1_path.joinpath(f'train_ori_{age}.pkl') for age in age_cut]

test_ori_data_path_list = [preprocess_1_path.joinpath(f'test_{age}.pkl') for age in age_cut]

output_path = get_path("data/processed/4_results/")

if not output_path.exists():
    output_path.mkdir(parents=True)


#%% import models
# synthetic = pd.read_csv(synthetic_path)
synthetic_data_list = list(
    map(lambda x: pd.read_csv(x), synthetic_data_path_list))

train_ori_list = list(
    map(lambda x: pd.read_pickle(x), train_ori_data_path_list))

test_ori_list = list(
    map(lambda x: pd.read_pickle(x), test_ori_data_path_list))


#%% preprocess data

def preprocess_original(data) :
    data.drop(['PT_SBST_NO'], axis=1, inplace=True)
    data = data.rename(columns = {"RLPS DIFF" : "RLPS_DIFF"})
    return data

def preprocess_synthetic(data) :
    data= data.drop(columns = ['PT_SBST_NO', "Unnamed: 0"])
    return data

#%%

train_data_dict = {age_cut[idx] : preprocess_original(data) for idx, data in enumerate(train_ori_list) }
test_data_dict = {age_cut[idx] : preprocess_original(data) for idx, data in enumerate(test_ori_list) }
synthetic_data_dict = {age_cut[idx] : preprocess_synthetic(data) for idx, data in enumerate(synthetic_data_list) }

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
from sklearn.model_selection import train_test_split

outcome = "DEAD"
drop_columns = [outcome, "DEAD_DIFF"] 
columns = make_columns(train_data_dict[50], drop_columns)

def make_train_valid_dataset(data, test=False) :

    # split x and y
    x = process_for_training(data, outcome, columns)
    y = data[outcome]
    if test :
        return x, y

    train_x, valid_x, train_y, valid_y = train_test_split(x, y, test_size=0.2, random_state=0, stratify=y)

    return (train_x, train_y) , (valid_x, valid_y)

ori_train_dict = {age : make_train_valid_dataset(data, test=False) for age, data in train_data_dict.items()}
syn_train_dict = {age : make_train_valid_dataset(data, test=False) for age, data in synthetic_data_dict.items()}
test_dict = {age : make_train_valid_dataset(data, test=True) for age, data in test_data_dict.items()}

#%%

# make train test here!

#%%
model_names = [
    "DecisionTree", "RandomForest", "XGBoost"
]
model_path = output_path.joinpath("best_models_age")

if not model_path.exist() :
    model_path.mkdir(parents = True)


scores = []
for i, model in enumerate(model_names):

    for age in config['age_cut'] :
        print(f'processing ...' + str(model))

        test_x, test_y = test_dict[age]
        (ori_train_x, ori_train_y), (ori_valid_x, ori_valid_y) = ori_train_dict[age]

        # original 
        accuracy, auc, f1 = train_and_test(model,
                                train_x = ori_train_x ,
                                train_y = ori_train_y ,
                                valid_x = ori_valid_x ,
                                valid_y = ori_valid_y ,
                                test_x = test_x,
                                test_y = test_y,
                                model_path = model_path,
                                scaler = None)
        result = {"model": model, "type":"original", "age":age, "f1_score": f1, "auroc": auc, "accuracy": accuracy}
        scores.append(result)

        (syn_train_x, syn_train_y), (syn_valid_x, syn_valid_y) = syn_train_dict[age]

        # original 
        accuracy, auc, f1 = train_and_test(model,
                                train_x = syn_train_x ,
                                train_y = syn_train_y ,
                                valid_x = syn_valid_x ,
                                valid_y = syn_valid_y ,
                                test_x = test_x,
                                test_y = test_y,
                                model_path = model_path,
                                scaler = None)

        result = {"model": model, "type":"synthetic", "age":age, "f1_score": f1, "auroc": auc, "accuracy": accuracy}
        scores.append(result)

    print("ended evaluating for {}...".format(age))
print("finished calculating the outcomes...!")

#%%
scoreDF = pd.DataFrame(scores)

#%%

scoreDF.to_csv(output_path.joinpath('scoredf.csv'),index=False)

#%% load scoredf

scoreDF = pd.read_csv(output_path.joinpath('scoredf.csv'))

#%%
import matplotlib.pyplot as plt
import seaborn as sns

model_names = [
    "DecisionTree", "RandomForest", "XGBoost"
]

#%%

fig, axs = plt.subplots(figsize = (10,5))
colors = ['blue','cyan']

tempdf = scoreDF[scoreDF.model == model_names[0]].copy()
tempdf = tempdf[['type','age','f1_score']]

# sns.barplot(x = 'age', data = tempdf, hue='type', y = 'f1_score')
tempdf.set_index('age').plot.bar(ax = axs, legend=False, color = colors)
axs.invert_xaxis()
axs.set_title(model_names[0])
plt.show()


#%%

fig, axs = plt.subplots(figsize = (10,5))
colors = ['blue','cyan']

tempdf = scoreDF[scoreDF.model == model_names[1]].copy()
tempdf = tempdf[['type','age','auroc','f1_score']]

sns.barplot(x = 'age', data = tempdf, hue='type', y = 'f1_score')
# tempdf.set_index('age').plot.bar(ax = axs, legend=False, color = colors)
axs.invert_xaxis()
axs.set_title(model_names[1])
plt.show()

#%%
fig, axs = plt.subplots(figsize = (10,5))
colors = ['blue','cyan']

tempdf = scoreDF[scoreDF.model == model_names[2]].copy()
tempdf = tempdf[['type','age','auroc','f1_score']]

sns.barplot(x = 'age', data = tempdf, hue='type', y = 'f1_score')
axs.invert_xaxis()
axs.set_title(model_names[2])
plt.show()



#%%
tempdf
