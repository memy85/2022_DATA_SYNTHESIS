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
from src.MyModule.utils import *

print(f" this is project path : {project_path} ")

#%% settings

age = 50

#%% path settings

config = load_config()

input_path = get_path("data/processed/2_produce_data/synthetic_decoded")

synthetic_path = input_path.joinpath(f"Synthetic_data_epsilon10000_{age}.csv")

synthetic_data_path_list = [input_path.joinpath(
    f"Synthetic_data_epsilon{eps}_{age}.csv") for eps in config['epsilon']]

train_ori_path = get_path(f"data/processed/preprocess_1/train_ori_{age}.pkl")

testset_path = get_path(f"data/processed/preprocess_1/test_{age}.pkl")

output_path = get_path("data/processed/3_evaluate_data/")

if not output_path.exists():
    output_path.mkdir(parents=True)


#%% import models
# synthetic = pd.read_csv(synthetic_path)
synthetic_data_list = list(
    map(lambda x: pd.read_csv(x), synthetic_data_path_list))
train_ori = pd.read_pickle(train_ori_path)
test = pd.read_pickle(testset_path)

#%%

test.drop(["PT_SBST_NO"], axis=1, inplace=True)
test = test.rename(columns = {"RLPS DIFF":"RLPS_DIFF"})
train_ori.drop(["PT_SBST_NO"], axis=1, inplace=True)
train_ori = train_ori.rename(columns={"RLPS DIFF": "RLPS_DIFF"})
synthetic_data_list = list(map(lambda x: x.drop(
    ["PT_SBST_NO", "Unnamed: 0"], axis=1), synthetic_data_list))
#%%

synthetic_data_dict = {"eps_{}".format(config['epsilon'][idx]) : data for idx, data in enumerate(synthetic_data_list) }

#%%

train_ori.shape
test.shape
synthetic_data_list[0].shape

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
################################################################# Machine Learning
################################################################# Machine Learning
################################################################# Machine Learning
from sklearn.model_selection import train_test_split


outcome = "DEAD"
drop_columns = [outcome, "DEAD_DIFF"] 
columns = make_columns(train_ori, drop_columns)

ori_x = process_for_training(train_ori, outcome, columns)
ori_y = train_ori[outcome]

ori_train_x, ori_valid_x, ori_train_y, ori_valid_y = train_test_split(ori_x, ori_y, test_size=0.2, random_state=0, stratify=ori_y)

syn_x_dict = {k : process_for_training(df, outcome, columns) for k, df in synthetic_data_dict.items()}
syn_y_dict = {k : df[outcome] for k, df in synthetic_data_dict.items()}

test_x = process_for_training(test, outcome, columns)
test_y = test[outcome]


# make train test here!

#%%

def check_if_the_columns_are_same(data1, data2):
    set1 = set(data1.columns.tolist())
    set2 = set(data2.columns.tolist())
    output = set1 - set2

    if len(output) == 0:
        return True
    else :
        return False

assert check_if_the_columns_are_same(syn_x_dict["eps_10000"], test_x), "The columns are not the same"


#%% define models
# models = [DecisionTreeClassifier(),
#           KNeighborsClassifier(),
#           RandomForestClassifier(n_jobs=-1),
#           XGBClassifier()]

model_names = [
    "DecisionTree", "RandomForest", "XGBoost"
]
model_path = output_path.joinpath("best_models")


#%%
# tstr_scores = []
# real_scores = []

scores = []
for i, model in enumerate(model_names):
    print(f'processing ...' + str(model).split('c')[0])
    # result = output(real, syn_data, model, bar=True, target=outcome)
    accuracy, auc, f1 = train_and_test(model,
                            train_x = ori_train_x ,
                            train_y = ori_train_y ,
                            valid_x = ori_valid_x ,
                            valid_y = ori_valid_y ,
                            test_x = test_x,
                            test_y = test_y,
                            model_path = model_path)
    result = {"model": model, "type":"original", "f1_score": f1, "auroc": auc, "accuracy": accuracy}
    scores.append(result)

    for key, df in synthetic_data_dict.items():
        validation_set_counts = len(ori_valid_x)
        syn_x = syn_x_dict[key]
        syn_y = syn_y_dict[key]
        
        syn_train_x, syn_valid_x, syn_train_y, syn_valid_y = train_test_split(syn_x, syn_y, test_size=0.2, random_state=0, stratify=syn_y)

        accuracy, auc, f1 = train_and_test(model,
                                train_x = syn_train_x,
                                train_y = syn_train_y,
                                valid_x = syn_valid_x,
                                valid_y = syn_valid_y,
                                test_x = test_x,
                                test_y = test_y,
                                model_path = model_path)

        result = {"model": model_names[i], "type":key, "f1_score": f1, "auroc": auc, "accuracy": accuracy}
        scores.append(result)

    print("ended evaluating for {}...".format(model_names[i]))
print("finished calculating the outcomes...!")


#%%
scoreDF = pd.DataFrame(scores)
scoreDF


#%%
scoreDF.to_csv(output_path.joinpath("fig1score.csv"), index=False)


################################################################# Machine Learning
################################################################# Machine Learning
################################################################# Machine Learning

#%%

scoreDF = pd.read_csv(output_path.joinpath("fig1score.csv"))


#%%

scoreDF = scoreDF[["model","type","accuracy","auroc","f1_score"]]


#%% plot data
import scienceplots
import matplotlib as mpl
import matplotlib.pyplot as plt

plt.style.use(["science","ieee"])
mpl.rcParams.update({"font.size": 8})

#%%
fig, ax = plt.subplots(figsize=(8,6))

models = scoreDF["model"].unique().tolist()
ticklabels = ["Original", "$\epsilon$ \n 0", "$\epsilon$ \n 0.1", "$\epsilon$ \n 1", "$\epsilon$ \n 10", "$\epsilon$ \n 100", "$\epsilon$ \n 1000", "$\epsilon$ \n 10000"]

rows = ["Model", "Accuracy", "AUROC", "F1 Score"]

tempdf = scoreDF[scoreDF.model == "DecisionTree" ].copy()
tempdf.set_index("type").plot.bar(ax = ax, legend=False, xticks = [])
cell_text = [tempdf[cols].apply(lambda x : round(x,3)).tolist() for cols in ["accuracy", "auroc", "f1_score"]]

ax.set(xlabel  = [])
cell_text.insert(0, ticklabels)

ax.table(cellText = cell_text,
              rowLabels = rows, 
              loc = "bottom",
              )
table.auto_set_font_size(False)
table.auto_fontsize(8)

# ax.set_xticks(ticks = [i for i in range(0, len(ticklabels))], labels = ticklabels, rotation= 90)
# plt.tight_layout()
plt.show()

#%%

tempdf
