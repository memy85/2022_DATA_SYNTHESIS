#%%
import scienceplots

from xgboost import XGBClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from imblearn.under_sampling import RandomUnderSampler
import pandas as pd
import numpy as np
import os
import sys
from pathlib import Path
import argparse
from sklearn.model_selection import train_test_split

# project_path = Path(__file__).absolute().parents[2]
project_path = Path().cwd()

os.sys.path.append(project_path.as_posix())

from src.MyModule.distribution_comparison import *
from src.MyModule.ml_function import *
from src.MyModule.utils import *

print(f" this is project path : {project_path} ")

#%% path settings

config = load_config()

def argument_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--age", default = 50, type = int )
    args = parser.parse_args()
    return args

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


def main() :
    args = argument_parse()
    age = args.age

    input_path = get_path("data/processed/2_produce_data/synthetic_decoded")

    synthetic_path = input_path.joinpath(f"Synthetic_data_epsilon10000_{age}.csv")

    synthetic_data_path_list = [input_path.joinpath(
        f"Synthetic_data_epsilon{eps}_{age}.csv") for eps in config['epsilon']]

    train_ori_path = get_path(f"data/processed/preprocess_1/train_ori_{age}.pkl")

    testset_path = get_path(f"data/processed/preprocess_1/test_{age}.pkl")

    output_path = get_path("data/processed/4_results/")

    figure_path = get_path("figures")

    if not output_path.exists():
        output_path.mkdir(parents=True)


#%% import models
# synthetic = pd.read_csv(synthetic_path)
    synthetic_data_list = list(
        map(lambda x: pd.read_csv(x), synthetic_data_path_list))
    train_ori = pd.read_pickle(train_ori_path)
    test = pd.read_pickle(testset_path)

    test.drop(["PT_SBST_NO"], axis=1, inplace=True)
    test = test.rename(columns = {"RLPS DIFF":"RLPS_DIFF"})
    train_ori.drop(["PT_SBST_NO"], axis=1, inplace=True)
    train_ori = train_ori.rename(columns={"RLPS DIFF": "RLPS_DIFF"})
    synthetic_data_list = list(map(lambda x: x.drop(
        ["PT_SBST_NO", "Unnamed: 0"], axis=1), synthetic_data_list))
    
    synthetic_data_dict = {"eps_{}".format(config['epsilon'][idx]) : data for idx, data in enumerate(synthetic_data_list) }

    # train_ori.shape
    # test.shape
    # synthetic_data_list[0].shape


#%% set all the outcome and needless variables
################################################################# machine learning
################################################################# machine learning
################################################################# machine learning


    outcome = "DEAD"
    drop_columns = [outcome, "DEAD_DIFF", "OVR_SURV"] 
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

    assert check_if_the_columns_are_same(syn_x_dict["eps_10000"], test_x), "the columns are not the same"
    
    original = (ori_train_x, ori_train_y, ori_valid_x, ori_valid_y)
    synthetic = (syn_x_dict, syn_y_dict)
    test = (test_x, test_y)

    get_results(original, synthetic, test, synthetic_data_dict,  output_path)

#%% define models

def get_results(original, synthetic, test, synthetic_dict, output_path) :
    (ori_train_x, ori_train_y, ori_valid_x, ori_valid_y) = original
    (syn_x_dict, syn_y_dict) = synthetic
    synthetic_data_dict = synthetic_dict
    test_x, test_y = test


    model_names = [
        "DecisionTree", "RandomForest", "XGBoost",
    ]
    model_path = output_path.joinpath("best_models")
    #%%

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
                                model_path = model_path,
                                scaler = None)
        result = {"model": model, "type":"original", "f1_score": f1, "auroc": auc, "accuracy": accuracy}
        scores.append(result)

        for key, df in synthetic_data_dict.items():
            validation_set_counts = len(ori_valid_x)
            syn_x = syn_x_dict[key]
            syn_y = syn_y_dict[key]

            rus = RandomUnderSampler(sampling_strategy = 0.5, random_state = 0)
            syn_x, syn_y = rus.fit_resample(syn_x, syn_y)

            
            syn_train_x, syn_valid_x, syn_train_y, syn_valid_y = train_test_split(syn_x, syn_y, test_size=0.2, random_state=0, stratify=syn_y)

            accuracy, auc, f1 = train_and_test(model,
                                    train_x = syn_train_x,
                                    train_y = syn_train_y,
                                    valid_x = syn_valid_x,
                                    valid_y = syn_valid_y,
                                    test_x = test_x,
                                    test_y = test_y,
                                    model_path = model_path,
                                    scaler =None)

            result = {"model": model_names[i], "type":key, "f1_score": f1, "auroc": auc, "accuracy": accuracy}
            scores.append(result)

        print("ended evaluating for {}...".format(model_names[i]))
    print("finished calculating the outcomes...!")

    scoreDF = pd.DataFrame(scores)

    scoreDF.to_csv(output_path.joinpath("fig1score_tstr.csv"), index=False)

    print("finished saving the output!")

if __name__ == "__main__" :
    main()



#%%

# output_path = get_path("data/processed/4_results/")
# scoreDF = pd.read_csv(output_path.joinpath("fig1score.csv"))

#scoreDF = scoreDF[scoreDF.type != 'eps_0'][['model', 'type', 'auroc', 'f1_score']]
##%%
#scoreDF

##%% plot data
#import scienceplots
#import matplotlib as mpl
#import matplotlib.pyplot as plt

#plt.style.use(["science","ieee"])
#mpl.rcParams.update({"font.size": 7})

#colors = ['#3182bd', "#9ecae1"]
#lnewdth = 0.7
#title_fontsize = 10.0

#fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize = (4,6), sharex=True)

#ticklabels = ["Original", "$\epsilon$ = 0.1", "$\epsilon$ = 1", "$\epsilon$ = 10", "$\epsilon$ = 100", "$\epsilon$ = 1000", "$\epsilon$ = \n 10000"]

## decision tree
#tempdf = scoreDF[scoreDF.model == "DecisionTree"].copy()
#ori_auc = tempdf[tempdf.type == 'original']['auroc'].item()
#ori_f1 = tempdf[tempdf.type == 'original']['f1_score'].item()

#tempdf.set_index("type").plot.bar(ax = ax1, legend=False, color =colors)
#ax1.axhline(y = ori_auc, xmin = 0.04, linewidth = lnewdth, color = 'gray',linestyle = '--')
#ax1.axhline(y = ori_f1, xmin = 0.08, linewidth = lnewdth, color = 'gray',linestyle = '--')
#ax1.set_xticklabels([], rotation = 90)
#ax1.set_title('Decision Tree', fontsize= title_fontsize)

## RandomForest
#tempdf = scoreDF[scoreDF.model == "RandomForest"].copy()
#ori_auc = tempdf[tempdf.type == 'original']['auroc'].item()
#ori_f1 = tempdf[tempdf.type == 'original']['f1_score'].item()

#tempdf.set_index("type").plot.bar(ax = ax2, legend = False, color = colors)
#ax2.set_xticklabels([], rotation = 90)
#ax2.axhline(y = ori_auc, xmin = 0.04, linewidth = lnewdth, color = 'gray',linestyle = '--')
#ax2.axhline(y = ori_f1, xmin = 0.08, linewidth = lnewdth, color = 'gray',linestyle = '--')
#ax2.set_title('RandomRorest', fontsize = title_fontsize)
#ax2.legend(handles = [], labels = ['AUROC', "F1 Score"], labelcolor = colors, loc = 'center right',
#           bbox_to_anchor = (1.1, 0.3, 0.2, 0.2))

## XGboost
#tempdf = scoreDF[scoreDF.model == "XGBoost"].copy()
#ori_auc = tempdf[tempdf.type == 'original']['auroc'].item()
#ori_f1 = tempdf[tempdf.type == 'original']['f1_score'].item()

#tempdf.set_index("type").plot.bar(ax = ax3, legend = False, color = colors)
#ax3.set_xticklabels(ticklabels, rotation = 0)
#ax3.axhline(y = ori_auc, xmin = 0.04, linewidth = lnewdth, color = 'gray',linestyle = '--')
#ax3.axhline(y = ori_f1, xmin = 0.08, linewidth = lnewdth, color = 'gray',linestyle = '--')
#ax3.set_title('XGBoost', fontsize =title_fontsize)
#ax3.set_xlabel(" ")

#plt.tight_layout()
#plt.savefig(figure_path.joinpath("figure1.png"), dpi = 300)
## plt.show()


##%%
