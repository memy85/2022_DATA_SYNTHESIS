
import os, sys
from pathlib import Path
import pickle
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
import numpy as np
import argparse
import scienceplots

projectPath = Path().cwd()
# projectPath = Path(__file__).parents[2]

os.sys.path.append(projectPath.as_posix())

#%%

from src.MyModule.utils import *

#%%
# seed = 0

#%%

def load_model(model_name, epsilon, age,  seed) :
    '''
    model_name : DecisionTree, RandomForest, XGBoost
    epsilon : 0 ~ 10,000
    '''
    # modelPath = get_path(f'data/processed/seed{seed}/4_results/models/')
    modelPath = get_path(f'data/processed/seed{seed}/4_results/best_models_age/')
    modelweightpath = modelPath.joinpath(f'age{age}/{model_name}.pkl')

    with open(modelweightpath, 'rb') as f :
        model = pickle.load(f)
    return model

#%%
model = load_model('XGBoost',0,50,0)

#%%

def load_test_data(randomseed=0) :
    test_data_path = get_path(f'data/processed/seed{randomseed}/1_preprocess/test_50.pkl')
    test = pd.read_pickle(test_data_path)


    test.drop(['PT_SBST_NO',  'REGN_CSTR_PRPS_NT_1', 'REGN_CASB_CSTR_PRPS_NM_1', 
               'REGN_TIME_DIFF_1', 'REGN_START_DIFF_1',  'REGN_CSTR_CYCL_VL_END_1'], axis=1, inplace=True)
    test_x = test.rename(columns = {"RLPS_DIFF" : "RLPS DIFF"})
    # test_x = make_train_valid_dataset(test, False, True)
    test_x.drop("REGN_CSTR_REGN_NM_1", axis=1, inplace=True)
    

    return test_x
#%%
testX = load_test_data(0)


#%%
import matplotlib.pyplot as plt

def plot_feature_importance(model, plot_name, seed):
    x = load_test_data(randomseed=seed)
    figurepath = projectPath.joinpath('figures/')

    feature_names = x.columns.tolist()

    importance = model.feature_importances_

    importance_series = pd.Series(importance, index=feature_names)
    importance_series = importance_series.sort_values(ascending=False)

    importance_series = importance_series.iloc[:5].sort_values(ascending=True)

    plt.style.use(['science', 'ieee'])
    plt.rcParams.update({"font.size": 18})

    plt.figure(figsize=(6, 6))
    importance_series.plot.barh()
    # plt.xlim(0, 0.8)  # x축 범위 설정
    # plt.ylabel("features")
    # plt.xlabel("top 5 feature importance")

    plt.tight_layout()
    plt.savefig(figurepath.joinpath(plot_name), dpi=300)
    plt.show()

    return importance_series
#%%
plot_feature_importance(model, 'test.jpg', 0)


#%%
# def plot_feature_importance(model, plot_name , seed):
#
#     x = load_test_data(randomseed=seed)
#     figurepath = projectpath.joinpath('figures/')
#
#     feature_names = x.columns.tolist()
#
#     importance = model.feature_importances_
#
#     importance_series = pd.series(importance, index=feature_names)
#     importance_series = importance_series.sort_values(ascending=false)
#
#     importance_series = importance_series.iloc[:5].sort_values(ascending=true)
#
#     plt.style.use(['science', 'ieee'])
#     plt.rcparams.update({"font.size": 24})
#
#     # fig, ax = plt.subplots(figsize=(15,15))
#     plt.figure(figsize=(6,8))
#
#     importance_series.plot.barh()
#     # ax.set_ylabel("features")
#     # ax.set_xlabel("top 5 feature importance")
#
#     # fig.tight_layout()
#     plt.savefig(figurepath.joinpath(plot_name), dpi=300)
#     plt.show()
#
#     pass

#%%

def plot_feature_importance_subplots(model, plot_name, seed) :

    X = load_test_data(randomseed=seed)
    figurePath = projectPath.joinpath('figures/')

    feature_names = X.columns.tolist()

    importance = model.feature_importances_

    importance_series = pd.Series(importance, index=feature_names)
    importance_series = importance_series.sort_values(ascending=False)

    importance_series = importance_series.iloc[:5].sort_values(ascending=True)

    plt.style.use(['science', 'ieee'])

    fig, ax = plt.subplots(2,3)
    importance_series.plot.barh(ax=ax)
    ax.set_ylabel("Features")
    ax.set_xlabel("Top 5 Feature Importance")
    fig.tight_layout()
    plt.savefig(figurePath.joinpath(plot_name), dpi=300)
    plt.show()


    pass

# #%%

# importance_series.sort_values(ascending=False)
#
# #%%
# importance_series.iloc[0:10]


#%%

def argument_parser() :
    parser = argparse.ArgumentParser()
    parser.add_argument('--random_seed', default = 0, type = int)
    args = parser.parse_args()
    return args


def main():
    args = argument_parser()
    config = load_config()
    epsilons = config['epsilon']
    ages = config['age_cut']

    models = ['DecisionTree', 'RandomForest', 'XGBoost']

    for age in ages : 
        for epsilon in epsilons :
            for model_name in models : 

                model = load_model(model_name, epsilon, age, args.random_seed)
                figure_name = model_name + f'_{epsilon}' + f'_{age}' + f'_{args.random_seed}.png'

                plot_feature_importance(model, figure_name, args.random_seed)

    pass

def main2():
    args = argument_parser()
    config = load_config()
    ages = config['age_cut']

    models = ['DecisionTree', 'RandomForest', 'XGBoost']

    for age in ages :
        for model_name in models : 

            model = load_model(model_name, 10000, age, args.random_seed)
            figure_name = model_name + f'_{age}.png'

            plot_feature_importance(model, figure_name, args.random_seed)

    pass
if __name__ == "__main__" :
    main2()




