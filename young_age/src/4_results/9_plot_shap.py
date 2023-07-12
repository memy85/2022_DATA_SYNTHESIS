#%%
import pandas as pd
import pickle
from pathlib import Path
import os, sys

import shap

# project_path = Path().cwd()
project_path = Path(__file__).parents[2]
os.sys.path.append(project_path.as_posix())

from src.MyModule.utils import *
#%%
seed = 0

result_path = get_path(f'data/processed/seed{seed}/4_results/shap_values')
figure_path = get_path('figures')

#%%
def load_shap_value(model_name, epsilon, age) :

    shap_path = result_path.joinpath(f"{model_name}_{age}_{epsilon}.pkl")
    with open(shap_path, 'rb') as f:
        shap_values = pickle.load(f)
    return shap_values

def load_test_data(randomseed=0) :
    test_data_path = get_path(f'data/processed/seed{randomseed}/1_preprocess/test_50.pkl')
    test = pd.read_pickle(test_data_path)

    test_x = test.drop(['DEAD','DEAD_DIFF','PT_SBST_NO','OVR_SURV'], axis=1)
    test_y = test['DEAD']

    return test_x
#%%
def show_shapley(model_name, epsilon, age,
                 figurePath, figureName) :
    '''
    Input : model_name, epsilon, age
        model_name : DecisionTree, RandomForest, XGBoost
    Output : Figure
    ''' 

    testD = load_test_data()

    shapvalue = load_shap_value(model_name, epsilon, age)

    shapvalue.display_data = testD.values

    shap.plots.bar(shapvalue.abs.max(0))
    plt.show()

    pass
#%%

testD = load_test_data()

#%%
testD

#%%
shapvalue = load_shap_value("DecisionTree", 0.1, 50)

#%%
shapvalue.display_data = testD.values

#%%
import shap
import matplotlib.pyplot as plt

shap.plots.bar(shapvalue.abs.mean(0))
plt.show()

#%%
shap.summary_plot(shapvalue)

#%%
shapvalue.abs.max(0)

#%%
shap.plots.bar(shapvalue)
plot.show()



