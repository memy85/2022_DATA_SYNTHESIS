#%%
import pandas as pd
import pickle
from pathlib import Path
import os, sys

import shap

project_path = Path().cwd()
# project_path = Path(__file__).parents[2]
os.sys.path.append(project_path.as_posix())

from src.MyModule.utils import *
#%%

result_path = get_path('data/processed/4_results/shap_values')
figure_path = get_path('figures')

#%%
def load_shap_value(model_name, epsilon, age) :

    shap_path = result_path.joinpath(f"{model_name}_{age}_{epsilon}.pkl")
    with open(shap_path, 'rb') as f:
        shap_values = pickle.load(f)
    return shap_values

def load_test_data() :
    test_data_path = get_path('data/processed/preprocess_1/test_50.pkl')
    test = pd.read_pickle(test_data_path)

    test_x = test.drop(['DEAD','DEAD_DIFF','PT_SBST_NO','OVR_SURV'], axis=1)
    test_y = test['DEAD']

    return test_x

testD = load_test_data()

#%%
shapvalue =  load_shap_value("DecisionTree", 0.1, 50)

#%%
shap.summary_plot(shapvalue, testD)

#%%
import matplotlib.pyplot as plt

testdata = load_test_data()

#%%


shap.plots.bar(extremecase_results['shap_values'][0].base_values)
plt.show()

#%%

