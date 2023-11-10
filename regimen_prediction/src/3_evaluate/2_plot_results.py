#%%

import os, sys
from pathlib import Path

project_path = Path().cwd()

os.sys.path.append(project_path.as_posix())

from src.MyModule.distribution_comparison import *
from src.MyModule.ml_function import *
from src.MyModule.utils import *

print(f" this is project path : {project_path} ")

#%% load results
config = load_config()
projectPath = Path(config['project_path'])
seed = 0
#%%

def load_model(model_name, epsilon, age,  seed) :
    '''
    model_name : DecisionTree, RandomForest, XGBoost
    epsilon : 0 ~ 10,000
    '''
    # modelPath = get_path(f'data/processed/seed{seed}/4_results/models/')
    modelPath = get_path(f'data/processed/seed{seed}/4_results/best_models_age/')
    modelweightpath = modelPath.joinpath(f'age{age}_eps{epsilon}/{model_name}.pkl')

    with open(modelweightpath, 'rb') as f :
        model = pickle.load(f)
    return model

#%%


def load_test_data(randomseed=0) :
    test_data_path = get_path(f'data/processed/seed{randomseed}/4_results/testX.pkl')
    y_path = get_path(f"data/processed/seed{randomseed}/4_results/testY.pkl")
    test = pd.read_pickle(test_data_path)
    y = pd.read_pickle(y_path)
    test = test.astype("float64")

    return test, y
#%%
testX, y = load_test_data(0)

model = load_model('XGBoost',10000,50,0)

#%%

result = model.predict(testX)

#%%
testX["REGN_PREDICTED"] = result
testX["REGN_CSTR_REGN_NM_1"] = y

#%%
labeldict_path = get_path(f"data/processed/seed{seed}/1_preprocess/label_dict_50.pkl")
labelDict = pd.read_pickle(labeldict_path)
idencodePath = get_path(f"data/processed/seed{seed}/1_preprocess/LabelEncoder_50.pkl")
idEncoder = pd.read_pickle(idencodePath)

#%%

encoder = labelDict['REGN_CSTR_REGN_NM_1']
inversedEncoder = {v : k for k, v in encoder.items()}

#%%
testX['REGN_PREDICTED'] = testX['REGN_PREDICTED'].replace(inversedEncoder)
testX['REGN_CSTR_REGN_NM_1'] = testX['REGN_CSTR_REGN_NM_1'].replace(inversedEncoder)


#%% restore patient id

test = pd.read_pickle(projectPath.joinpath("data/processed/seed0/1_preprocess/test_50.pkl"))

patientID = idEncoder[0][1].inverse_transform(test['PT_SBST_NO'])

testX["PT_SBST_NO"] = patientID

testX.to_csv(projectPath.joinpath("data/processed/seed0/4_results/testResults.csv"))







