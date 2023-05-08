#%%
import pandas as pd
import numpy as np
import pickle

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import auc
from sklearn.metrics import accuracy_score


#%%


class MembershipInference :

    '''
    Membership inference problem
    original : original data without patient_id that were used for synthesizing 
               data
    holdout : original data that were not used in patient id
    synthetic : synthetic data without the patient id
    '''

    def __init__(self, original, holdout, synthetic) :
        self.original = original.copy()
        self.holdout = holdout.copy()
        self.synthetic = synthetic.copy()

        self.classifier = KNeighborsClassifier()


    def infer(self) :
        self.preprocess()

        # train model
        synthetic = self.synthetic.iloc[:, :-1]
        original = self.synthetic.iloc[:, :-1]
        holdout = self.holdout.iloc[:, :-1]

        testset = pd.concat([original, holdout], ignore_index=True)
        testy1 = np.zeros(original.shape[0])
        testy2 = np.ones(holdout.shape[0])
        testy = np.concatenate([testy1, testy2])

        trainy = np.ones(synthetic.shape[0])

        self.classifier.fit(synthetic, trainy)
        pred = self.classifier.predict(testset)

        score = accuracy_score(pred, testy)
        return score


    def preprocess(self) :
        self.original['label'] = 'original'
        self.holdout['label'] = 'holdout'
        self.synthetic['label'] = 'original'


#%%
class Reidentification :

    '''
    original : original data without the patient_id index
    synthetic : synthetic data without the patient_id index 
    feature_split : proportion (0 ~ 1)
    '''

    def __init__(self, original, synthetic, feature_split) :
        self.original = original
        self.synthetic = synthetic
        self.feature_split = feature_split

        self.features = len(original.columns)

        self.classifier = KNeighborsClassifier()

    def infer(self) :
        
        pass

    def split_data(self) :
        self.subfeatureIdx = int(self.features * self.feature_split) # length * 0.5
        self.split1 = self.synthetic.loc[:, :self.subfeatureIdx].copy()
        self.split2 = self.synthetic.loc[:, self.subfeatureIdx:].copy()

        return 0


#%%
class AttributeDisclosure :
    '''
    original : original data without the patient_id index
    synthetic : synthetic data without the patient_id index 
    feature_split : proportion (0 ~ 1)
    '''

    def __init__(self, original, synthetic, feature_split) :
        self.original = original
        self.synthetic = synthetic

        self.classifier = KNeighborsClassifier()

    def infer(self) :
        pass


#%%
import scienceplots
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from scipy.spatial.distance import hamming
import pandas as pd
import numpy as np
import os
import sys
from pathlib import Path
import argparse
import swifter

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

output_path = get_path("data/processed/4_results/")

figure_path = get_path("figures")

if not output_path.exists():
    output_path.mkdir(parents=True)

#%%

def load_pickle(path) :
    df = pd.read_pickle(path) 
    return df.copy()

def load_data(age) :

    test_data_path = get_path(f"data/processed/preprocess_1/test_{age}.pkl")
    test_data = load_pickle(test_data_path)
    label_encoder_path = get_path(f"data/processed/preprocess_1/LabelEncoder_{age}.pkl")
    label_encoder = load_pickle(label_encoder_path)

    matched_original_path = get_path(f'data/processed/3_evaluate_data/matched_org_{age}.pkl')
    original_path = get_path(f"data/processed/preprocess_1/original_{age}.pkl")
    matched_original_data = pd.read_pickle(matched_original_path)
    original = pd.read_pickle(original_path)

    synthetic_data_path_list = []
    for epsilon in config['epsilon'] : 
        synthetic_path = get_path(f'data/processed/3_evaluate_data/matched_syn_{epsilon}_{age}.pkl')
        synthetic_data_path_list.append(synthetic_path)

    synthetic_data_list = list(map(load_pickle, synthetic_data_path_list))

    matched_original_data["PT_SBST_NO"] = original["PT_SBST_NO"]
    test_data['PT_SBST_NO'] = label_encoder[0].inverse_transform(test_data['PT_SBST_NO'])

    total = set(matched_original_data['PT_SBST_NO'].tolist())
    test_cohort = set(test_data['PT_SBST_NO'].tolist())
    train_cohort = total - test_cohort

    train = matched_original_data[matched_original_data.PT_SBST_NO.isin(train_cohort)].copy()
    test = matched_original_data[matched_original_data.PT_SBST_NO.isin(test_cohort)].copy()

    train.drop(columns = 'PT_SBST_NO', inplace=True)
    test.drop(columns = "PT_SBST_NO", inplace=True)

    return train, test, synthetic_data_list
#%%
train, test, synList = load_data(age)

#%%
mif = MembershipInference(train, test, synList[0])





