
import scienceplots

from xgboost import XGBClassifier
from sklearn.neighbors import KNeighborsClassifier
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

project_path = Path(__file__).absolute().parents[2]
# project_path = Path().cwd()

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


class Classifier :

    def __init__(self, original, synthetic , threshold, identifier) :
        self.original = original
        self.synthetic = synthetic
        self.threshold = threshold
        self.identifier = identifier # identification column

    def classify(self) :
        '''
        classify model with the input data
        '''
        original, synthetic = self.preprocess_data()
        
        self.hamming_distance = original.swifter.apply(lambda x : synthetic.swifter.apply(lambda y : hamming(x, y), axis = 1),axis=1)
        self.classified_results = self.hamming_distance < self.threshold
        return self.classified_results


    def preprocess_data(self) :
        if self.check_data_columns() :
            raise Exception("the columns are different")

        if self.identifier is not None : 
            original = self.original.drop(columns = self.identifier)
            synthetic = self.synthetic.drop(columns = self.identifier)

        else : 
            original = self.original
            synthetic = self.synthetic

        return original, synthetic

    def calculate_hamming_distance(self, threshold) :

        return self.hamming_distance < threshold

    def check_data_columns(self) :
        if self.identifier is not None :
            
            try : 
                self.original[self.identifier]
            except :
                print("the original does not have the identifier column")

            try : 
                self.synthetic[self.identifier]
            except :
                print("the synthetic does not have the identifier column")

        diff = set(self.original.columns.tolist()) - set(self.synthetic.columns.tolist())
        diff = list(diff)

        if len(diff) == 0 :
            return 0

        else :

            return 1

#%%

class IdentityDisclosureRisk : 

    def __init__(self, train, test, synthetic, threshold, identifier) :

        self.train = train
        self.test = test
        self.synthetic = synthetic
        self.threshold = threshold
        self.identifier = identifier

        self.total_combination = (self.train.shape[0] * self.synthetic.shape[0]) + (self.test.shape[0] * self.synthetic.shape[0])
        self.train_total = self.train.shape[0] * self.synthetic.shape[0]
        self.test_total = self.test.shape[0] * self.synthetic.shape[0]

    def calculate_privacy_risk(self) :
        self.trainclassifier = Classifier(self.train, self.synthetic, self.threshold, self.identifier)
        self.testclassifier = Classifier(self.test, self.synthetic, self.threshold, self.identifier)

        trainhamming_dist = self.trainclassifier.classify()
        testhamming_dist = self.testclassifier.classify()

        self.tp = trainhamming_dist.sum().sum()
        self.fp = testhamming_dist.sum().sum()
        self.fn = self.train_total - self.tp
        self.tn = self.test_total - self.fp

        return self.make_score()

    def calculate_metrics(self, train_result, test_result) :
        tp = train_result.sum().sum()
        fp = test_result.sum().sum()
        fn = self.train_total - tp
        tn = self.test_total - fp

        if tp + fp == 0 : 
            precision = 0.0

        recall = tp / (tp + fn)
        f1_score = 2*(precision * recall) / (precision + recall)
        if np.isnan(f1_score) :
            f1_score = 0.0

        score = pd.DataFrame([
            {"tp" : tp, "tn" : tn, "fp" : fp, "fn" : fn, 
             "precision" : tp / (tp + fp), 
             "recall" : tp / (tp + fn),
             "f1_score" : f1_score }
        ])

        if tp + fp <= 0.0  :
            score['precision'] = 0.0

        return score
    
    def calculate_with_new_threshold(self, threshold) :
        train_result = self.trainclassifier.calculate_hamming_distance(threshold)
        test_result = self.testclassifier.calculate_hamming_distance(threshold)

        return self.calculate_metrics(train_result, test_result)

    def make_score(self) :

        precision = self.tp / (self.tp + self.fp)

        if np.isnan(precision) :
            precision = 0.0
        recall = self.tp / (self.tp + self.fn)
        self.f1_score = 2*(precision * recall) / (precision + recall)
        if np.isnan(f1_score) :
            self.f1_score = 0.0

        self.score = pd.DataFrame([
            {"tp" : self.tp, "tn" : self.tn, "fp" : self.fp, "fn" : self.fn, 
             "precision" : self.tp / (self.tp + self.fp), 
             "recall" : self.tp / (self.tp + self.fn),
             "f1_score" : self.f1_score}
            ])

        if self.tp + self.fp <= 0.0  :
            self.score['precision'] = 0.0

        return self.score
#%%

def load_pickle(path) :
    df = pd.read_pickle(path) 
    return df.copy()

#%% prepare data
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

def parse_args() :

    parser = argparse.ArgumentParser()
    parser.add_argument("--age", default = 50, type= int)
    args = parser.parse_args()
    return args

def main() :
    args = parse_args()
    train, test, synthetic_data_list = load_data(args.age)

    data_lists = []
    for idx, epsilon in enumerate(config["epsilon"]) :
        privacy = IdentityDisclosureRisk(train, test, synthetic_data_list[idx], 0.3, None)
        privacy.calculate_privacy_risk()
        df = privacy.calculate_with_new_threshold(0.3)
        df['epsilon'] = epsilon
        data_lists.append(df)
    #%%

    metric_results = pd.concat(data_lists)
    metric_results.to_csv(output_path.joinpath(f'privacy_test_{age}.csv'), index=False)

#%%
if __name__ == "__main__" :

    main()
#%%

