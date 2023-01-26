#%%
import multiprocessing
import pandas as pd
import pickle
import argparse
from pathlib import Path
import os, sys
from itertools import repeat

from DataSynthesizer.DataDescriber import DataDescriber
from DataSynthesizer.DataGenerator import DataGenerator
from DataSynthesizer.ModelInspector import ModelInspector

from pathos.pools import _ProcessPool, ProcessPool

PROJ_PATH = Path(__file__).parents[2]
sys.path.append(PROJ_PATH.joinpath('src').as_posix())

from MyModule.utils import *
import random
config = load_config()

#%%

PROJ_PATH = Path(config['path_config']['project_path'])
INPUT_PATH = PROJ_PATH.joinpath('data/processed/1_apply_bayesian/preprocess_data')
OUTPUT_PATH = PROJ_PATH.joinpath('data/processed/1_apply_bayesian/apply_bayesian')

#%%

if not OUTPUT_PATH.exists():
    OUTPUT_PATH.mkdir(parents=True)

#%%

def load_categorical():
    with open(INPUT_PATH.joinpath('categorical_columns.pkl'),'rb') as f:
        return pickle.load(f)      

#%%
def load_data(name):
    '''
    returns the data and the categoricals
    '''
    path = INPUT_PATH.joinpath(f'pt_{name}.csv')
    df = pd.read_csv(path)
    categoricals = load_categorical()
    cats = set(df.columns.tolist()) & set(categoricals)
    cats = {cat : True for cat in cats}
    return df, cats

def create_bayesian(name, epsilon):
    _, cats = load_data(name)
    thresholds = config['bayesian_config']['threshold_value']
    degree_of_network = config['bayesian_config']['degree_of_network']
    num_tuples_to_generate = config['bayesian_config']['number_of_tuples']
    
    description_file = OUTPUT_PATH.joinpath(f'out/epsilon{epsilon}/description_{name}.json')
    synthetic_data = OUTPUT_PATH.joinpath(f'out/epsilon{epsilon}/synthetic_data_{name}.csv')
    
    if not OUTPUT_PATH.joinpath(f'out/epsilon{epsilon}').exists():
        OUTPUT_PATH.joinpath(f'out/epsilon{epsilon}').mkdir(parents=True)
    
    candidate_keys = {"PT_SBST_NO":True}
    
    describer = DataDescriber(category_threshold=thresholds)
    
    describer.describe_dataset_in_correlated_attribute_mode(dataset_file=INPUT_PATH.joinpath(f'pt_{name}.csv'),
                                                        epsilon=epsilon,
                                                        k=degree_of_network,
                                                        attribute_to_is_categorical=cats,
                                                        attribute_to_is_candidate_key=candidate_keys)
    describer.save_dataset_description_to_file(description_file)
    return


def main():
    # data = PROJ_PATH.joinpath('data/processed/0_preprocess/D0.pkl')
    argparse.ArgumentParser()
    data = read_file(PROJ_PATH, 'data/processed/0_preprocess/D0.pkl')
    patients = data['PT_SBST_NO'].unique().tolist()
    epsilons  = config['epsilon']
    
    random.seed(config['random_seed'])
    # sampled_patients = random.sample(patients, 100)

    for epsilon in epsilons :
        
        with _ProcessPool(8) as p :
            p.starmap(create_bayesian, zip(patients, repeat(epsilon)))
            
if __name__ == "__main__":
    main()
