#%%#
import pandas as pd
import argparse

from DataSynthesizer.DataGenerator import DataGenerator
from DataSynthesizer.DataDescriber import DataDescriber
from DataSynthesizer.ModelInspector import ModelInspector
from DataSynthesizer.lib.utils import read_json_file, display_bayesian_network

import os,sys
from pathlib import Path

folder_path = Path().cwd().absolute().as_posix()
sys.path.append(folder_path)

from src.MyModule.utils import *

#%%
config = load_config()
input_path = get_path('data/processed/preprocess_1')
output_path = get_path('data/processed/2_produce_data')
if not output_path.exists():
    output_path.mkdir(parents=True)

#%%

def train_bayesian_network(input_data,
                           threshold,
                           degree_of_bn,
                           categoricals,
                           candidates,
                           epsilon_value,
                           num_generate,
                           args):
    """
    This functions trains the network and saves the network with the given epsilon
    """
    print("starting describing the data..." )
    describer = DataDescriber(category_threshold=threshold)
    describer.describe_dataset_in_correlated_attribute_mode(dataset_file=input_data,
                                                            epsilon=epsilon_value,
                                                            k=degree_of_bn,
                                                            attribute_to_is_categorical=categoricals,
                                                            attribute_to_is_candidate_key=candidates)

    description_file = output_path.joinpath(f'description_mult_encoded_{epsilon_value}_{args.age}.json').as_posix()
    synthetic_data_path = output_path.joinpath(f'S0_mult_encoded_{epsilon_value}_{args.age}.csv').as_posix()

    describer.save_dataset_description_to_file(description_file)    

    print("starting generating the data..." )
    generator = DataGenerator()
    generator.generate_dataset_in_correlated_attribute_mode(num_generate, description_file)
    generator.save_synthetic_data(synthetic_data_path)

    pass

#%%

def argument_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--age", default = 50, type = int)
    args = parser.parse_args()
    return args

def main():
    args = argument_parse()
    ## settings for bayesian network

    threshold_value = 200
    # epsilon_value = 10000
    degree_of_bayesian_network = 4 # the maximum number of parents
    df_path = input_path.joinpath(f"encoded_D0_to_syn_{args.age}.csv").as_posix()
    df = pd.read_csv(df_path)

    columns_list = df.columns
    cats = {cat : True for cat in columns_list} # treat all the columns as categorical data
    candidate_keys = {'PT_SBST_NO': True}
    num_tuples_to_generate = len(df) * config['multiply']

    # for epsilons create describers

    # train_bayesian_network(df_path, 800, 2, cats, candidate_keys, epsilon_value, num_tuples_to_generate, args)

    ## If you wish to do for epsilons untoggle below

    epsilons = config['epsilon'] 
    for epsilon in epsilons:
        train_bayesian_network(input_data=df_path,
                               threshold=threshold_value,
                               degree_of_bn=degree_of_bayesian_network,
                               categoricals=cats,
                               candidates=candidate_keys,
                               epsilon_value=epsilon,
                               num_generate=num_tuples_to_generate,
                               args = args)
#%%
if __name__ == "__main__" : 
    main()


#%%
## check created file
# import pandas as pd
# import matplotlib.pyplot as plt
# df = pd.read_csv(output_path.joinpath("S0_mult_encoded_10000_degree2.csv"))

# df.OVR_SURV.hist()
# plt.show()

#%%


df = pd.read_csv(input_path.joinpath("encoded_D0_to_syn_50.csv"))
#%%

df.DEAD.value_counts()

