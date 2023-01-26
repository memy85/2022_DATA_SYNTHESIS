
##
import pandas as pd
from DataSynthesizer.DataGenerator import DataGenerator
from DataSynthesizer.DataDescriber import DataDescriber
from DataSynthesizer.ModelInspector import ModelInspector
from DataSynthesizer.lib.utils import read_json_file, display_bayesian_network

import os,sys
from pathlib import Path

folder_path = os.path.abspath("")
sys.path.append(folder_path)

from src.MyModule.utils import *

##
config = load_config()
input_path = get_path('data/processed/preprocess_1')
output_path = get_path('data/processed/2_produce_data')
if not output_path.exists():
    output_path.mkdir(parents=True)

## settings for bayesian network
threshold_value = 800
degree_of_bayesian_network = 2
df_path = input_path.joinpath("encoded_D0_to_syn.csv").as_posix()
df = pd.read_csv(df_path)

columns_list = df.columns
cats = {cat : True for cat in columns_list}
candidate_keys = {'PT_SBST_NO': True}
num_tuples_to_generate = len(df)*5
epsilons = [0,0.1,1,10,100,1000,10000]

##
# for epsilons create describers

def train_bayesian_network(input_data,
                           threshold,
                           degree_of_bn,
                           categoricals,
                           candidates,
                           epsilon_value,
                           num_generate):
    """
    This functions trains the network and saves the network with the given epsilon
    """

    describer = DataDescriber(category_threshold=threshold)
    describer.describe_dataset_in_correlated_attribute_mode(dataset_file=input_data,
                                                            epsilon=epsilon_value,
                                                            k=degree_of_bn,
                                                            attribute_to_is_categorical=categoricals,
                                                            attribute_to_is_candidate_key=candidates)

    description_file = output_path.joinpath(f'description_mult_encoded_{epsilon_value}_degree2.json').as_posix()
    synthetic_data_path = output_path.joinpath(f'S0_mult_encoded_{epsilon_value}_degree2.csv').as_posix()

    describer.save_dataset_description_to_file(description_file)    

    generator = DataGenerator()
    generator.generate_dataset_in_correlated_attribute_mode(num_generate, description_file)
    generator.save_synthetic_data(synthetic_data_path)


##

train_bayesian_network(df_path, 800, 2, cats, candidate_keys, 10000, num_tuples_to_generate)

## If you wish to do for epsilons untoggle below
# for epsilon in epsilons:
#     train_bayesian_network(input_data=df,
#                            threshold=threshold_value,
#                            degree_of_bn=degree_of_bayesian_network,
#                            categoricals=cats,
#                            candidates=candidate_keys,
#                            epsilon_value=epsilon,
#                            num_generate=num_tuples_to_generate)

## check created file
import pandas as pd
import matplotlib.pyplot as plt
df = pd.read_csv(output_path.joinpath("S0_mult_encoded_10000_degree2.csv"))
##

##
# df.OVR_SURV.hist()
# plt.show()


