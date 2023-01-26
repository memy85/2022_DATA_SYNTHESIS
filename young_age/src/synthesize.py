
##
import pandas as pd
from DataSynthesizer.DataGenerator import DataGenerator
from DataSynthesizer.ModelInspector import ModelInspector
from DataSynthesizer.lib.utils import read_json_file, display_bayesian_network

import os
from pathlib import Path

folder_path = os.path.abspath("")
sys.path.append(folder_path)

from src.MyModule.utils import *

##
config = load_config()
input_path = get_path('data/raw')
output_path = get_path('data/processed/synthesize')

##
threshold_value = 800
#categorical_attributes = cat
columns_list = pd.read_csv('encoded_D0_to_syn.csv').columns
cats = {cat : True for cat in columns_list}
candidate_keys = {'PT_SBST_NO': True}

#epsilon = 100

degree_of_bayesian_network = 2
num_tuples_to_generate = len(pd.read_csv('encoded_D0_to_syn.csv'))*5
epsilons = [0,0,1,1,10,100,1000,10000]
#for i in range(len(cut_df)):
    # input dataset
#input_data = 'cut_df/'+df_name[4]+'.csv'
input_data = '/home/wonseok/2022_DATA_SYNTHESIS/young_age/data/processed/encoded_D0_to_syn.csv'

describer = DataDescriber(category_threshold=threshold_value)

for epsilon in epsilons:
    describer.describe_dataset_in_correlated_attribute_mode(dataset_file=input_data,
                                                            epsilon=epsilon,
                                                            k=degree_of_bayesian_network,
                                                            attribute_to_is_categorical=cats,
                                                            attribute_to_is_candidate_key=candidate_keys)


    description_file = f'/home/wonseok/2022_DATA_SYNTHESIS/young_age/tion/description_mult_encoded_{epsilon}_degree2.json'
    synthetic_data = f'/home/wonseok/2022_DATA_SYNTHESIS/young_age//S0_mult_encoded_{epsilon}_degree2.csv'

    describer.save_dataset_description_to_file(description_file)    
    generator = DataGenerator()
    generator.generate_dataset_in_correlated_attribute_mode(num_tuples_to_generate, description_file)
    generator.save_synthetic_data(synthetic_data)

##
import pandas as pd
import matplotlib.pyplot as plt
df = pd.read_csv("data/processed/encoded/S0_mult_encoded_10000_degree2.csv")

##
df.OVR_SURV.hist()
plt.show()

