
import pandas as pd
import numpy as np
from DataSynthesizer.DataDescriber import DataDescriber
from DataSynthesizer.DataGenerator import DataGenerator
from DataSynthesizer.ModelInspector import ModelInspector
from DataSynthesizer.lib.utils import read_json_file, display_bayesian_network
import warnings

warnings.filterwarnings(action='ignore')

# |%%--%%| <VQmhNpL7IH|JkVWi5cpLd>

threshold_value = 800
#categorical_attributes = cat
cats = {cat : True for cat in list(pd.read_csv('/home/dogu86/young_age_colon_cancer/final_src/encoded_D0_to_syn.csv').columns)}
candidate_keys = {'PT_SBST_NO': True}

#epsilon = 100

degree_of_bayesian_network = 2
num_tuples_to_generate = len(pd.read_csv('/home/dogu86/young_age_colon_cancer/final_src/encoded_D0_to_syn.csv'))*5
epsilons = [0,0,1,1,10,100,1000,10000]
#for i in range(len(cut_df)):
    # input dataset
#input_data = 'cut_df/'+df_name[4]+'.csv'
input_data = '/home/dogu86/young_age_colon_cancer/final_src/encoded_D0_to_syn.csv'

describer = DataDescriber(category_threshold=threshold_value)
#for epsilon in epsilons:
epsilon = 10000
describer.describe_dataset_in_correlated_attribute_mode(dataset_file=input_data,
                                                        epsilon=epsilon,
                                                        k=degree_of_bayesian_network,
                                                        attribute_to_is_categorical=cats,
                                                        attribute_to_is_candidate_key=candidate_keys)


description_file = f'/home/dogu86/young_age_colon_cancer/final_data/description/description_mult_encoded_{epsilon}_degree2.json'
synthetic_data = f'/home/dogu86/young_age_colon_cancer/final_data/synthetic/S0_mult_encoded_{epsilon}_degree2.csv'

describer.save_dataset_description_to_file(description_file)    
generator = DataGenerator()
generator.generate_dataset_in_correlated_attribute_mode(num_tuples_to_generate, description_file)
generator.save_synthetic_data(synthetic_data)

# |%%--%%| <JkVWi5cpLd|ufZV5EjUJ2>


