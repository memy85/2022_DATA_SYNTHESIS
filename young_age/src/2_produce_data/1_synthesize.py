#%%#
import pandas as pd
import argparse
import warnings
import os,sys
from pathlib import Path

project_path = Path(__file__).absolute().parents[2]
# project_path = Path().cwd()
sys.path.append(project_path.as_posix())

from src.MyModule.utils import *

warnings.filterwarnings('ignore')

from DataSynthesizer.DataGenerator import DataGenerator
from DataSynthesizer.DataDescriber import DataDescriber
from DataSynthesizer.ModelInspector import ModelInspector
from DataSynthesizer.lib.utils import read_json_file, display_bayesian_network

#%%
config = load_config()
input_path = get_path('data/processed/preprocess_1')
output_path = get_path('data/processed/2_produce_data')
if not output_path.exists():
    output_path.mkdir(parents=True)

#%%
def load_cohort_data(age):
    """
    loads cohort data
    """
    files = os.listdir(input_path.absolute().as_posix())
    filtered_age = list(filter(lambda x : x[-6:-4] == str(age), files))
    cohort_data = list(filter(lambda x : "cohort" in x, filtered_age))
    return cohort_data

def extract_cohort_information(cohort_file_name):
    """
    return cohort information
    """
    cohort = str.split(cohort_file_name, '/')[-1].split('_')[1]
    return str(cohort)

#%%

def train_bayesian_network(input_data,
                           which_cohort,
                           threshold,
                           degree_of_bn,
                           categoricals,
                           candidates,
                           epsilon_value,
                           num_generate,
                           seed,
                           args):
    """
    This functions trains the network and saves the network with the given epsilon
    input_data : path for the input data
    """
    
    print("starting describing the data..." )

    describer = DataDescriber(category_threshold=threshold)
    describer.describe_dataset_in_correlated_attribute_mode(dataset_file=input_data,
                                                            epsilon=epsilon_value,
                                                            k=degree_of_bn,
                                                            attribute_to_is_categorical=categoricals,
                                                            attribute_to_is_candidate_key={'PT_SBST_NO' : True},
                                                            seed = seed)

    seed_output_path = output_path.joinpath(f'seed{seed}')
    description_file = output_path.joinpath(f'description_{which_cohort}_{epsilon_value}_{args.age}.json').as_posix()
    synthetic_data_path = output_path.joinpath(f'S0_{which_cohort}_{epsilon_value}_{args.age}.csv').as_posix()

    describer.save_dataset_description_to_file(description_file)    

    print("starting generating the data..." )

    generator = DataGenerator()
    generator.generate_dataset_in_correlated_attribute_mode(num_generate, description_file)
    generator.save_synthetic_data(synthetic_data_path)

    pass
#%%
def clean_unused_files(path) :
    files = os.listdir(path)

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
    degree_of_bayesian_network = 2 # the maximum number of parents
    
    cohort_files = load_cohort_data(args.age)
    cohort_file_path = [input_path.joinpath(file).as_posix() for file in cohort_files]

    ## If you wish to do for epsilons untoggle below

    epsilons = config['epsilon'] 
    seeds = [i for i in range(0,config['randomseed'])]

    for seed in seeds :
        output_path.joinpath(f'seed{seed}').mkdir(parents=True, exist_ok = True)
        seed_output_path = output_path.joinpath(f'seed{seed}')

        for epsilon in epsilons:

            for df_path in cohort_file_path:

                df = pd.read_csv(df_path)

                if df.shape[0] < 2 : 
                    cohort_file_path.remove(df_path)
                    continue

                print("df path is {}".format(df_path))

                columns_list = df.columns.tolist()
                cats = {cat : True for cat in columns_list} # treat all the columns as categorical data
                del cats['PT_SBST_NO']

                cohort_info = extract_cohort_information(df_path)
                num_tuples_to_generate = df.shape[0] * config["multiply"] 
                candidate_keys = {'PT_SBST_NO': True}

                train_bayesian_network(input_data=df_path,
                                       which_cohort= cohort_info,
                                       threshold=threshold_value,
                                       degree_of_bn=degree_of_bayesian_network,
                                       categoricals=cats,
                                       candidates=candidate_keys,
                                       epsilon_value=epsilon,
                                       num_generate=num_tuples_to_generate,
                                       seed = seed,
                                       args = args)

        print("finished synthesizing for all epsilons and cohorts")

        cohort_null_columns_dict = pd.read_pickle(input_path.joinpath(f"null_columns_dict_{args.age}.pkl"))

    #%%
        for epsilon in epsilons:
            df_list = []
            cohort_info_list = []

            for df_path in cohort_file_path:
                cohort = extract_cohort_information(df_path)
                try : 
                    syn_df =  pd.read_csv(seed_output_path.joinpath(f"S0_{cohort}_{epsilon}_{args.age}.csv"))     
                except :
                    print(f'cohort {cohort} does not exist')
                    continue

                null_columns = cohort_null_columns_dict[cohort]

                if len(null_columns) < 1 :
                    pass
                else :
                    syn_df[null_columns] = np.NaN
                df_list.append(syn_df)

            synthesized = pd.concat(df_list, axis=0)
            synthesized = synthesized[["PT_SBST_NO","BSPT_SEX_CD","BSPT_IDGN_AGE","BSPT_FRST_DIAG_NM", "BSPT_STAG_CLSF_CD", "BSPT_STAG_VL","RLPS","RLPS DIFF", "DEAD", "DEAD_DIFF", "OVR_SURV", "BPTH", "OPRT","SGPT", "MLPT","IMPT","REGN"]]

            
            synthesized.to_csv(seed_output_path.joinpath(f"S0_mult_encoded_{epsilon}_{args.age}.csv"), index=False)
            print("saved for epsilon {}".format(epsilon))

#%%
if __name__ == "__main__" : 
    main()
    print("finished synthesizing for all epsilons and cohorts")


##%%
#threshold_value = 200
#degree_of_bayesian_network = 2 # the maximum number of parents

#cohort_files = load_cohort_data(50)
#cohort_file_path = [input_path.joinpath(file).as_posix() for file in cohort_files]

## df_path = input_path.joinpath(f"encoded_D0_to_syn_{args.age}.csv").as_posix()
#sample = pd.read_csv(cohort_file_path[0])

##%%

#columns_list = sample.columns
#cats = {cat : True for cat in columns_list} # treat all the columns as categorical data
#candidate_keys = {'PT_SBST_NO': True}

#epsilons = config['epsilon'] 


##%%

#args = {"age":50}

#for epsilon in epsilons:

#    for df_path in cohort_file_path[-2:-1]:

#        cohort_info = extract_cohort_information(df_path)
#        num_tuples_to_generate = len(pd.read_csv(df_path)) * config["multiply"] 
#        df = pd.read_csv(df_path)

#        train_bayesian_network(input_data=df_path,
#                               cohort = cohort_info,
#                               threshold=threshold_value,
#                               degree_of_bn=degree_of_bayesian_network,
#                               categoricals=cats,
#                               candidates=candidate_keys,
#                               epsilon_value=epsilon,
#                               num_generate=num_tuples_to_generate,
#                               args = args)
##%%

#pd.read_csv(cohort_file_path[-2])


##%%
#cohort_file_path
#dfList = []
#for file in cohort_file_path:
#    dfList.append(pd.read_csv(file))

##%%
#dfList[2].isna().all()

##%%


#cols = dfList[3].columns.tolist()
#[df[col].value_counts().sum() for col in cols]
#idx = [df[col].value_counts().sum() for col in cols]


##%%

#[df[col].value_counts().sum() for col in cols]

##%%
#df.iloc[:, 9]

##%%
#df.iloc[:, 9]

#def check_null_column(data):
#    """
#    return columns that are null
#    """
#    columns = data.isnull().all()[data.isnull().all()].index.tolist()
#    if len(columns) < 1:
#        return 0
#    else :
#        return columns

    

#%%

#df = pd.read_csv(output_path.joinpath("S0_mult_encoded_10000_50.csv"))


##%%

#bind_columns = pd.read_pickle(project_path.joinpath(f"data/processed/preprocess_1/bind_columns_50.pkl"))


##%%
#bind_columns

##%%

#df.columns

#%%

#pd.read_csv(input_path.joinpath("unmodified_D0_"))


##%%

#raw_path = get_path("data/raw/D0_Handmade_ver1.1.xlsx")
#data = pd.read_excel(raw_path)

##%%
#new_df_path = get_path("data/processed/2_produce_data/S0_mult_encoded_0.1_50.csv")
#new_df= pd.read_csv(new_df_path)
##%%

#len(new_df.columns)


##%%

#len(df.columns)


