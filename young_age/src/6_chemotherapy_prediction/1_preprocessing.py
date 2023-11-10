#%%
from pathlib import Path
import os, sys
import argparse

# projectPath = Path(__file__).parents[2].as_posix()
projectPath = Path().absolute()
os.sys.path.append(projectPath.as_posix())

from src.MyModule.utils import *

#%% define argparse

config = load_config()

#%% load config

def arg_parser() :
    parser = argparse.ArgumentParser()
    parser.add_argument("--random_seed", default = 0, type=int)
    parser.add_argument("--age", default = 50, type = int)
    args = parser.parse_args()
    return args


#%% load data 
# load real

def prepare_original_data(seed, age) :
    original_data_path = get_path(f"data/processed/seed{seed}/1_preprocess/encoded_D0_{age}.csv")
    data = pd.read_csv(original_data_path)

    bind_columns = pd.read_pickle(projectPath.joinpath(f"data/processed/seed{seed}/1_preprocess/bind_columns_{age}.pkl"))

    tables= []
    for col in bind_columns:
        tables.append('_'.join(col.split('_')[0:1]))
        
    try:
        data = data.drop(columns = 'Unnamed: 0')

    except:
        pass

    data = data.astype(str)

    # for col in data.iloc[:,11:]:
    #     data[col] = data[col].str.replace('r','')
    #     
    # decoded = decode(data.iloc[:,11:], tables, bind_columns)
    # decoded.columns = bind_columns

    data.reset_index(drop=True, inplace=True)
    
    # data = pd.concat([data.iloc[:,:11],decoded],axis=1)
    data = data.rename(columns = {'RLPS DIFF' : 'RLPS_DIFF'})
    data = data.drop(columns = "PT_SBST_NO")

    return data

#%%

def prepare_synthetic_data(seed, age) :

    epsilons = config['epsilon']
    synthetic_data_list = []

    bind_columns = pd.read_pickle(projectPath.joinpath(f"data/processed/seed{seed}/1_preprocess/bind_columns_{age}.pkl"))
    input_path = get_path(f"data/processed/seed{seed}/2_produce_data")

    tables= []
    for col in bind_columns:
            tables.append('_'.join(col.split('_')[0:1]))

    for epsilon in epsilons:

        syn = pd.read_csv(input_path.joinpath(f'S0_mult_encoded_{epsilon}_{age}.csv'))

        try:
            syn = syn.drop('Unnamed: 0', axis=1)
        except:
            pass
        syn = syn.astype(str)

        for col in syn.iloc[:,11:]:
            syn[col] =syn[col].str.replace('r','')
            
        decoded = decode(syn.iloc[:,11:], tables, bind_columns)
        decoded.columns = bind_columns
        
        syn = pd.concat([syn.iloc[:,:11],decoded],axis=1)
        syn = syn.rename(columns = {'RLPS DIFF' : 'RLPS_DIFF'})
        syn = syn.drop(columns = 'PT_SBST_NO')

        synthetic_data_list.append(syn)

    return synthetic_data_list
#%%
ori_data = prepare_original_data(0, 50)
syntheticDataList = prepare_synthetic_data(0, 50)

#%%
ori_data

#%%

def load_data() :
    config = load_config()
    projectPath = config["project_path"]

    # load real data



    

    pass



#%% preprocess
# 



#%%

def main() :
    args = arg_parser()

    pass

#%%

if __name__ == "__main__"  :
    main()


