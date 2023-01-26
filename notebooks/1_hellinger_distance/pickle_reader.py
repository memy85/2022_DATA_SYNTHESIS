#%%
import pandas as pd
from pathlib import Path
import os, sys

PROJECT_PATH = Path(__file__).resolve().parents[2]

os.sys.path.append(PROJECT_PATH.as_posix())
from src.MyModule.utils import *
config = load_config()

#%%
PROJECT_PATH = Path(config['path_config']['project_path'])
D0_PATH = PROJECT_PATH.joinpath('data/processed/0_preprocess/D0.pkl')
S1_PATH = PROJECT_PATH.joinpath('data/processed/2_restore/restore_to_s1/')


def reverse_encoding(encoding):
    encoding.pop('PT_SBST_NO')
    def reverse(book):
        return {v : k for k,v in book.items()}
    return {k : reverse(book) for k, book in encoding.items()}
#%%
def read_pickle_file(file, synthetic:bool):
    if synthetic :
        for eps in config['epsilon']:
            
            try :
                path = file.joinpath(f'S1_{eps}.pkl')
                df = pd.read_pickle(path)
                df.to_csv(f'data/S1_{eps}.csv', index=False)
            except :
                pass
    else :
        df = pd.read_pickle(file)
        with open('/home/wonseok/2022_DATA_SYNTHESIS/data/processed/0_preprocess/encoding.pkl', 'rb') as f:
            encoding = pickle.load(f)
        encoding_rev = reverse_encoding(encoding)
        df = df.replace(encoding_rev)
        
        df.replace()
        df.to_csv('data/D0.csv', index=False)
        

def main():
    read_pickle_file(D0_PATH, False)
    read_pickle_file(S1_PATH, True)

if __name__ == "__main__":
    main()