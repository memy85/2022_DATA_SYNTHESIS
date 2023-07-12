#%%
from pathlib import Path
import os
import pandas as pd
import pickle

projectPath = Path().cwd()
# projectPath = Path(__file__).parents[2]

os.sys.path.append(projectPath.as_posix())

from src.MyModule.utils import *

#%%

config = load_config()

#%%

def load_raw_table():
    dataPath = projectPath.joinpath("data/raw/D0_Handmade_ver2.csv")

    data = pd.read_csv(dataPath)

    return data

#%%

def save_variable_types_and_names(data, savepath):

    variableTable = data.dtypes.reset_index().reset_index()

    variableTable['level_0'] += 1

    variableTable.to_csv(savepath.joinpath("datatypes.csv"))

    pass


def main():

    rawtable = load_raw_table()
    figPath = projectPath.joinpath('figures/')

    save_variable_types_and_names(rawtable, figPath)

    print("the results are saved!")

    pass
    

if __name__ == "__main__" :

    main()
