
#%%
import pandas as pd
from pathlib import Path
import pickle

project_path = Path().cwd()
input_path = project_path.joinpath('data/processed/preprocess_1')


#%%

data_path = input_path.joinpath('train_ori_50.pkl')
train = pd.read_pickle(data_path)

labelencoder = pd.read_pickle(input_path.joinpath('LabelEncoder_50.pkl'))
#%%

train_patients = labelencoder[0].inverse_transform(train['PT_SBST_NO']).tolist()
#%%
with open('train_patients.pkl', 'wb') as f:
    pickle.dump( train_patients, f)


