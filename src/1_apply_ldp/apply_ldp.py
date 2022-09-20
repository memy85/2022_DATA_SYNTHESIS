
#%%
from pathlib import Path
import os, sys
from pandas import Series
from pandas.api.types import is_object_dtype, is_int64_dtype, is_float_dtype

project_dir = Path.cwd().parents[1]
os.sys.path.append(project_dir.as_posix())

#%%
from src.MyModule.utils import *
from src.MyModule.localDP import *
import yaml

#%%
config = load_config()

#%%
project_path = Path(config['path_config']['project_path'])
output_path = Path(config['path_config']['output_path'])

# redirect
data_path = output_path.joinpath('0_preprocess')
output_path = output_path.joinpath('1_apply_ldp')
#%%
original_data = read_file(data_path, 'D0.pkl')

original_data['DG_RCNF_RLPS'] = original_data['DG_RCNF_RLPS'].astype('int64')
original_data['DEAD_NFRM_DEAD'] = original_data['DEAD_NFRM_DEAD'].astype('int64')


#%% normalize all
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler((-1,1))

scaler.fit(original_data.select_dtypes('float32'))
scaled_values = scaler.transform(original_data.select_dtypes('float32'))

# %%

float_columns = original_data.select_dtypes('float32').columns.tolist()
original_data[float_columns] = scaled_values

#%%
original_data



#%% 1. static or dynamic

def apply_ldp_to_each_columns(col : Series):
    column_type = col.dtype    
    
    if (column_type is int) | (column_type):
        
    pass




#%% 2. apply LDP for continuous or categorical
columns = 




#%%