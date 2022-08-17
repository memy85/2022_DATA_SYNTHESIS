
#%%
from pathlib import Path
import os, sys
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

#%%
