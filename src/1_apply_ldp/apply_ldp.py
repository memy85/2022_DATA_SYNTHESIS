
#%%
from pathlib import Path
import os, sys
project_dir = Path.cwd().parents[1]
os.sys.path.append(project_dir.as_posix())

#%%
from src.MyModule.utils import *
import yaml

#%%
config = load_config()

#%%
config
    
#%%
project_path = config['path_config']['project_path']