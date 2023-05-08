#%%
import os, sys
from pathlib import Path
import subprocess

projectPath = Path().cwd()
dataPath = projectPath.joinpath("data/processed/")

subprocess.call(projectPath.joinpath("do_process.sh"))

#%%


import pandas as pd
import numpy as np

#%%
