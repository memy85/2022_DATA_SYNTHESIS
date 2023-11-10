
import os, sys
from pathlib import Path

project_path = Path().cwd().parents[0]
input_path = project_path.joinpath("data/processed/4_results/")
figure_path = project_path.joinpath("figures")

# |%%--%%| <sdo0oz0msd|2oDbJ0tooA>

import pandas as pd
training_strategy_results = input_path.joinpath('training_strategy.csv')
df = pd.read_csv(training_strategy_results)
