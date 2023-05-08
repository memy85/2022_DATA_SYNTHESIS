
import pandas as pd
from pathlib import Path
import os, sys

project_path = Path().cwd().absolute().parents[0]
os.sys.path.append(project_path.as_posix())

# |%%--%%| <5FeukjY9NR|3XfHWKfbKO>

from src.MyModule.utils import *

config = load_config()
d0_path = get_path("data/raw/D0_Handmade_ver1.1.xlsx")

# |%%--%%| <3XfHWKfbKO|vvI4sr0UfV>

data = pd.read_excel(d0_path)

# |%%--%%| <vvI4sr0UfV|H5B6Ig3vMG>

data.filter(like="BSPT")

# |%%--%%| <H5B6Ig3vMG|H1V0saY8oL>

data["BSPT_SEX_CD"].hist()
plt.show()

# |%%--%%| <H1V0saY8oL|mPW6Y6HpaT>

data["BSPT_IDGN_AGE"].hist()
plt.show()


#|%%--%%| <mPW6Y6HpaT|uneoUFJlJ9>


