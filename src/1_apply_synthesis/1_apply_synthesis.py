#%%
import pandas as pd
from pathlib import Path
import os, sys


project_path = Path().cwd()
# project_path = Path(__file__).absolute().parents[2]

os.sys.path.append(project_path.as_posix())

from src.MyModule.preprocessing import get_dataloader
from src.MyModule.postprocessing import StaticRestorer, restore_manually 
from src.MyModule.models import StaticGenerator, StaticDiscriminator
from src.MyModule.train_model import train_gan
from src.MyModule.utils import *

from src.CTGAN.ctgan.synthesizers import ctgan

#%%
data_path = project_path.joinpath('data/raw')
file_name = 'LUNG_PT_BSNF.xlsx'
mapping_path = project_path.joinpath('mapping/lung/PT_BSNF.yaml')

batch_size = 64
shuffle = True
epochs = 100
lr = 0.01

dataloader = get_dataloader(data_path, file_name, mapping_path, batch_size, shuffle)
input_size = dataloader.dataset.processed_dataset.shape[1]

# staticgenerator = StaticGenerator(input_size, 10)
# staticdiscriminator = StaticDiscriminator(input_size)
#%%
discrete = dataloader.dataset.processor.mappings.discrete
#%%
data = dataloader.dataset.processed_dataset
data = data.numpy()

#%%
gan = ctgan.CTGAN()
#%%
gan.fi

#%%
ctgan._generator


#%%
ctgan = CTGAN(epochs = epochs)
ctgan.fit(data)
samples = ctgan.sample(10000)
#%%
with torch.no_grad() :
    ctgan._generator

#%%
samples

#%%
result = restore_manually(dataloader.dataset, samples)
#%%
result.BSPT_SEX_CD.value_counts()
result.BSPT_STAG_VL.value_counts()

#%%
original = dataloader.dataset.df

#%%
dataloader.dataset.processed_dataset

#%%
original.BSPT_SEX_CD.value_counts()
original.BSPT_STAG_VL.value_counts()


#%%
result = train_gan(staticdiscriminator, staticgenerator, dataloader, epochs, lr)

#%%
generator = result[0]

#%%
static_dataset = dataloader.dataset
#%%
restorer = StaticRestorer(generator, static_dataset, 50000)

#%%
restored = restorer.restore_to_static()

#%%
import matplotlib.pyplot as plt

restored.BSPT_SEX_CD.hist()
plt.show()
#%%
restored.BSPT_SEX_CD.value_counts()

#%%
restorer.raw_output.shape

#%%
static_dataset.df



#%%
import matplotlib.pyplot as plt
plt.plot(result[3], label = 'generator')
plt.plot(result[2], label = 'discriminator')
plt.legend()
plt.show()

#%%
