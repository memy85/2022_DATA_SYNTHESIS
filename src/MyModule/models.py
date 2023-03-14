#%%
import torch
import torch.nn as nn
import torch.nn.utils as utils
import torch.nn.functional as F
import torch.optim as optimizer


#%%

class BlockModule(nn.Module) :

    def __init__(self, input_size) :
        super().__init__()
        self.input_size = input_size
        self.blocks = nn.Sequential(
                nn.Linear(self.input_size, self.input_size),
                nn.BatchNorm1d(self.input_size),
                nn.Dropout(),
                nn.ReLU(),
                )

    def forward(self, x) :
        output = self.blocks(x)
        return output



class StaticGenerator(nn.Module) :

    def __init__(self, input_size, num_blocks) :
        super().__init__()
        self.input_size = input_size
        self.num_blocks = num_blocks
        
        self.model = nn.Sequential(
                *[BlockModule(self.input_size) for i in range(0, num_blocks)],

                nn.Linear(self.input_size, self.input_size),
                nn.Sigmoid(),
                )

    def forward(self, x) :
        output = self.model(x)
        return output
#%%

class StaticDiscriminator(nn.Module) :

    def __init__(self, input_size) :
        super().__init__()
        self.input_size = input_size

        self.model = nn.Sequential(
                nn.Linear(self.input_size, self.input_size),
                nn.BatchNorm1d(self.input_size),
                nn.Dropout(),
                nn.ReLU(),
                
                nn.Linear(self.input_size, self.input_size),
                nn.BatchNorm1d(self.input_size),
                nn.Dropout(),
                nn.ReLU(),

                nn.Linear(self.input_size,1),
                nn.Sigmoid()
                )


    def forward(self, x) :
        result = self.model(x)
        return result


