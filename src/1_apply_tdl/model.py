
#%%
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

#%%


#%%


class Creator(nn.Module):
    
    def __init__(self, metadata : dict):
        '''
        given a metadata, the creator creates a generator for each data types
        '''
        self.metadata = metadata
        
        if 'categorical-static' in self.metadata:      
            self.cat_stat_generator = Generator('categorical','static')
        
        elif 'continuous-static' in self.metadata:
            self.cont_stat_generator = Generator('continuous','static')
        
        elif 'categorical-dynamic' in self.metadata:
            self.cat_dyn_generator = Generator('categorical','dynamic')

        elif 'continuous-dynamic' in self.metadata:
            self.cont_stat_generator = Generator('continuous','static')
        
        
        # make loss function
        self.gen_criterion, self.dis_criterion = return_criterion(self.create_type)
    
    def forward(self, cat_stat, cont_stat, cat_dyn, cont_dyn):
        '''
        cat_stat : shape -> (batch_size, feature_size)
        cont_stat : shape -> (batch_size, feature_size)
        cat_dyn : shape -> (batch_size, sequence_length, feature_size)
        cont_dyn : shape -> (batch_size, sequence_length, feature_size)
        '''
        random_data = torch.rand_like(x)
        
        synthetic_data = self.generator.generate(random_data)
        # synthetic data shape : (batch_size, x_shape[1], x_shape[2])
        # or (batch_size, x_shape[1])
        
        d_syn = self.discriminator(synthetic_data)
        d_ori = self.discriminator(x)
        return d_syn, d_ori

class Generator(nn.Module):
    
    def __init__(self, creator_type):
        self.creator_type  = creator_type
    
    def forward(self, x):
        
        return 0
    

class Discriminator(nn.Module):
    
    def __init__(self, creator_type):
        
        self.creator_type = creator_type
        
    def forward(self, x):
        
        return 0
        
        