#%%

from pathlib import Path
import os, sys

# project_path = Path(__file__).absolute().parents[2]
project_path = Path().cwd()
os.sys.path.append(project_path.as_posix())
#%%
from src.MyModule.utils import *

config = load_config()
input_path = get_path("data/processed/2_produce_data/synthetic_decoded/")

ouput_path = get_path("data/processed/3_evaluate_data/")
#%%
import pandas as pd
import numpy as np
#%%
class Variable:

    '''
    properties : data type and data with list
    functions : sort values and count values
    '''
    
    def __init__(self,
                 dtype : str,
                 data : list):

        self.dtype = dtype
        self.data = data
        if isinstance(self.data, pd.Series):
            self.data = self.data.tolist()

        # remove nan values
        self.remove_nan_values()

    def __repr__(self):
        print(self.data)
        return self.dtype

    def remove_nan_values(self):
        myarray = np.array(self.data)
        new_array = myarray(np.logical_not(np.isnan(my_array)))
        self.data = new_array.tolist()

    def count_values(self):
        '''
        count values in a list and returns dictionary
        '''
        NotImplementedError

    def probability_distribution(self):
        '''
        creates probability distribution
        '''
        NotImplementedError
#%%
class ContinuousVariable(Variable):

    def __init__(self, data):
        super().__init__(dtype = 'continuous', data = data)
    
    def count_values(self, bins, range = None, density = False):
        """
        returns count information
        """
        counts, _ = np.histogram(self.data, bins, range = range, density = density)
        return counts

    def probability_distribution(self, bins, range = None):
        """
        returns distribution histogram series
        """
        counts = self.count_values(bins) 
        total = counts.sum()
        return pd.Series(data = counts / total)
#%%

class CategoricalVariable(Variable):

    def __init__(self, data):
        super().__init__(dtype="categorical", data= data)
        self.categories = set(self.data) 

    def count_values(self):
        counts = [self.data.count(element) for element in self.categories]
        return counts
    
    def probability_distribution(self):
        counts = self.count_values()
        data = np.array(counts) / sum(counts)
        return pd.Series(data = data, index=self.categories)


#%%
def hellinger_distance(var1 : Variable,
                       var2 : Variable,
                       bins = None):

    """
    calculates the hellinger distance of two variables
    dtype : categorical or continuous
    var1, var2 are variables
    bins : np.arange shape
    """
    assert var1.dtype == var2.dtype, "the two data type are not the same!"
    
    if var1.dtype == 'categorical' :
        var1prob, var2prob = var1.probability_distribution(), var2.probability_distribution()

        var1prob.name = "var1"
        var2prob.name = "var2"

        df = pd.merge(var1prob, var2prob, left_index=True, right_index=True, how = "outer")
        df = df.fillna(0)

        df['distance'] = df.apply(lambda x : (np.sqrt(x['var1']) - np.sqrt(x['var2']))**2, axis=1)

        return np.round((1/np.sqrt(2))*np.sqrt(df['distance'].sum()), 3)


    else :
        '''dtype is continuous'''
        assert bins is not None, "continous variables require bins"
        var1prob, var2prob = var1.probability_distribution(bins = bins), var2.probability_distribution(bins = bins)

        var1prob.name = "var1"
        var2prob.name = "var2"

        df = pd.merge(var1prob, var2prob, left_index=True, right_index=True, how = "outer")
        df = df.fillna(0.0)

        df['distance'] = df.apply(lambda x : (np.sqrt(x['var1']) - np.sqrt(x['var2']))**2, axis=1)

        return np.round((1/np.sqrt(2))*np.sqrt(df['distance'].sum()), 3)

#%%
        


#%% calculate hellinger distance between variables

#%% load original data and synthetic data

# load original
original_path = get_path('data/raw/D0_Handmade_ver1.1.xlsx')
synthetic_path = get_path('data/processed/2_produce_data/synthetic_restore/Synthetic_data_epsilon10000_50.csv')

original = pd.read_excel(original_path) 
synthetic = pd.read_csv(synthetic_path, encoding = 'cp949')


#%% preprocessing

original


#%%

synthetic = synthetic.drop(columns = "Unnamed: 0")

#%%
set(synthetic.columns) - set(original.columns)

#%%
synthetic

#%%
original.columns
