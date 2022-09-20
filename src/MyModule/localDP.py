from numpy import array
import numpy as np
from pandas import Series

from syndp.algorithm.original_timedp import timeseries_dp as tdp
from syndp.mechanism.bounded_laplace_mechanism import boundedlaplacemechanism
from syndp.mechanism.bounded_laplace_mechanism import boundedlaplacemechanism as blm


def continuous_value_ldp(original_data : array, epsilon):
    '''
    local differential privacy mechanism for continuous values
    leverages bounded laplace mechanism
    '''
    check_is_na = lambda data : not any(np.isnan(data))  
    assert check_is_na(original_data) , "there is a nan value"
    
    if len(original_data) < 2 : 
        boundedlaplacemechanism()
    
    synthesized = tdp(original_data, epsilon)
    return synthesized



def categorical_value_ldp(original_data : array, all_values : array, epsilon):
    '''
    utilize randomized response to synthesize categorical variables
    '''
    dimension = len(all_values)
    
    def general_randomized_response(original_value):
        if np.isnan(original_value):
            mask = np.isnan(all_values)
        else :
            mask = original_value == all_values    
        probs = mask.copy()
        
        probs[mask] = np.e**epsilon / (np.e**epsilon + dimension -1)
        probs[~mask] = 1 / (np.e**epsilon + dimension -1)
        
        value = np.random.choice(all_values, 1, probs)
        
    grr = np.vectorize(general_randomized_response)
    synthesized = grr(original_data)

    return synthesized