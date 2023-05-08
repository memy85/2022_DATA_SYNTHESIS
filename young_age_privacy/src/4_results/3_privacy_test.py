#%%
import os, sys
from pathlib import Path
import pandas as pd
import numpy as np
import argparse

projectPath = Path().cwd()
# projectPath = Path(__file__).parents[2]
print(projectPath)
dataPath = projectPath.joinpath("data/processed/")

#%%
os.sys.path.append(projectPath.as_posix())

from src.MyModule.privacy_test import MembershipInference, Reidentification, AttributeDisclosure
from src.MyModule.utils import *


#%%

def arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--age", default = 50, type=int)
    parser.add_argument("--random_seed", default = 0, type = int)
    parser.add_argument("--feature_split", default = 0.5,  type = float)
    args = parser.parse_args()
    return args

def load_data(projectpath, args) :
    '''
    load the datasets
    '''
    age = args.age
    random_seed = args.random_seed

    originalDict, syntheticDict = get_machine_learning_data(projectpath, age, random_seed)

    return originalDict, syntheticDict


def test_membership(args, **kwargs) :

    '''
    Membership disclosure
    - args - arguments
    - kwargs : original, holdout, synthetic
    '''

    age = args.age
    randomseed = args.random_seed

    original = kwargs['original']
    holdout = kwargs['holdout']
    synthetic = kwargs['synthetic'] # this is the dictionary 

    records = {}

    for key, synData in synthetic.items() :
        
        membershipTest = MembershipInference(original, holdout, synData)
        result = membershipTest.infer()
        records[key] = np.round(result, 3)

    print("-------------------- finished membership testing! -------------------------")
    return records
    

#%%
def test_reidentification(args, **kwargs) :

    '''
    Reidentification test
    '''
    age = args.age
    randomseed = args.random_seed
    featureSplit = args.feature_split

    original = kwargs['original']
    holdout = kwargs['holdout']
    syntheticDict = kwargs['synthetic']

    records = {}

    for key, synData in syntheticDict.items() :
        
        reidentificationTest = Reidentification(original, holdout, synData, feature_split=featureSplit)
        baseline, result = reidentificationTest.infer()

        records[key] = "{} / {}".format(np.round(baseline,3), np.round(result,3))

    print("-------------------- finished reidentification testing! -------------------------")
    return records


#%%

def test_attribute(args, **kwargs):
    '''
    Attribute disclosure test
    args -> arguments
    kwargs : original, synthetic_data dict
    '''

    age = args.age
    randomseed = args.random_seed
    config = load_config()

    original = kwargs['original']
    # holdout = kwargs['holdout']
    syntheticDict = kwargs['synthetic']

    sensitive_features = config['sensitive_features']

    records = []

    for key, synData in syntheticDict.items() :
        
        attributedisclosureTest = AttributeDisclosure(original, synData, 
                                                      sensitive_features=sensitive_features)
        answer = attributedisclosureTest.infer()

        data = {(outerkey, innerkey) : [np.round(values, 3)] for outerkey, innerDict \
                in zip(['baseline','synthetic'], answer)\
                for innerkey, values in innerDict.items()}

        df = pd.DataFrame(data).T.rename(columns = {0:key})
        records.append(df.copy())

    records = pd.concat(records, axis=1)

    return records


#%%

def save_object(path, name, object):
    with open(path.joinpath(name), 'wb') as f : 
        pickle.dump(object, f)

#%%

def main():

    args = arguments()
    age = args.age
    randomseed = args.random_seed
    feature_split = args.feature_split
    
    preprocessedDict = get_machine_learning_data(projectPath, age, randomseed)

    ## test membership
    original = preprocessedDict['original']
    holdout = preprocessedDict['holdout']
    syntheticDict = preprocessedDict['synthetic_dict']
    

    membershipResult = test_membership(args, 
                    original = original, 
                    holdout = holdout, 
                    synthetic = syntheticDict)

    ## test reidentification

    reidentificationResult = test_reidentification(args,
                                                   original = original,
                                                   holdout = holdout,
                                                   synthetic = syntheticDict,
                                                   feature_split = feature_split,
                                                   seed = randomseed
                                                   )

    ## test attribute
    attributeResult = test_attribute(args,
                                     original = original,
                                     synthetic = syntheticDict,
                                     )
    
    #### ----- save results 
    figurePath = projectPath.joinpath("figures/")

    result1 = pd.DataFrame([membershipResult, reidentificationResult], index=['MembershipInference',
                                                                    'Reidentification'])

    result1.to_csv(figurePath.joinpath('privacy_test1.csv'), index=True)

    attributeResult.to_csv(figurePath.joinpath('privacy_test2.csv'), index=True)

    pass



if __name__ == "__main__" :

    main()
