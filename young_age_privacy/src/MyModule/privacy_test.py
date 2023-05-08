#%%
import pandas as pd
import numpy as np
import pickle
import random

from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from sklearn.metrics import auc
from sklearn.metrics import accuracy_score

#%%


class MembershipInference :

    '''
    Membership inference problem
    original : original data without patient_id that were used for synthesizing 
               data
    holdout : original data that were not used in patient id
    synthetic : synthetic data without the patient id
    '''

    def __init__(self, original, holdout, synthetic) :
        self.original = original.copy()
        self.holdout = holdout.copy()
        self.synthetic = synthetic.copy()

        self.classifier = KNeighborsClassifier()


    def infer(self) :

        # train model
        trainX = self.synthetic.iloc[:, :-1] # we train the model with synthetic data
        trainy = np.ones(self.synthetic.shape[0]) # we only fit the data with the synthetic datas

        original = self.synthetic.iloc[:, :-1]
        holdout = self.holdout.iloc[:, :-1]

        testset = pd.concat([original, holdout], ignore_index=True)
        testy1 = np.zeros(original.shape[0])
        testy2 = np.ones(holdout.shape[0])
        testy = np.concatenate([testy1, testy2])

        # first train, fit the model with synthetic data
        self.classifier.fit(trainX, trainy)

        # test the model with the concatenated original data
        pred = self.classifier.predict(testset)

        score = accuracy_score(pred, testy)
        return score

#%%
class Reidentification :

    '''
    original : original data *without the patient_id index
    synthetic : synthetic data *without the patient_id index 
    feature_split : proportion (0 ~ 1)

    we have to split the data and using the proportional data, we infer the most right row.
    We check whether the rows can be matched
    '''

    def __init__(self, 
                 original, 
                 holdout, 
                 synthetic, 
                 feature_split: float, 
                 seed = 0) :

        self.original = original
        self.holdout = holdout
        self.synthetic = synthetic
    
        assert (original.columns == synthetic.columns).all(), "the columns should be the same!"
 
        self.feature_split = feature_split
        self.features_length = len(original.columns)
        
        self.seed = 0
        random.seed(self.seed)
        columns = original.columns.tolist()
        random.shuffle(columns)
        self.features = columns

        self.classifier1 = KNeighborsClassifier()
        self.classifier2 = KNeighborsClassifier()
        
        # split the data in the init method!-> returns self.synthetic1, self.synthetic2
        self.split_data()

        # create labels
        self.originalY = np.ones(self.original.shape[0])
        self.holdoutY = np.zeros(self.holdout.shape[0])
        self.syntheticY = np.ones(self.synthetic.shape[0])

    def infer(self) :

        # first calculate baseline
        baselineScore = self.baseline()

        # first fit the data to the synthetic1 and synthetic2
        self.classifier1.fit(self.synthetic1, self.syntheticY)
        self.classifier2.fit(self.synthetic2, self.syntheticY)

        # classify the data : synthetic1, synthetic2
        pred1 = self.classifier1.kneighbors(self.original, n_neighbors=1, return_distance=False)
        pred2 = self.classifier2.kneighbors(self.original, n_neighbors=1, return_distance=False)

        syntheticScore = sum(pred1 == pred2)/len(pred1)
        self.syntheticScore = syntheticScore

        return baselineScore, syntheticScore

    def baseline(self) :
        self.classifier1.fit(self.holdout1, self.holdoutY)
        self.classifier2.fit(self.holdout2, self.holdoutY)

        # classify the data : synthetic1, synthetic2
        pred1 = self.classifier1.kneighbors(self.original, n_neighbors=1, return_distance=False)
        pred2 = self.classifier2.kneighbors(self.original, n_neighbors=1, return_distance=False)

        score = sum(pred1 == pred2)/len(pred1)
        self.baseline_score = score
        return score

    def split_data(self) :

        # assert features first
        self.subfeatureIdx = int(self.features_length * self.feature_split) # length * 0.5
        self.featureset1, self.featureset2 = self.features[:self.subfeatureIdx], \
                self.features[self.subfeatureIdx:]


        # split data for original-holdout data
        self.holdoutsplit1 = self.holdout.loc[:, self.featureset1].copy()
        holdoutdummy1  = self.holdoutsplit1.copy()
        holdoutdummy1.loc[:,:] = 0

        self.holdoutsplit2 = self.holdout.loc[:, self.featureset2].copy()
        holdoutdummy2 = self.holdoutsplit2.copy()
        holdoutdummy2.loc[:,:] = 0

        self.holdout1 = pd.concat([self.holdoutsplit1, holdoutdummy2], axis=1)
        self.holdout2 = pd.concat([holdoutdummy1, self.holdoutsplit2], axis=1)

        # split data for synthetic data
        self.split1 = self.synthetic.loc[:, self.featureset1].copy()
        split1dummy = self.split1.copy()
        split1dummy.loc[:,:] = 0 

        self.split2 = self.synthetic.loc[:, self.featureset2].copy()
        split2dummy = self.split2.copy()
        split2dummy.loc[:,:] = 0 

        self.synthetic1 = pd.concat([self.split1, split2dummy],axis=1)
        self.synthetic2 = pd.concat([split1dummy, self.split2],axis=1)

        return (self.synthetic1, self.synthetic2), (self.holdout1, self.holdout2)


#%%
class AttributeDisclosure :
    '''
    The adversary has the information of the subject
    sensitive information : age, sex, diagnosis information

    original : original data without the patient_id index
    synthetic : synthetic data without the patient_id index 
    '''

    def __init__(self, original, synthetic, sensitive_features : list) :
        self.original = original
        self.synthetic = synthetic
        self.sensitive_features = sensitive_features

        # self.classifier = KNeighborsClassifier() # you can choose whatever you want!
        self.classifier = RandomForestClassifier(max_depth = 2, random_state=0)

        # score dictionary for features
        self.baseline_score = {}
        self.synthetic_score = {}

        # devide the data into train and tests
        self.preprocessing()

    def preprocessing(self) : 
        self.originalTrain, self.originalTest, _, _ = train_test_split(self.original, pd.Series([0]*len(self.original)), \
                test_size = 0.33, random_state = 0)

        self.syntheticTrain, self.syntheticTest, _, _ = train_test_split(self.synthetic, pd.Series([0]*len(self.synthetic)), \
                test_size = 0.33, random_state = 0)

        self.oriTrain = self.originalTrain.drop(columns = self.sensitive_features).copy()
        self.oriTest = self.originalTest.drop(columns = self.sensitive_features).copy()

        self.synTrain = self.syntheticTrain.drop(columns = self.sensitive_features).copy()
        self.synTest = self.syntheticTest.drop(columns = self.sensitive_features).copy()


    def train_test_synthetic(self, label) : 
        print(f"------------------------inferring for label : {label}-----------------------")
        print(f"------------------------inferring for synthetic data -----------------------")
        # first fit in train - synthetic
        x = self.synTrain
        y = self.syntheticTrain[label]
        testX = self.synTest 
        testY = self.syntheticTest[label]

        # fit on train 
        self.classifier.fit(x, y)

        # test on the test data
        pred = self.classifier.predict(testX)
        score = accuracy_score(pred, testY)
        self.synthetic_score[label] = score

    def baseline(self, label) :

        print(f"--------------------------infering for label : {label}----------------------")
        print(f"------------------------inferring for original data ------------------------")

        x = self.oriTrain
        y = self.originalTrain[label]
        testX = self.oriTest
        testY = self.originalTest[label]
        
        # fit the classifier to the train
        self.classifier.fit(x,y)

        # test the original
        pred = self.classifier.predict(testX)

        score = accuracy_score( pred, testY)
        self.baseline_score[label] = score

    def infer(self) :

        print("--"*10)

        for feature in self.sensitive_features : 
            self.baseline(feature)
            self.train_test_synthetic(feature)

        return self.baseline_score, self.synthetic_score

