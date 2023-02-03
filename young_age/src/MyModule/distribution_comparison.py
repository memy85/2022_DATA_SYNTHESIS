import matplotlib.pyplot as plt
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

from src.MyModule.ml_function import *



def count_value_plot(org, syn, col):
    
    plt.bar(org[col].value_counts().index.astype(str),
            org[col].value_counts())
    plt.title(f'Original {col} Distribution')
    plt.ylabel('Count')
    plt.show()
    
    plt.bar(syn[col].value_counts().index.astype(str),
            syn[col].value_counts())
    plt.title(f'Synthesized {col} Distribution')
    plt.ylabel('Count')
    plt.show()    
    
    
    
def hellinger_distance(org, syn, col):
        n = len(org)
        sum = 0.0
        for i in range(n):
                sum += (np.sqrt(org[i]) - np.sqrt(syn[i]))**2
                result = (1.0 / np.sqrt(2.0)) * np.sqrt(sum)
        return result


def age_cut_comparison(data):
        scores = []
        ages =[]
        pt_len = []
        cut = 5
        for i in range(5):
                age_cut = data[data['BSPT_IDGN_AGE']<(10-i)]
                print((10-i)*5)
                ages.append((10-i)*5)
                pt_len.append(len(age_cut))
                #new_d0_real = ml_train(age_cut, XGBClassifier(), 1, save = False, importance = False)
                new_d0_dt = ml_train(age_cut, DecisionTreeClassifier(), 1, save = False, importance = False)
                new_d0_rf = ml_train(age_cut, RandomForestClassifier(), 1, save = False, importance = False)
                #new_d0_knn = ml_train(age_cut, KNeighborsClassifier(), 1, save = False, importance = False)
                
                print(new_d0_dt[1],new_d0_rf[1])
                scores.append([new_d0_dt[1],new_d0_rf[1]])
        
        print('Done '+str(i+1)+'/5!')
        
        ages = np.array(ages).astype(int).astype(str)
        
        plt.figure(figsize=(10,5))

        plt.plot(ages,np.array(scores).transpose()[0][0], label = 'decision tree f1')
        plt.plot(ages,np.array(scores).transpose()[0][1], label = 'random forest f1')
        #plt.plot(ages,np.array(scores).transpose()[0][2], label = 'knn f1')
        
        plt.plot(ages,np.array(scores).transpose()[2][0], label = 'decision tree auc')
        plt.plot(ages,np.array(scores).transpose()[2][1], label = 'random forest auc')
        #plt.plot(ages,np.array(scores).transpose()[2][2], label = 'knn auc')


        plt.xlabel('Age')
        plt.ylabel('Score')
        plt.title('Age reduction subset data Classification Metrics')
        plt.legend()