
##
#|%%--%%| <WmgDJQhMzp|U4NoANv5kb>
import os, sys
from pathlib import Path
# project_path = Path(__file__).absolute().parents[2]
project_path = Path().cwd() 
os.sys.path.append(project_path.as_posix())
print(f"this is project path : {project_path} ")
from src.MyModule.utils import *

## path settings
#|%%--%%| <U4NoANv5kb|RmUiaSN5qD>
config = load_config()
input_path = get_path("data/processed/2_produce_data/")
output_path = get_path("data/processed/3_evaluate_data/")


## import models 
#|%%--%%| <RmUiaSN5qD|uXoWjiBPSF>

import numpy as np
import pickle
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

from xgboost import XGBClassifier
from utils import *
from ml_function import *
from distribution_comparison import *


## 
#|%%--%%| <uXoWjiBPSF|TeXr5YNn0j>
original = pd.read_csv('/final_src/modified_D0.csv')
original.drop(['PT_SBST_NO','Unnamed: 0'],axis = 1, inplace=True)
real = original
real = real.rename(columns={'DEAD_NFRM_DEAD':'DEAD'})
#real = real.drop('DEAD_DIFF',axis=1)
#real = real.drop('BSPT_DEAD_YMD',axis=1)
#real = real.drop('CENTER_LAST_VST_YMD',axis=1)
#real = real.drop('DEAD.1',axis=1)
real = real.drop('DEAD_DIFF',axis=1)
#real = real.drop('5YR_DEAD',axis=1)


real = real.replace(-1,0)
real.replace(np.NaN,0, inplace=True)

##
#|%%--%%| <TeXr5YNn0j|hVa1nnJgZ1>
syn_data = []

for epsilon in [0.1,1,10,100,1000,10000]:
    temp = pd.read_csv(f'/home/dogu86/young_age_colon_cancer/final_data/synthetic_decoded/Synthetic_data_epsilon{epsilon}.csv')
    temp = temp.rename(columns={'DEAD_NFRM_DEAD':'DEAD'})
    temp = temp.drop('DEAD_DIFF', axis=1)
    temp.drop(['Unnamed: 0','PT_SBST_NO'],axis=1,inplace=True)
    temp.replace(np.NaN,0,inplace=True)
    syn_data.append(temp.astype(float))
    

##
#|%%--%%| <hVa1nnJgZ1|sf6PLY6ty4>

models = [DecisionTreeClassifier(),
          #KNeighborsClassifier(),
          #RandomForestClassifier(n_jobs=-1)
          #XGBClassifier()]
            ]
tstr_scores = []
real_scores = []
for i, model in enumerate(models):
    print(f'Processing ...'+ str(model).split('C')[0])
    result = output(real, syn_data, model)
    
    tstr = result[0]
    real_score = result[1]
    
    tstr_scores.append(tstr)
    real_scores.append(real_score)
    print(f'Done {i+1} / {len(models)}')
    



##
#|%%--%%| <sf6PLY6ty4|zi4OAcYFli>
test = pd.concat([real[real['DEAD']==1].sample(30),real[real['DEAD']==0].sample(270)])

##
#|%%--%%| <zi4OAcYFli|tTgXclzxz4>

train = real.drop(test.index)
times5=pd.DataFrame()
for _ in range(5):
    times5 = pd.concat([times5,train.copy()])

##
#|%%--%%| <tTgXclzxz4|NluDZ1Lbng>
model = RandomForestClassifier()
model.fit(train.drop('DEAD',axis=1),train['DEAD'])
pred = model.predict(test.drop('DEAD',axis=1))

f1 = f1_score(test['DEAD'],pred, average='macro')
acc = accuracy_score(test['DEAD'],pred)
roc = roc_auc_score(test['DEAD'],pred)

##
#|%%--%%| <NluDZ1Lbng|oxMpkalk8g>
[f1,acc,roc]

##
#|%%--%%| <oxMpkalk8g|gsm87xTFwG>
data = pd.read_csv('/home/dogu86/young_age_colon_cancer/final_src/modified_syn_0.csv')
data

##
#|%%--%%| <gsm87xTFwG|oMdZcGb49M>
times5 = real
for _ in range(4):
    times5 = pd.concat([times5,real.copy()])

##
#|%%--%%| <oMdZcGb49M|I6J4BevacV>
times5

##
#|%%--%%| <I6J4BevacV|xO3d7Iz2s7>
data = pd.read_csv('/home/dogu86/young_age_colon_cancer/final_data/synthetic_decoded/Synthetic_data_epsilon8000.csv', encoding = 'cp949')
data = data.rename(columns={'DEAD_NFRM_DEAD':'DEAD'})
data.drop(['Unnamed: 0','PT_SBST_NO'],axis=1,inplace=True)
data = data.drop('DEAD_DIFF',axis=1)
data.replace(np.NaN,0,inplace=True)

#data = syn_data[5]

def get_train(data):

    new_d0_dt = ml_train(data, DecisionTreeClassifier(), 1, save = False, importance = False)
    new_d0_rf = ml_train(data, RandomForestClassifier(max_depth=10), 1, save = False, importance = True)
    new_d0_knn = ml_train(data, KNeighborsClassifier(), 1, save = False, importance = False)

    #whole data

    return pd.DataFrame([new_d0_dt[1],new_d0_rf[1],new_d0_knn[1]],columns = ['F1','Acc','ROC','Loss'], 
                index = ['DecisionTree','RandomForest','KNN'])
    
a = get_train(data)
a

##
#|%%--%%| <xO3d7Iz2s7|5kHPpKJWBp>
data = times5
#data = syn_data[5]

def get_train(data):

    new_d0_dt = ml_train(data, real, DecisionTreeClassifier())
    new_d0_rf =  ml_train(data, real, RandomForestClassifier())
    #new_d0_knn =  get_best_model(data, real, KNeighborsClassifier())

    #whole data

    return pd.DataFrame([new_d0_dt[1],new_d0_rf[1]],columns = ['F1','Acc','ROC'], 
                index = ['DecisionTree','RandomForest'])
    
a = get_train(times5)
a

##
#|%%--%%| <5kHPpKJWBp|dEDG0W97p4>

import scienceplots


plt.style.use(['science','no-latex'])
plt.rcParams.update({
    "font.family": "serif",   # specify font family here
    "font.serif": ["Times"],  # specify font here
    "font.size":7})          # specify font size here

plt.figure()
real['DEAD'].hist()
plt.title('Relapse, epsilon 10000')

##
#|%%--%%| <dEDG0W97p4|bTkNjug3yt>
plt.figure(figsize=(20,3))

plt.subplot(1,6,1)
syn_data[0]['DG_RCNF_RLPS'].hist()
plt.title('Relapse, epsilon 0.1')

plt.subplot(1,6,2)
syn_data[1]['DG_RCNF_RLPS'].hist()
plt.title('Relapse, epsilon 1')

plt.subplot(1,6,3)
syn_data[2]['DG_RCNF_RLPS'].hist()
plt.title('Relapse, epsilon 10')

plt.subplot(1,6,4)
syn_data[3]['DG_RCNF_RLPS'].hist()
plt.title('Relapse, epsilon 1000')

plt.subplot(1,6,5)
syn_data[5]['DG_RCNF_RLPS'].hist()
plt.title('Relapse, epsilon 10000')

plt.subplot(1,6,6)
real['DG_RCNF_RLPS'].hist()
plt.title('Relapse, real')

##
#|%%--%%| <bTkNjug3yt|VZvys49CZa>
epsilons = [0.1,1,10,100,1000,10000]
plt.figure(figsize=(25,3))
col = 'DEAD'
for i in range(6):
    plt.subplot(1,7,i+1)
    syn_data[i][col].hist()
    plt.title(str(col)+' epsilon ' + str(epsilons[i]))
    
plt.subplot(1,7,7)
real[col].hist()
plt.title(str(col) + ' Real')


##
#|%%--%%| <VZvys49CZa|DxE8QMMD3L>


new_d0_dt = ml_train(real, DecisionTreeClassifier(), 1, save = False, over_sampling=False ,importance = False)
new_d0_rf = ml_train(real, RandomForestClassifier(), 1, save = False, over_sampling=False, importance = False)
new_d0_knn = ml_train(real, KNeighborsClassifier(), 1, save = False,  over_sampling=False, importance = False)


#real = real.iloc[:,:35]
#whole data

pd.DataFrame([new_d0_dt[1],new_d0_rf[1],new_d0_knn[1]],columns = ['F1','Acc','ROC','Loss'], 
             index = ['DecisionTree','RandomForest','KNN'])
##
#|%%--%%| <DxE8QMMD3L|tg1QzcuEIN>
real['DEAD.1']

##
#|%%--%%| <tg1QzcuEIN|1BxwbqWVE0>
pd.read_csv('/home/dogu86/colon_synthesis_2/synthetic/S0_1.csv')

##
#|%%--%%| <1BxwbqWVE0|UrazfeGcux>
syn_data[3]

##
#|%%--%%| <UrazfeGcux|rKgqIVkQw7>
age_cut_comparison(syn_data[5])

##
#|%%--%%| <rKgqIVkQw7|KwEWlhVsSp>
age_cut_comparison(real)

##
#|%%--%%| <KwEWlhVsSp|8A9IRDqrw3>
real.info()



#|%%--%%| <8A9IRDqrw3|1lQQdLaqsr>


