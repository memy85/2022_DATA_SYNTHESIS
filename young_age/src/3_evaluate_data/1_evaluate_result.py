#%%
import os, sys
from pathlib import Path

# project_path = Path(__file__).absolute().parents[2]
project_path = Path().cwd() 
os.sys.path.append(project_path.as_posix())
print(f" this is project path : {project_path} ")
from src.MyModule.utils import *

#%% settings

age = 50

#%% path settings
config = load_config()

input_path = get_path("data/processed/2_produce_data/synthetic_decoded")

synthetic_path = input_path.joinpath(f"Synthetic_data_epsilon10000_{age}.csv")

real_path = get_path(f"data/processed/preprocess_1/encoded_D0_to_syn_{age}.csv")

testset_path = get_path(f"data/processed/preprocess_1/encoded_D0_to_valid_{age}.csv")

whole_path = get_path(f"data/processed/preprocess_1/encoded_D0_{age}.csv")

output_path = get_path("data/processed/3_evaluate_data/")

if not output_path.exists() : 
    output_path.mkdir(parents=True)


#%% import models
import numpy as np
import pickle
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

from xgboost import XGBClassifier
from src.MyModule.utils import *
from src.MyModule.ml_function import *
from src.MyModule.distribution_comparison import *

#%%

synthetic = pd.read_csv(synthetic_path)
real = pd.read_csv(real_path)
testset = pd.read_csv(testset_path)
whole = pd.read_csv(whole_path)

#%%
whole

#%%

testset.drop(["PT_SBST_NO"], axis=1, inplace=True)
real.drop(["PT_SBST_NO"], axis=1, inplace=True)
synthetic.drop(["PT_SBST_NO", "Unnamed: 0"], axis=1, inplace=True)


#%% make testset and real, synthetic the same format

label_encoder_path = get_path(f"data/processed/preprocess_1/LabelEncoder_{age}.pkl")
label_encoder = pd.read_pickle(label_encoder_path)

#%%
label_encoder

#%%
for encoder in label_encoder:
    real

#%%
train_idx = real.index

real["RLPS DIFF"] = (whole["RLPS_DIAG_YMD"] - whole["BSPT_FRST_DIAG_YMD"]).dt.days

#%%
valid["BSPT_IDGN_AGE"]  = data["BSPT_IDGN_AGE"]
valid["DEAD_DIFF"] = (data["BSPT_DEAD_YMD"] - data["BSPT_FRST_DIAG_YMD"]).dt.days
valid["OVR_SURV"] = (data["CENTER_LAST_VST_YMD"]- data["BSPT_FRST_DIAG_YMD"]).dt.days
# valid["OPRT_SURV"] = (data["CENTER_LAST_VST_YMD"]- data["BSPT_FRST_DIAG_YMD"]).dt.days

for i in range(1,9):
    start = pd.to_datetime(data[f'TRTM_CASB_STRT_YMD{i}'], format = "%Y%m%d")
    end = pd.to_datetime(data[f'TRTM_CASB_CSTR_YMD2_{i}'], format = "%Y%m%d")

    monthly_diff = (end-start).dt.days
    start_diff = (start-data["BSPT_FRST_DIAG_YMD"]).dt.days
    valid[f"REGN_TIME_DIFF_{i}"] = monthly_diff
    valid[f"REGN_START_DIFF_{i}"] = start_diff



#%%
# real = valid.copy()
real = real.rename(columns={'DEAD_NFRM_DEAD':'DEAD'})
#real = real.drop('DEAD_DIFF',axis=1)
#real = real.drop('BSPT_DEAD_YMD',axis=1)
#real = real.drop('CENTER_LAST_VST_YMD',axis=1)
#real = real.drop('DEAD.1',axis=1)
real = real.drop('DEAD_DIFF',axis=1)
#real = real.drop('5YR_DEAD',axis=1)

real = real.replace(-1,0)
real.replace(np.NaN,0, inplace=True)

#%%
# syn_data = []
# 

# for epsilon in [10000]:
#     temp = pd.read_csv(input_path(f'synthetic_decoded/Synthetic_data_epsilon{epsilon}_{age}.csv'))
#     temp = temp.rename(columns={'DEAD_NFRM_DEAD':'DEAD'})
#     temp = temp.drop('DEAD_DIFF', axis=1)
#     temp.drop(['Unnamed: 0','PT_SBST_NO'],axis=1,inplace=True)
#     temp.replace(np.NaN,0,inplace=True)
#     syn_data.append(temp.astype(float))


#%% check columns

set(real.columns) == set(synthetic.columns)

#%%
len(real.columns)
len(synthetic.columns)
len(testset.columns)

#%%
set(real.columns) == set(testset.columns)
    
#%%
set(synthetic.columns) == set(testset.columns)

#%%
models = [DecisionTreeClassifier(),
          KNeighborsClassifier(),
          RandomForestClassifier(n_jobs=-1),
          XGBClassifier()]

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
    
#%%
test = pd.concat([real[real['DEAD']==1].sample(30),real[real['DEAD']==0].sample(270)])

#%%
train = real.drop(test.index)
times5=pd.DataFrame()
for _ in range(5):
    times5 = pd.concat([times5,train.copy()])

#%%
model = RandomForestClassifier()
model.fit(train.drop('DEAD',axis=1),train['DEAD'])
pred = model.predict(test.drop('DEAD',axis=1))

f1 = f1_score(test['DEAD'],pred, average='macro')
acc = accuracy_score(test['DEAD'],pred)
roc = roc_auc_score(test['DEAD'],pred)

#%%
[f1,acc,roc]

#%%
data = valid.copy()

times5 = real.copy()
for _ in range(4):
    times5 = pd.concat([times5,real.copy()])

times5

#%%
real = valid.copy()
real= data.rename(columns={'DEAD_NFRM_DEAD':'DEAD'})
real.drop(['Unnamed: 0','PT_SBST_NO'],axis=1,inplace=True)
real = real.drop('DEAD_DIFF',axis=1)
real.replace(np.NaN,0,inplace=True)

#data = syn_data[5]

def get_train(data):

    new_d0_dt = ml_train(data, DecisionTreeClassifier(), 1, save = False, importance = False)
    new_d0_rf = ml_train(data, RandomForestClassifier(max_depth=10), 1, save = False, importance = True)
    new_d0_knn = ml_train(data, KNeighborsClassifier(), 1, save = False, importance = False)

    #whole data

    return pd.DataFrame([new_d0_dt[1],new_d0_rf[1],new_d0_knn[1]],columns = ['F1','Acc','ROC','Loss'], 
                index = ['DecisionTree','RandomForest','KNN'])
    
a = get_train(real)
a

#%%
##
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

#%%
##

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
#%%
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

#%%
epsilons = [10000]
plt.figure(figsize=(25,3))
col = 'DEAD'
for i in range(6):
    plt.subplot(1,7,i+1)
    syn_data[i][col].hist()
    plt.title(str(col)+' epsilon ' + str(epsilons[i]))
    
plt.subplot(1,7,7)
real[col].hist()
plt.title(str(col) + ' Real')


#%%

new_d0_dt = ml_train(real, DecisionTreeClassifier(), 1, save = False, over_sampling=False ,importance = False)
new_d0_rf = ml_train(real, RandomForestClassifier(), 1, save = False, over_sampling=False, importance = False)
new_d0_knn = ml_train(real, KNeighborsClassifier(), 1, save = False,  over_sampling=False, importance = False)


#real = real.iloc[:,:35]
#whole data

pd.DataFrame([new_d0_dt[1],new_d0_rf[1],new_d0_knn[1]],columns = ['F1','Acc','ROC','Loss'], 
             index = ['DecisionTree','RandomForest','KNN'])
##
#%%
real['DEAD.1']

#%%
pd.read_csv('/home/dogu86/colon_synthesis_2/synthetic/S0_1.csv')

#%%
##
syn_data[3]

#%%
age_cut_comparison(syn_data[5])

age_cut_comparison(real)
