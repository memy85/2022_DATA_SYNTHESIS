#%%
import scienceplots
from src.MyModule.distribution_comparison import *
from src.MyModule.ml_function import *
from xgboost import XGBClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
import pandas as pd
import numpy as np
from src.MyModule.utils import *
import os
import sys
from pathlib import Path

# project_path = Path(__file__).absolute().parents[2]
project_path = Path().cwd()

os.sys.path.append(project_path.as_posix())
print(f" this is project path : {project_path} ")

#%% settings

age = 50

#%% path settings

config = load_config()

input_path = get_path("data/processed/2_produce_data/synthetic_decoded")

synthetic_path = input_path.joinpath(f"Synthetic_data_epsilon10000_{age}.csv")

synthetic_data_path_list = [input_path.joinpath(
    f"Synthetic_data_epsilon{eps}_{age}.csv") for eps in config['epsilon']]

train_ori_path = get_path(f"data/processed/preprocess_1/train_ori_{age}.pkl")

testset_path = get_path(f"data/processed/preprocess_1/test_{age}.pkl")

output_path = get_path("data/processed/3_evaluate_data/")

if not output_path.exists():
    output_path.mkdir(parents=True)


#%% import models
# synthetic = pd.read_csv(synthetic_path)
synthetic_data_list = list(
    map(lambda x: pd.read_csv(x), synthetic_data_path_list))
train_ori = pd.read_pickle(train_ori_path)
test = pd.read_pickle(testset_path)

#%%

test.drop(["PT_SBST_NO"], axis=1, inplace=True)
train_ori.drop(["PT_SBST_NO"], axis=1, inplace=True)
train_ori = train_ori.rename(columns={"RLPS DIFF": "RLPS_DIFF"})
synthetic_data_list = list(map(lambda x: x.drop(
    ["PT_SBST_NO", "Unnamed: 0"], axis=1), synthetic_data_list))
#%%

synthetic_data_dict = {"eps_{}".format(config['epsilon'][idx]) : data for idx, data in enumerate(synthetic_data_list) }

#%%

train_ori.shape
test.shape
synthetic_data_list[0].shape

#%%
# real = valid.copy()
real = real.rename(columns={'DEAD_NFRM_DEAD': 'DEAD'})
#real = real.drop('DEAD_DIFF',axis=1)
#real = real.drop('BSPT_DEAD_YMD',axis=1)
#real = real.drop('CENTER_LAST_VST_YMD',axis=1)
#real = real.drop('DEAD.1',axis=1)
real = real.drop('DEAD_DIFF', axis=1)
#real = real.drop('5YR_DEAD',axis=1)

real = real.replace(-1, 0)
real.replace(np.NaN, 0, inplace=True)

#%%
train_ori_columns = set(train_ori.columns)
synthetic_columsn = set(synthetic_data_list[0].columns)


train_ori_columns - synthetic_columsn

#%%
len(real.columns)
len(synthetic.columns)
len(testset.columns)


#%%
models = [DecisionTreeClassifier(),
          KNeighborsClassifier(),
          RandomForestClassifier(n_jobs=-1),
          XGBClassifier()]

tstr_scores = []
real_scores = []

for i, model in enumerate(models):
    print(f'Processing ...' + str(model).split('C')[0])
    result = output(real, syn_data, model)

    tstr = result[0]
    real_score = result[1]

    tstr_scores.append(tstr)
    real_scores.append(real_score)
    print(f'Done {i+1} / {len(models)}')

#%%
test = pd.concat([real[real['DEAD'] == 1].sample(30),
                 real[real['DEAD'] == 0].sample(270)])

#%%
train = real.drop(test.index)
times5 = pd.DataFrame()
for _ in range(5):
    times5 = pd.concat([times5, train.copy()])

#%%
model = RandomForestClassifier()
model.fit(train.drop('DEAD', axis=1), train['DEAD'])
pred = model.predict(test.drop('DEAD', axis=1))

f1 = f1_score(test['DEAD'], pred, average='macro')
acc = accuracy_score(test['DEAD'], pred)
roc = roc_auc_score(test['DEAD'], pred)

#%%
[f1, acc, roc]

#%%
data = valid.copy()

times5 = real.copy()
for _ in range(4):
    times5 = pd.concat([times5, real.copy()])

times5

#%%
real = valid.copy()
real = data.rename(columns={'DEAD_NFRM_DEAD': 'DEAD'})
real.drop(['Unnamed: 0', 'PT_SBST_NO'], axis=1, inplace=True)
real = real.drop('DEAD_DIFF', axis=1)
real.replace(np.NaN, 0, inplace=True)

#data = syn_data[5]

def get_train(data):

    new_d0_dt = ml_train(data, DecisionTreeClassifier(),
                         1, save=False, importance=False)
    new_d0_rf = ml_train(data, RandomForestClassifier(
        max_depth=10), 1, save=False, importance=True)
    new_d0_knn = ml_train(data, KNeighborsClassifier(),
                          1, save=False, importance=False)

    # whole data

    return pd.DataFrame([new_d0_dt[1], new_d0_rf[1], new_d0_knn[1]], columns=['F1', 'Acc', 'ROC', 'Loss'],
                        index=['DecisionTree', 'RandomForest', 'KNN'])

a = get_train(real)
a

#%%
data = times5
#data = syn_data[5]


def get_train(data):

    new_d0_dt = ml_train(data, real, DecisionTreeClassifier())
    new_d0_rf = ml_train(data, real, RandomForestClassifier())
    #new_d0_knn =  get_best_model(data, real, KNeighborsClassifier())

    # whole data

    return pd.DataFrame([new_d0_dt[1], new_d0_rf[1]], columns=['F1', 'Acc', 'ROC'],
                        index=['DecisionTree', 'RandomForest'])


a = get_train(times5)
a

# %%
##


plt.style.use(['science', 'no-latex'])
plt.rcParams.update({
    "font.family": "serif",   # specify font family here
    "font.serif": ["Times"],  # specify font here
    "font.size": 7})          # specify font size here

plt.figure()
real['DEAD'].hist()
plt.title('Relapse, epsilon 10000')

##
# %%
plt.figure(figsize=(20, 3))

plt.subplot(1, 6, 1)
syn_data[0]['DG_RCNF_RLPS'].hist()
plt.title('Relapse, epsilon 0.1')

plt.subplot(1, 6, 2)
syn_data[1]['DG_RCNF_RLPS'].hist()
plt.title('Relapse, epsilon 1')

plt.subplot(1, 6, 3)
syn_data[2]['DG_RCNF_RLPS'].hist()
plt.title('Relapse, epsilon 10')

plt.subplot(1, 6, 4)
syn_data[3]['DG_RCNF_RLPS'].hist()
plt.title('Relapse, epsilon 1000')

plt.subplot(1, 6, 5)
syn_data[5]['DG_RCNF_RLPS'].hist()
plt.title('Relapse, epsilon 10000')

plt.subplot(1, 6, 6)
real['DG_RCNF_RLPS'].hist()
plt.title('Relapse, real')

# %%
epsilons = [10000]
plt.figure(figsize=(25, 3))
col = 'DEAD'
for i in range(6):
    plt.subplot(1, 7, i+1)
    syn_data[i][col].hist()
    plt.title(str(col)+' epsilon ' + str(epsilons[i]))

plt.subplot(1, 7, 7)
real[col].hist()
plt.title(str(col) + ' Real')


# %%

new_d0_dt = ml_train(real, DecisionTreeClassifier(), 1,
                     save=False, over_sampling=False, importance=False)
new_d0_rf = ml_train(real, RandomForestClassifier(), 1,
                     save=False, over_sampling=False, importance=False)
new_d0_knn = ml_train(real, KNeighborsClassifier(), 1,
                      save=False,  over_sampling=False, importance=False)


#real = real.iloc[:,:35]
# whole data

pd.DataFrame([new_d0_dt[1], new_d0_rf[1], new_d0_knn[1]], columns=['F1', 'Acc', 'ROC', 'Loss'],
             index=['DecisionTree', 'RandomForest', 'KNN'])
##
# %%
real['DEAD.1']

# %%
pd.read_csv('/home/dogu86/colon_synthesis_2/synthetic/S0_1.csv')

# %%
##
syn_data[3]

# %%
age_cut_comparison(syn_data[5])

age_cut_comparison(real)
