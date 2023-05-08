
import pandas as pd
import os
from pathlib import Path
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score
import pickle

# |%%--%%| <UwXNc8rDNF|dmPnRVuiHg>

cur_file = Path(os.getcwd())
working_dir = cur_file.parent
parent_dir = working_dir.parent
data_dir = parent_dir.joinpath('data')

# |%%--%%| <dmPnRVuiHg|8LUfG1aSxt>

org_dir = data_dir.joinpath('processed/preprocess_1')
syn_dir = data_dir.joinpath('processed/decoded')

# |%%--%%| <8LUfG1aSxt|jJAOsZwHq4>

train = pd.read_pickle(org_dir.joinpath('train_ori_50.pkl'))
test = pd.read_pickle(org_dir.joinpath('test_50.pkl'))
syn = pd.read_csv(syn_dir.joinpath('Synthetic_data_epsilon10000_50.csv'),index_col = 0)
#syn = pd.read_csv('/home/dogu86/2022_DATA_SYNTHESIS/young_age/Synthetic_data_epsilon0_50.csv',index_col=0)
#syn = pd.read_csv('/home/dogu86/young_age_colon_cancer/final_data/synthetic_decoded/Synthetic_data_epsilon10000.csv')
syn.rename(columns = {'RLPS_DIFF' : 'RLPS DIFF'}, inplace = True)

# |%%--%%| <jJAOsZwHq4|E9ChdxYaR5>

test_x = test.drop(['DEAD','DEAD_DIFF','PT_SBST_NO'], axis=1)
test_x = test_x.replace(np.NaN,999)

# |%%--%%| <E9ChdxYaR5|th0TonVwBg>

def get_feature_importances(model,feature):
    import seaborn as sns
    import matplotlib.pyplot as plt

    importances = model.feature_importances_
    #feature = x.columns
    imp, feat = (list(x) for x in zip(*sorted(zip(importances, feature), reverse=True)))

    plt.figure(figsize=(13, 10))
    sns.barplot(imp[:30], feat[:30])
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    plt.title('Classification Feature Importances (Top 30)', fontsize=18)
    plt.show()

# |%%--%%| <th0TonVwBg|mk6xewgiS0>

def get_best_model(data,valid,model_name):

    if model_name in ['dt','rf']:
        grid_parameters = {"max_depth": [2,4,5,7,9,10,50],
                       "min_samples_split": [2,3,4]}
        return tree_like(model_name,data,valid,grid_parameters)
        
    elif model_name == 'xgb':
        grid_parameters = {"max_depth": [4,5,7,10,50], 
                           'learning_rate':[0.01, 0.1]}
        return get_xgb(model_name,data,valid,grid_parameters)
    
    elif model_name == 'mlp':
        grid_parameters = {"hidden_layer_size": [10,50,70,100], 
                           'learning_rate_int':[0.01, 0.1]}
        return get_mlp(model_name,data,valid,grid_parameters)        
        
    else:
        print('model name error')
        return 0

# |%%--%%| <mk6xewgiS0|daJ0QjRGLB>

def tree_like(model_name,data,valid,param):
    from sklearn.model_selection import StratifiedKFold
    kfold = StratifiedKFold(n_splits=5,shuffle=True, random_state=0) # KFold 객체 생성
    
    x, y, valid_x, valid_y = get_data(data,valid)

    scores = []
    models = []
    
    for i in range(len(param['max_depth'])):
        for k in range(len(param['min_samples_split'])):
            if model_name == 'dt':
                model = DecisionTreeClassifier(max_depth = param['max_depth'][i], 
                                               min_samples_split=param['min_samples_split'][k],
                                              random_state=0)
            elif model_name == 'rf':
                model = RandomForestClassifier(max_depth = param['max_depth'][i], 
                                               min_samples_split=param['min_samples_split'][k],n_jobs=-1,
                                              random_state=0)

            
            model.fit(x,y)
            
            pred = model.predict(valid_x)
            cur_score = f1_score(valid_y,pred, average='macro')
            #cur_score = cross_val_score(model,valid_x,valid_y,scoring='f1_macro',cv=kfold,n_jobs=-1).mean()
            
            scores.append(cur_score)
            models.append(model)
    
    best_idx = scores.index(max(scores))
    
    print(max(scores), models[best_idx])
    
    return models[best_idx]


# |%%--%%| <daJ0QjRGLB|unYL2w9OOP>

def get_xgb(model_name,data,valid,param):
    x, y, valid_x, valid_y = get_data(data,valid)
    
    scores = []
    models = []    
    evals = [(valid_x, valid_y)]
    
    
    for i in range(len(param['max_depth'])):
        for k in range(len(param['learning_rate'])):
            model = XGBClassifier(n_estimators=100, early_stoping_rounds=50,eval_set=evals,
                learning_rate=param['learning_rate'][k], 
                                        max_depth=param['max_depth'][i],objective='binary:logistic',n_jobs=-1)

            model.fit(x, y,verbose=False,early_stopping_rounds=50, 
                        eval_metric='logloss',eval_set=evals)

            cur_score = cross_val_score(model,valid_x,valid_y,scoring='f1_macro',cv=5,n_jobs=-1).mean()

            scores.append(cur_score)
            models.append(model)   
                
    best_idx = scores.index(max(scores))
    
    print(max(scores), models[best_idx])
    
    return models[best_idx]

# |%%--%%| <unYL2w9OOP|4bSbU4nxOu>

def get_mlp(model_name,data,valid,param):
    x, y, valid_x, valid_y = get_data(data,valid)
    
    scores = []
    models = []   
    
    for i in range(len(param['hidden_layer_size'])):
        for k in range(len(param['learning_rate_int'])):
            model = MLPClassifier(activation='tanh',solver='adam',learning_rate='constant',
                learning_rate_init=param['learning_rate_int'][k], 
                                        hidden_layer_sizes=param['hidden_layer_size'][i])

            model.fit(x, y)

            cur_score = cross_val_score(model,valid_x,valid_y,scoring='f1_macro',cv=3,n_jobs=-1).mean()

            scores.append(cur_score)
            models.append(model)   
                
    best_idx = scores.index(max(scores))
    
    print(max(scores), models[best_idx])
    
    return models[best_idx]

# |%%--%%| <4bSbU4nxOu|0GZiyMSl0y>

def get_data(data,test):
    data = data.drop(['PT_SBST_NO'],axis=1)
    data = data.astype(float)


    x = data.drop(['DEAD','DEAD_DIFF'], axis=1)
    x = x.replace(np.NaN,999)
    y = data['DEAD']
    
    
    test_x = test.drop(['DEAD','DEAD_DIFF','PT_SBST_NO'], axis=1)
    test_x = test_x.replace(np.NaN,999)
    test_y = test['DEAD']
    
    
    return [x,y,test_x,test_y]

# |%%--%%| <0GZiyMSl0y|gpIPUeUM06>

train_x = train.drop('DEAD',axis=1)
train_y = train['DEAD']

# |%%--%%| <gpIPUeUM06|nGwKfYEcex>

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(train_x, train_y, test_size = 0.2, stratify=train_y)

# |%%--%%| <nGwKfYEcex|VgFoTU9scP>

X_train, X_test, y_train, y_test = train_test_split(train_x, train_y, test_size = 0.2, stratify=train_y)
real_valid = pd.concat([X_test,y_test],axis=1)
real_train = pd.concat([X_train,y_train],axis=1)

# |%%--%%| <VgFoTU9scP|TeWzZBYigt>



'''real_dead = train[train['DEAD']==1]
real_alive = train.sample((len(real_dead)))
real_valid = pd.concat([real_dead,real_alive])
real_train = train.drop(real_valid.index)'''

# Get Train Syn - Valid Real
dt_model = get_best_model(syn,real_valid,'dt')
rf_model = get_best_model(syn,real_valid,'rf')
xgb_model = get_best_model(syn,real_valid,'xgb')
#mlp_model = get_best_model(syn,real_valid,'mlp')
tstr_models = [dt_model,rf_model,xgb_model]
#tstr_models = [dt_model,rf_model,mlp_model]

# Get Train Real - Valid Real
dt_model_real = get_best_model(real_train,real_valid,'dt')
rf_model_real = get_best_model(real_train,real_valid,'rf')
xgb_model_real = get_best_model(real_train,real_valid,'xgb')
#mlp_model_real = get_best_model(real_train,real_valid,'mlp')
trtr_models = [dt_model_real,rf_model_real,xgb_model_real]
#trtr_models = [dt_model_real,rf_model_real,mlp_model_real]
               

# |%%--%%| <TeWzZBYigt|IOtpRGaRfL>

dt_model_real = get_best_model(real_train,real_valid,'dt')
rf_model_real = get_best_model(real_train,real_valid,'rf')

# |%%--%%| <IOtpRGaRfL|FiZcztrEt5>

dt_model = get_best_model(syn,real_valid,'dt')
rf_model = get_best_model(syn,real_valid,'rf')

# |%%--%%| <FiZcztrEt5|M40iqJARMs>

print(len(real_train),len(syn))

# |%%--%%| <M40iqJARMs|w4c046qyye>

get_feature_importances(dt_model_real, feature=test_x.columns)

# |%%--%%| <w4c046qyye|sj71DkyHU5>

get_feature_importances(dt_model, feature=test_x.columns)

# |%%--%%| <sj71DkyHU5|lOzFOZf6Pl>

from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score


pred = dt_model_real.predict(test_x)
f1 = f1_score(test['DEAD'],pred,average='macro')
f1

# |%%--%%| <lOzFOZf6Pl|jR7sc3ETio>

pred = dt_model.predict(test_x)
f1 = f1_score(test['DEAD'],pred,average='macro')
f1

# |%%--%%| <jR7sc3ETio|4C1juInS3p>

X_train, X_test, y_train, y_test = train_test_split(syn.drop('DEAD',axis=1), syn['DEAD'], test_size = 0.2, stratify=syn['DEAD'])
syn_valid = pd.concat([X_test,y_test],axis=1)
syn_train = pd.concat([X_train,y_train],axis=1)

# Get Train Syn - Valid Syn
dt_model_tsts = get_best_model(syn_train,syn_valid,'dt')
rf_model_tsts = get_best_model(syn_train,syn_valid,'rf')
xgb_model_tsts = get_best_model(syn_train,syn_valid,'xgb')
#mlp_model_tsts = get_best_model(syn_train,syn_valid,'mlp')
tsts_models = [dt_model_tsts,rf_model_tsts,xgb_model_tsts]
#tsts_models = [dt_model_tsts,rf_model_tsts,mlp_model_tsts]

# Get Train Real - Valid Syn
dt_model_trts = get_best_model(real_train,syn_valid,'dt')
rf_model_trts = get_best_model(real_train,syn_valid,'rf')
xgb_model_trts = get_best_model(real_train,syn_valid,'xgb')
#mlp_model_trts = get_best_model(real_train,syn_valid,'mlp')
trts_models = [dt_model_trts,rf_model_trts,xgb_model_trts]
#trts_models = [dt_model_trts,rf_model_trts,mlp_model_trts]

# |%%--%%| <4C1juInS3p|djtiXm0MoB>

#모델 저장

save_dir = cur_file.joinpath('ml_models')

with open(save_dir.joinpath('trts_models.pkl'), 'wb') as f:
    pickle.dump(trts_models, f, protocol=pickle.HIGHEST_PROTOCOL)
with open(save_dir.joinpath('tstr_models.pkl'), 'wb') as f:
    pickle.dump(trts_models, f, protocol=pickle.HIGHEST_PROTOCOL)
with open(save_dir.joinpath('tsts_models.pkl'), 'wb') as f:
    pickle.dump(trts_models, f, protocol=pickle.HIGHEST_PROTOCOL)
with open(save_dir.joinpath('trtr_models.pkl'), 'wb') as f:
    pickle.dump(trtr_models, f, protocol=pickle.HIGHEST_PROTOCOL)
    


# |%%--%%| <djtiXm0MoB|8ota1kgsY0>

# load
save_dir = cur_file.joinpath('ml_models')
with open(save_dir.joinpath('trts_models.pkl'), 'rb') as f:
    trts_models = pickle.load(f)
with open(save_dir.joinpath('tstr_models.pkl'), 'rb') as f:
    tstr_models = pickle.load(f)
with open(save_dir.joinpath('tsts_models.pkl'), 'rb') as f:
    tsts_models = pickle.load(f)
with open(save_dir.joinpath('trtr_models.pkl'), 'rb') as f:
    trtr_models = pickle.load(f)



# |%%--%%| <8ota1kgsY0|acwwOlLEUk>

model_arr = [trtr_models,tstr_models,trts_models,tsts_models]
score_by_case = []
for models in model_arr:
    scores = []
    for model in models:
        f1 = cross_val_score(model,test_x,test['DEAD'],scoring='f1_macro',cv=5,n_jobs=-1)
        scores.append(f1.mean())
    score_by_case.append(scores)

# |%%--%%| <acwwOlLEUk|sAGVPspPJ9>

score_by_case = []
from sklearn.metrics import f1_score
for models in model_arr:
    scores = []
    for model in models:
        pred = model.predict(test_x)
        f1 = f1_score(test['DEAD'],pred,average='macro')
        scores.append(f1)
    score_by_case.append(scores)

# |%%--%%| <sAGVPspPJ9|qs24LdWzUb>

import matplotlib.pyplot as plt
import scienceplots

plt.style.use(['science','grid','no-latex','vibrant'])
plt.figure(figsize=(14,8))

X1=[1,3,5,7]
data1 = np.array(score_by_case).transpose()[0]
plt.bar(X1, data1,width=0.4,label='Decision Tree', color='skyblue')
plt.axhline(np.array(score_by_case).transpose()[0][0], 0.05, 0.95, linestyle='--',color='skyblue', linewidth=1)
for i, v in enumerate(data1):
    plt.text(X1[i], v, str(round(v,2)), ha='center', fontsize=11)


X2=[1+0.5,3+0.5,5+0.5,7+0.5]
data2 = np.array(score_by_case).transpose()[1]
plt.bar(X2, data2,width=0.4,label='Random Forest',color = 'cornflowerblue')
plt.axhline(np.array(score_by_case).transpose()[1][0], 0.15, 0.95, color = 'cornflowerblue' ,linestyle='--', linewidth=1)
for i, v in enumerate(data2):
    plt.text(X2[i], v, str(round(v,3)), ha='center', fontsize=11)


X3=[1+1,3+1,5+1,7+1]
data3 = np.array(score_by_case).transpose()[2]
plt.bar(X3, data3,width=0.4,label='XGB',color = 'royalblue')
plt.axhline(np.array(score_by_case).transpose()[2][0], 0.2, 0.95, color = 'royalblue',  linestyle='--', linewidth=1)
for i, v in enumerate(data3):
    plt.text(X3[i], v, str(round(v,2)), ha='center', fontsize=11)



plt.legend(loc='upper right',bbox_to_anchor=(1.17, 1))
ticklabel=['Train real Valid in real','Train syn Valid in real','Train syn Valid in syn','Train real Valid in syn']
plt.xticks(X2,ticklabel, fontsize=11)

plt.xlabel('Scenario',fontsize=14)
plt.ylabel('F1-macro Score',fontsize=14)

plt.yticks(np.arange(0,1,0.1))

plt.title('Training Strategy results comparison(Epsilon=10000)', fontsize=14)

plt.show()

# |%%--%%| <qs24LdWzUb|oWN73NXCaF>

#trtr
get_feature_importances(tsts_models[1],feature=test_x.columns)


# |%%--%%| <oWN73NXCaF|PcxWxpllki>

#tstr
get_feature_importances(model_arr[1][1],feature=test_x.columns)

# |%%--%%| <PcxWxpllki|oXRzgRiRNG>

syn['DEAD'].value_counts()

# |%%--%%| <oXRzgRiRNG|QqU6wbZFVA>

syn['RLPS'].value_counts()

# |%%--%%| <QqU6wbZFVA|hyBJvD1GmF>

train['RLPS'].value_counts()

# |%%--%%| <hyBJvD1GmF|Bd4TMnlKsB>

syn.columns

# |%%--%%| <Bd4TMnlKsB|PCHY4aREFw>

import os
print(os.environ['PATH'])

# |%%--%%| <PCHY4aREFw|tX013BTGaD>

import matplotlib.pyplot as plt

# Create sample data
data = [5, 10, 15, 20]

# Create the bar plot
fig, ax = plt.subplots()
ax.bar(range(len(data)), data)

# Add the numeric information to the plot
for i, v in enumerate(data):
    ax.text(i, v + 0.5, str(v), ha='center')

# Show the plot
plt.show()

# |%%--%%| <tX013BTGaD|NbY1ariRez>


