from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, cross_val_score
from imblearn.over_sampling import SMOTE
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import KFold
from src.MyModule.utils import *

def ml_train(x, y,  model, epsilon, save = False, over_sampling = False , importance = False):
    
    '''
    이 함수는 데이터와 모델을 받아서 훈련시키는 함수
    '''

    x = x.astype(float)
    y = y.astype(float)
    
    feature = x.columns
    
    dists = {
        'max_depth' : [3,5,10,15,20,30], # search space
        'max_features' : [3,5,10] # search features
    }
    
    from sklearn.model_selection import StratifiedKFold
    skf = StratifiedKFold(n_splits = 10, shuffle = True, random_state=123)
        
    f1_arr = []
    acc_arr =[]
    roc_arr = []
    loss_arr = []
    
    skf = StratifiedKFold(n_splits=10)
    for train, test in skf.split(x, y):
        models = RandomizedSearchCV(model, param_distributions=dists, cv=3, refit=True)
        
        x_train = x.iloc[train]
        y_train = y.iloc[train]
        x_test = x.iloc[test]
        y_test = y.iloc[test]
        
        if over_sampling == True:
            smote = SMOTE(random_state=42)
            x_train, y_train = smote.fit_resample(x_train, y_train)
            x_train = pd.concat([x_train,x_train] )
            y_train = pd.concat([y_train,y_train])
        
        x_train = scale(x_train)
        x_test = scale(x_test)
        
        models.fit(x_train, y_train)
        pred = models.predict(x_test)
        
        f1 = f1_score(y_test,pred, average='macro')
        acc = accuracy_score(y_test,pred)
        roc = roc_auc_score(y_test,pred)
        
        proba = models.predict_proba(x_test)[:, 1]
        loss = cross_entropy(proba, y_test)
        
        f1_arr.append(f1)
        acc_arr.append(acc)
        roc_arr.append(roc)
        loss_arr.append(loss)
        
        
#    pred_proba = model.predict_proba(x_test)[:,-1]
    if (save == True):
        with open('clf_model'+str(epsilon)+'.pkl','wb') as fw:
            pickle.dump(models, fw)
            
    
    if importance == True:
        import seaborn as sns

        importances = model.feature_importances_
        
        imp, feat = (list(x) for x in zip(*sorted(zip(importances, feature), reverse=True)))
        
        plt.figure(figsize=(13, 10))
        sns.barplot(imp[:30], feat[:30])
        plt.xlabel('Importance')
        plt.ylabel('Feature')
        plt.title('Classification Feature Importances (Top 30)', fontsize=18)
        plt.show()
    
    return [models,[np.round(np.mean(f1_arr), 4),np.round(np.mean(acc_arr),4),np.round(np.mean(roc_arr), 4),
                  np.round(np.mean(loss_arr), 4)]]
    
def plot_graph(pred_real_to_real, syn_to_real, name, bar=True):
    nrow = 6 # 행의 갯수
    w = 0.25
    idx = np.arange(nrow)
    
    plt.figure(figsize=(17,8))
    plt.bar([-1.1],[pred_real_to_real[0]], width=w, color = 'yellowgreen', label = 'F1 Score, real')
    plt.bar([-1.1+w],[pred_real_to_real[2]], width=w, color = 'seagreen', label = 'AUC Score, real')
    plt.bar(idx -w*2+0.4, np.transpose(syn_to_real)[0], width = w, color = 'darkred', label = 'F1 Score, TSTR')
    plt.bar(idx -w+ 0.4, np.transpose(syn_to_real)[2], width = w, label = 'AUC Score, TSTR' ,color ='lightcoral')

    plt.plot(idx - w +0.4, np.transpose(syn_to_real)[2],color ='lightcoral')
    plt.plot(idx - w*2 +0.4, np.transpose(syn_to_real)[0],color = 'darkred')
    plt.xticks(np.arange(-2,6,1),['', 'real' ,'0.1','1','10','100','1000','10000'])

    plt.axhline([pred_real_to_real[2]], 0.15, 0.95,color = 'seagreen', linestyle = '--')
    plt.axhline([pred_real_to_real[0]], 0.1, 0.95,color = 'yellowgreen', linestyle = '--')
    x = np.arange(-2,6,1)

    plt.legend()
    plt.ylabel = 'Scores'
    plt.xlabel = 'Epsilon'
    plt.title(name + ' TSTR Results - Colon Cancer')   
    
def train_and_test(model_name, **kwargs):

    '''
    This function trains a model and outputs the true performance of the given machine learning model
    model name : DecisionTree, RandomForest, XGBoost
    kwargs : train_x, train_y, valid_x, valid_y, test_x, test_y and a model, also the model_path
    output : (auroc, f1_score)

    '''
    train_x, train_y = kwargs['train_x'], kwargs['train_y']
    valid_x, valid_y = kwargs['valid_x'], kwargs['valid_y']
    test_x, test_y = kwargs['test_x'], kwargs['test_y']
    model_path = kwargs['model_path']

    best_model, scaler = get_best_model(model_name, train_x, train_y, valid_x, valid_y)
    save_model(model_path, model_name, best_model)

    if best_model == 0 :
        assert False, "there the there is no model named {}".format(model_name)

    testset = (test_x, test_y)
    accuracy, auc, f1 = test_model(testset,best_model, scaler)
    return accuracy, auc, f1

def test_model(test_data, model, scaler):

    x, y = test_data
    # scaler = StandardScaler()
    x = scaler.fit_transform(x)

    pred = model.predict(x)
    accuracy = accuracy_score(y, pred)
    auc = roc_auc_score(y, pred)
    f1 = f1_score(y, pred)

    return accuracy, auc, f1
    
def get_best_model(model_name, train_x, train_y, valid_x, valid_y):
    
    train = (train_x, train_y)
    valid = (valid_x, valid_y)

    if model_name in ['DecisionTree','RandomForest']:
        grid_parameters = {"max_depth": [2,4,5,7,9,10,50],
                       "min_samples_split": [2,3,4]}
        return tree_like(model_name,train,valid,grid_parameters)
        
    elif model_name == 'XGBoost':
        grid_parameters = {"max_depth": [4,5,7,10,50], 
                           'learning_rate':[0.01, 0.1]}
        return get_xgb(model_name,train,valid,grid_parameters)
        
    else:
        print('model name error')
        return 0


#%%

def save_model(path, model_name, model):
    """
    saves the model to the path
    """
    if not path.exists() :
        path.mkdir(parents=True)

    with open(path.joinpath("{}.pkl".format(model_name)), 'wb') as f:
        pickle.dump(model, f)


#%%

def tree_like(model_name,train,valid,param):
    
    (x, y), (valid_x, valid_y) = train, valid
    cnt = 0
    prev = 0
    
    scaler = StandardScaler()
    x = scaler.fit_transform(x)
    valid_x = scaler.fit_transform(valid_x)
    
    scores = []
    models = []
    
    for i in range(len(param['max_depth'])):
        for k in range(len(param['min_samples_split'])):
            if model_name == 'DecisionTree':
                model = DecisionTreeClassifier(max_depth = param['max_depth'][i], 
                                               min_samples_split=param['min_samples_split'][k],random_state=0)
            elif model_name == 'RandomForest':
                model = RandomForestClassifier(max_depth = param['max_depth'][i], 
                                               min_samples_split=param['min_samples_split'][k],n_jobs=-1,
                                               random_state=0)

            model.fit(x,y)
            
            pred = model.predict(valid_x)
            # cur_score = cross_val_score(model,valid_x,valid_y,scoring='f1_macro',cv=10,n_jobs=-1).mean()
            cur_score = f1_score(valid_y, pred, average="macro")

            
            scores.append(cur_score)
            models.append(model)
    
    best_idx = scores.index(max(scores))
    
    print(max(scores), models[best_idx])
    
    return models[best_idx], scaler

def get_xgb(model_name,train,valid,param):
    (x, y), (valid_x, valid_y) = train, valid
    
    cnt = 0
    prev = 0
    scores = []
    models = []    
        
    scaler = StandardScaler()
    x = scaler.fit_transform(x)
    valid_x = scaler.fit_transform(valid_x)
    evals = [(valid_x, valid_y)]
    
    for i in range(len(param['max_depth'])):
        for k in range(len(param['learning_rate'])):
            model = XGBClassifier(n_estimators=100, early_stoping_rounds=50,eval_set=evals,
                learning_rate=param['learning_rate'][k], max_depth=param['max_depth'][i],objective='binary:logistic',n_jobs=-1, random_state=0)

            model.fit(x, y, verbose=True, early_stopping_rounds=100, 
                        eval_metric='logloss', eval_set=evals)

            pred = model.predict(valid_x)
            # cur_score = cross_val_score(model,valid_x,valid_y,scoring='f1_macro',cv=10,n_jobs=-1).mean()
            cur_score = f1_score(valid_y, pred, average="macro")

            scores.append(cur_score)
            models.append(model)   
                
    best_idx = scores.index(max(scores))
    
    print(max(scores), models[best_idx])
    
    return models[best_idx], scaler
