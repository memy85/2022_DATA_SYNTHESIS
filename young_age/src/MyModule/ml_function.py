
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
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
from sklearn.metrics import average_precision_score as auprc_score 

import shap
from shap.explainers import Tree
from src.MyModule.utils import *

    
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
    use_scaler = kwargs['scaler']

    if isinstance(train_x, pd.DataFrame) | isinstance(valid_x, pd.DataFrame):
        try :
            train_x = train_x.convert_objects(convert_numeric=True)
            valid_x = valid_x.convert_objects(convert_numeric=True)
        except :
            pass

    best_model, scaler = get_best_model(model_name, train_x, train_y, valid_x, valid_y)

    if best_model == 0 :
        assert False, "there the there is no model named {}".format(model_name)

    # first fit best model to all the training data
    updated_x = pd.concat([train_x, valid_x], axis = 0)
    updated_x = updated_x.astype('float64')
    updated_y = pd.concat([train_y, valid_y], axis = 0)
    updated_y = updated_y.astype('float64')

    updated_best_model = best_model.fit(updated_x, updated_y)

    # calculate shap
    
    # explainer = shap.Explainer(updated_best_model, algorithm = 'tree')
    if model_name == "DeepLearning" :
        explainer = shap.KernelExplainer(updated_best_model.predict_proba, updated_x)
        # shap_values = explainer.shap_values(X_test)
        shap_values = 0

    else : 
        explainer = Tree(updated_best_model, data = updated_x)
        shap_values = explainer(test_x)
    # shap_values = explainer(train_x)


    save_model(model_path, model_name, updated_best_model)

    testset = (test_x, test_y)
    accuracy, auc, f1, auprc = test_model(testset, updated_best_model, scaler)

    return accuracy, auc, f1, auprc, shap_values

#%%



def test_model(test_data, model, scaler = None):
    
    x, y = test_data
    x = x.astype('float64')
    if scaler is not None : 
        scaler = StandardScaler()
        x = scaler.fit_transform(x)

    pred = model.predict(x)
    accuracy = accuracy_score(y, pred)
    auc = roc_auc_score(y, pred)
    f1 = f1_score(y, pred)
    auprc = auprc_score(y, pred)

    return accuracy, auc, f1, auprc


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

    elif model_name == 'LogisticRegression' :
        grid_parameters = {"C" : [0.001, 0.01, 0.1, 10, 100]}
                           
        return get_logistic(model_name, train, valid, grid_parameters)

    elif model_name == 'KNN' :
        grid_parameters = {
            "n_neighbors" : list(range(2,10)),
        }
        return get_knn(model_name, train, valid, grid_parameters)

    elif model_name == "DeepLearning" :
        grid_parameters = {
                "hidden_layers" : [(100,), (100, 100,), (100, 100, 100,)],
                "alpha" : [0.0001, 0.001, 0.01, 0.1, 1]
        }
        return get_deeplearning(model_name, train, valid, grid_parameters)

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

def get_deeplearning(model_name,train,valid,param, scale = None):
    (x, y), (valid_x, valid_y) = train, valid

    x = x.astype('float64')
    valid_x = valid_x.astype('float64')
    
    cnt = 0
    prev = 0
    scores = []
    models = []    
    
    if scale :
        scaler = StandardScaler()
        x = scaler.fit_transform(x)
        valid_x = scaler.fit_transform(valid_x)

    else :
        scaler = None
        
    evals = [(valid_x, valid_y)]
    
    for i in param['hidden_layers']:
        for k in param['alpha']:
            model = MLPClassifier(hidden_layer_sizes=i, alpha=k, max_iter=300)

            model.fit(x, y)

            pred = model.predict(valid_x)
            # cur_score = cross_val_score(model,valid_x,valid_y,scoring='f1_macro',cv=10,n_jobs=-1).mean()
            # cur_score = f1_score(valid_y, pred, average="macro")
            cur_score = roc_auc_score(valid_y, pred, average="macro")

            scores.append(cur_score)
            models.append(model)   
                
    best_idx = scores.index(max(scores))
    
    print(max(scores), models[best_idx])
    
    return models[best_idx], scaler

def get_knn(model_name, train, valid, param, scale=True) : 

    (x, y), (valid_x, valid_y) = train, valid
   
    if scale : 
        scaler = StandardScaler()
        x = scaler.fit_transform(x)
        valid_x = scaler.fit_transform(valid_x)
    else :
        scaler = None
    
    scores = []
    models = []
    
    for n in param['n_neighbors']:
        model = KNeighborsClassifier(n_neighbors = n)
        model.fit(x,y)
            
        pred = model.predict(valid_x)
         # cur_score = f1_score(valid_y, pred, average="macro")
        cur_score = roc_auc_score(valid_y, pred)
        
        scores.append(cur_score)
        models.append(model)
    
    best_idx = scores.index(max(scores))
    
    print(max(scores), models[best_idx])
    
    return models[best_idx], scaler


def get_logistic(model_name,
                 train,
                 valid,
                 param,
                 scale = False) :

    (x, y), (valid_x, valid_y) = train, valid
   
    if scale : 
        scaler = StandardScaler()
        x = scaler.fit_transform(x)
        valid_x = scaler.fit_transform(valid_x)
    else :
        scaler = None
    
    scores = []
    models = []
    
    for C in param["C"] :
        model = LogisticRegression(penalty = "l2",C = C, random_state = 0)
        model.fit(x,y)
        
        pred = model.predict(valid_x)
        # cur_score = f1_score(valid_y, pred, average="macro")
        cur_score = roc_auc_score(valid_y, pred, average="macro")
        
        scores.append(cur_score)
        models.append(model)
    
    best_idx = scores.index(max(scores))
    
    print(max(scores), models[best_idx])
    
    return models[best_idx], scaler

#%%

def tree_like(model_name,
              train,
              valid,
              param, 
              scale = None):
    
    (x, y), (valid_x, valid_y) = train, valid
    cnt = 0
    prev = 0
   
    if scale : 
        scaler = StandardScaler()
        x = scaler.fit_transform(x)
        valid_x = scaler.fit_transform(valid_x)
    else :
        scaler = None
    
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
            # cur_score = f1_score(valid_y, pred, average="macro")
            cur_score = roc_auc_score(valid_y, pred, average="macro")
            
            scores.append(cur_score)
            models.append(model)
    
    best_idx = scores.index(max(scores))
    
    print(max(scores), models[best_idx])
    
    return models[best_idx], scaler

def get_xgb(model_name,train,valid,param, scale = None):
    (x, y), (valid_x, valid_y) = train, valid

    x = x.astype('float64')
    valid_x = valid_x.astype('float64')
    
    cnt = 0
    prev = 0
    scores = []
    models = []    
    
    if scale :
        scaler = StandardScaler()
        x = scaler.fit_transform(x)
        valid_x = scaler.fit_transform(valid_x)

    else :
        scaler = None
        
    evals = [(valid_x, valid_y)]
    
    for i in range(len(param['max_depth'])):
        for k in range(len(param['learning_rate'])):
            model = XGBClassifier(n_estimators=100, early_stoping_rounds=50,eval_set=evals,
                learning_rate=param['learning_rate'][k], max_depth=param['max_depth'][i],objective='binary:logistic',n_jobs=-1, random_state=0)

            model.fit(x, y, verbose=True, early_stopping_rounds=100, 
                        eval_metric='logloss', eval_set=evals)

            pred = model.predict(valid_x)
            # cur_score = cross_val_score(model,valid_x,valid_y,scoring='f1_macro',cv=10,n_jobs=-1).mean()
            # cur_score = f1_score(valid_y, pred, average="macro")
            cur_score = roc_auc_score(valid_y, pred, average="macro")

            scores.append(cur_score)
            models.append(model)   
                
    best_idx = scores.index(max(scores))
    
    print(max(scores), models[best_idx])
    
    return models[best_idx], scaler
