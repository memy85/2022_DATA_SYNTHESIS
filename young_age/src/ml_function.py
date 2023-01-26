from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_auc_score
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, cross_val_score
from imblearn.over_sampling import SMOTE
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import KFold
from utils import *

def ml_train(data, model, epsilon, save = False, over_sampling = False , importance = False):
    
    '''
    이 함수는 데이터와 모델을 받아서 훈련시키는 함수
    '''

    data = data.astype(float)
    
    x = data.drop(['DEAD'], axis=1)
    y = data['DEAD']
    
    
    #x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.25,random_state=123)
    
    feature = x.columns
    
    #x_test = scale(x_test)
    #x_train = scale(x_train)
    
    
    
    dists = {
        'max_depth' : [3,5,10,15,20,30], # search space
        'max_features' : [3,5,10] # search features
    }
    
    from sklearn.model_selection import StratifiedKFold
    skf = StratifiedKFold(n_splits = 10, shuffle = True, random_state=123)
    #models= model
    
    #model = model.fit(x_train,y_train)
    
    #pred = model.predict(x_test)
    #f1 = cross_val_score(model, x, y, scoring='f1', cv=10)
    #acc = cross_val_score(model, x, y, scoring='accuracy', cv=skf)
    #roc = cross_val_score(model, x, y, scoring='roc_auc', cv=skf)
    
    #print("After OverSampling, counts of label '1': {}".format(sum(y_train==1)))
    #print("After OverSampling, counts of label '0': {}".format(sum(y_train==0)))
        
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
            
#    f1 = f1_score(y_test,pred, average ='macro')
#    acc = accuracy_score(y_test,pred)
#    roc = roc_auc_score(y_test,pred)
    
    
#    print("f1 score : ",f1)
#    print("accuracy : ", acc)
#    print("auc : ", roc)
    
    
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
    
    
def output(real, syn_data,model, bar=True):
    target = 'DEAD'

    name = str(model).split('C')[0]
    if '()' in name:
        name= name[:-2]
    syn_to_real = []
    syn_to_syn = []

    real_test = real.sample(800)
    real_valid = real.drop(real_test.index)
    
    real_valid_y = real_valid[target]
    real_valid_y = real_valid_y.astype(int)
    real_valid_x = real_valid.drop([target],axis=1)

    # encoding ordinal columns
    #real_x = label_encoding(real_x, [['BSPT_SEX_CD']])
    real_valid_x = scale(real_valid_x)

    dt_real = ml_train(real ,model, epsilon =1)
    dt_real_model = dt_real[0]
    pred_real_to_real = dt_real[1]

    for i, syn in enumerate(syn_data):
        #dt_syn = ml_train(syn, model, epsilon=0)
        dt_syn_model = get_best_model(syn, real_test)[0]
        #dt_syn_model = dt_syn[0]
        #syn_score = dt_syn[1]
        # evaluate real value from modeled in synthesized

        # real value
        
        # 115
        pred_syn_to_real = dt_syn_model.predict(real_valid_x).astype(int)
        
        tstr_score = [f1_score(real_valid_y,pred_syn_to_real, average = 'macro'),
                      accuracy_score(real_valid_y,pred_syn_to_real),
                      roc_auc_score(real_valid_y,pred_syn_to_real) ]
        
        #tstr_score = [cross_val_score(dt_syn_model,real_x,real_y,scoring='f1_macro',cv=5,n_jobs=-1).mean(),
        #                cross_val_score(dt_syn_model,real_x,real_y,scoring='accuracy',cv=5,n_jobs=-1).mean(),
        #                cross_val_score(dt_syn_model,real_x,real_y,scoring='roc_auc',cv=5,n_jobs=-1).mean()]
        
        print(i, tstr_score)
        #tstr_score = dt_syn[1]
        #f1, acc, rocs
        #syn_to_syn.append(syn_score)
        syn_to_real.append(tstr_score)

    plot_graph(pred_real_to_real, syn_to_real, name, bar)
    return [syn_to_real, pred_real_to_real]


def get_best_model(syn, real):
    syn = syn.astype(float)
    real = real.astype(float)
    #real = pd.concat([real[real['DEAD']==1],real[real['DEAD']==0].sample(133)])
    syn_x = syn.drop('DEAD',axis=1)
    real_x = real.drop('DEAD',axis=1)
    
    syn_y = syn['DEAD']
    real_y = real['DEAD']

    syn_x = scale(syn_x)
    real_x = scale(real_x)
    
    param = {
    'max_depth':range(1, 21),
    'max_leaf_nodes':range(5, 101, 5),
    'criterion':['entropy','gini']
    }
    n_iter = 80

    
    grid_parameters = {"max_depth": [2,4,5,7,9,10,50,100],
                       "min_samples_split": [2,3,4]
                       }
    
    scores = []
    models = []
    prev = 0
    cnt = 0
    for i in range(len(grid_parameters['max_depth'])):
        for k in range(len(grid_parameters['min_samples_split'])):
            model = RandomForestClassifier(max_depth = grid_parameters['max_depth'][i], min_samples_split=grid_parameters['min_samples_split'][k],n_jobs=-1)
            model.fit(syn_x,syn_y)
            pred = model.predict(real_x).astype(int)
            f1 = f1_score(real_y,pred, average='macro')
            acc = accuracy_score(real_y,pred)
            roc = roc_auc_score(real_y,pred)
            
            
            #tstr_score = [cross_val_score(model,real_x,real_y,scoring='f1_macro',cv=5,n_jobs=-1).mean(),
            #    cross_val_score(model,real_x,real_y,scoring='accuracy',cv=5,n_jobs=-1).mean(),
            #    cross_val_score(model,real_x,real_y,scoring='roc_auc',cv=5,n_jobs=-1).mean()]
            
            
            if prev < (cross_val_score(model,real_x,real_y,scoring='f1_micro',cv=3,n_jobs=-1).mean()):
                print(cross_val_score(model,real_x,real_y,scoring='f1_micro',cv=3,n_jobs=-1).mean())
                best_model = model
                cnt = 0
            else:
                prev = (cross_val_score(model,real_x,real_y,scoring='f1_micro',cv=3,n_jobs=-1).mean())
                
            cnt+=1
            scores.append([f1,acc,roc])
            models.append(model)
            
            if cnt >2:
                print('break')
                break
                
    '''    
    model = RandomizedSearchCV(models,
                    param_distributions=param,
                    n_iter=n_iter, 
                    cv=5, 
                    n_jobs=-1,
                    scoring='f1_macro')
                    '''
    best = []
    for i in range(len(np.array(scores).transpose())):
        best.append(max(np.array(scores).transpose()[i]))
    sum_best = []
    for i in range(len(scores)):
        sum_best.append(sum(scores[i]))
        
    best_model = models[sum_best.index(max(sum_best))]
    
    return [best_model, best]
