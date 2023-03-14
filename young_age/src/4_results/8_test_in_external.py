from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
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
import pandas as pd
import pickle


#%% load pickle files

def load_pickle(file_name) :

    with open(file_name, 'rb') as f:
        return pickle.load(f)

trvr_model_list = load_pickle('trvr_models.pkl')

epsilon_per_strategy = {}
for epsilon in [0, 0.1, 1, 10, 100, 1000, 10000] :
    name = f'strategy_list_epsilon{epsilon}.pkl'
    strategylist = load_pickle(name)
    epsilon_per_strategy[epsilon] = strategylist


#%% load data

ncc_data = load_pickle("D0_ncc.pkl")

#%% process encoding

# encoder = load_pickle('labelEncoder.pkl')

#%% Test models in external setting

test_y = ncc_data['DEAD']
test_x = ncc_data.drop(columns = ['DEAD','OVR_SURV', 'DEAD_DIFF', 'PT_SBST_NO'])

#%%

def test_model(model_list, type, epsilon, test) :
    test_x, test_y = test
    model_names = ["DecisionTree", "RandomForest", "XGBoost"]

    scores = []
    for idx,(model, scaler) in enumerate(model_list):
        pred = model.predict(test_x)
        
        auroc = roc_auc_score(test_y,pred)
        f1score = f1_score(test_y, pred)
        accuracy = accuracy_score(test_y, pred)

        book = {'model' : model_names[idx],
                "type": type,
                "epsilon":  epsilon,
                "auroc" : auroc,
                "f1_score": f1score,
                "accuracy": accuracy}

        scores.append(book)
    return scores
#%%

rr_results = test_model(trvr_model_list, 'trtr', 0, (test_x, test_y))
all_test_result = rr_results.copy()
# rr_result -> list. Thus you need to append to this 

for idx, epsilon in enumerate([0,0.1,1,10,100,1000,10000]) :

    strategy_list = epsilon_per_strategy[epsilon]

    for j, train_type in enumerate(['tstr', 'trts', 'tsts']) :
        result = test_model(strategy_list[j], train_type, epsilon, (test_x, test_y))
        all_test_result.extend(result)

test_result = pd.DataFrame(all_test_result)
test_result.to_csv(output_path.joinpath('external_validation.csv'),index=False)
    


