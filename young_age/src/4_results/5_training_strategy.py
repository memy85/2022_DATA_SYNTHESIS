from src.MyModule.ml_function import *
from src.MyModule.utils import *
import os
from pathlib import Path
from imblearn.under_sampling import RandomUnderSampler




cur_file = Path(os.getcwd())
working_dir = cur_file.parent
parent_dir = working_dir.parent
data_dir = parent_dir.joinpath('data')

org_dir = data_dir.joinpath('processed/preprocess_1')
syn_dir = data_dir.joinpath('processed/decoded')

print("working directory : " + working_dir)

train = pd.read_pickle(org_dir.joinpath('train_ori_50.pkl'))
test = pd.read_pickle(org_dir.joinpath('test_50.pkl'))
syn = pd.read_csv(syn_dir.joinpath('Synthetic_data_epsilon10000_50.csv'),index_col = 0)
syn.rename(columns = {'RLPS_DIFF' : 'RLPS DIFF'}, inplace = True)
syn = syn.drop('OVR_SURV',axis=1)
train=train.drop('OVR_SURV',axis=1)

train_x = train.drop('DEAD',axis=1)
train_y = train['DEAD']

test_x = test.drop(['DEAD','DEAD_DIFF','PT_SBST_NO','OVR_SURV'], axis=1)
test_x = test_x.replace(np.NaN,999)

syn_x = syn.drop('DEAD',axis=1)
syn_y = syn['DEAD']
rus = RandomUnderSampler(sampling_strategy = 0.5, random_state = 0)
syn_x, syn_y = rus.fit_resample(syn_x, syn_y)
syn_train_x, syn_valid_x, syn_train_y, syn_valid_y = train_test_split(syn_x, syn_y, test_size=0.2, random_state=0, stratify=syn_y)
real_train_x, real_valid_x, real_train_y, real_valid_y = train_test_split(train_x, train_y, test_size = 0.2, stratify=train_y,
                                                   random_state=0)

def main():

    # Get Train Syn - Valid Real
    dt_model = get_best_model('DecisionTree', syn_train_x , syn_train_y, real_valid_x, real_valid_y)
    rf_model = get_best_model('RandomForest', syn_train_x , syn_train_y, real_valid_x, real_valid_y)
    xgb_model = get_best_model('XGBoost', syn_train_x , syn_train_y, real_valid_x, real_valid_y)
    tstr_models = [dt_model,rf_model,xgb_model]

    # Get Train Real - Valid Real
    dt_model_real = get_best_model('DecisionTree', real_train_x , real_train_y, real_valid_x, real_valid_y)
    rf_model_real = get_best_model('RandomForest', syn_train_x , syn_train_y, train_x, train_y)
    xgb_model_real = get_best_model('XGBoost', syn_train_x , syn_train_y, train_x, train_y)
    trtr_models = [dt_model_real,rf_model_real,xgb_model_real]

    # Get Train Syn - Valid Syn
    dt_model_tsts = get_best_model('DecisionTree', syn_train_x , syn_train_y, syn_valid_x, syn_valid_y)
    rf_model_tsts = get_best_model('RandomForest', syn_train_x , syn_train_y, syn_valid_x, syn_valid_y)
    xgb_model_tsts = get_best_model('XGBoost', syn_train_x , syn_train_y, syn_valid_x, syn_valid_y)
    tsts_models = [dt_model_tsts,rf_model_tsts,xgb_model_tsts]

    # Get Train Real - Valid Syn
    dt_model_trts = get_best_model('DecisionTree', real_train_x , real_train_y, syn_valid_x, syn_valid_y)
    rf_model_trts = get_best_model('RandomForest', real_train_x , real_train_y, syn_valid_x, syn_valid_y)
    xgb_model_trts = get_best_model('XGBoost', real_train_x , real_train_y, syn_valid_x, syn_valid_y)
    trts_models = [dt_model_trts,rf_model_trts,xgb_model_trts]



    score_by_case = []
    model_arr = [trtr_models,tstr_models,trts_models,tsts_models]
    for models in model_arr:
        scores = []
        for model in models:
            pred = model.predict(test_x)
            score = roc_auc_score(test['DEAD'],pred)
            scores.append(score)
        score_by_case.append(scores)

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
    plt.bar(X3, data3,width=0.4,label='KNN',color = 'royalblue')
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

if __name__ == "__main__" :
    main()