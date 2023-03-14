#%%
import os
from pathlib import Path

project_dir = Path(os.getcwd())
# cur_file = Path(__file__).absolute()
# project_dir = cur_file.parents[2]

os.sys.path.append(project_dir.as_posix())

data_dir = project_dir.joinpath('data')
org_dir = data_dir.joinpath('processed/preprocess_1')
# syn_dir = data_dir.joinpath('processed/2_produce_data/synthetic_decoded')
syn_dir = data_dir.joinpath('processed/no_bind/decoded')
output_path = project_dir.joinpath("data/processed/4_results")
figure_path = project_dir.joinpath("figures")

from src.MyModule.ml_function import *
from src.MyModule.utils import *
from pathlib import Path
from sklearn.model_selection import train_test_split 
from imblearn.under_sampling import RandomUnderSampler
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, average_precision_score
import argparse

config = load_config()

#%%


def argument_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--age", default = 50, type = int)
    args = parser.parse_args()
    return args

def process_synthetic(synthetic) :
    '''
    do under sampling and split x and y
    '''
    syn_x = synthetic.drop(['DEAD'],axis = 1)
    syn_y = synthetic["DEAD"]

    # rus = RandomUnderSampler(sampling_strategy = 0.5, random_state = 0)
    # syn_x, syn_y = rus.fit_resample(syn_x, syn_y)
    return (syn_x, syn_y)


def prepare_data(data) :
    data = data.copy()
    data.rename(columns = {"RLPS_DIFF" : "RLPS DIFF"}, inplace =True)
    data = data.drop(["DEAD_DIFF", "PT_SBST_NO", "OVR_SURV", "Unnamed: 0"], axis = 1)
    data = data.fillna(999)
    return data


def train_real_real(original):

    train_x, train_y = original
    
    real_train_x, real_valid_x, real_train_y, real_valid_y = train_test_split(train_x, train_y, test_size = 0.2, stratify=train_y, random_state=0)

    # get train real - valid real
    dt_model_real = get_best_model('DecisionTree', real_train_x , real_train_y, real_valid_x, real_valid_y)
    rf_model_real = get_best_model('RandomForest', real_train_x , real_train_y, real_valid_x, real_valid_y)
    xgb_model_real = get_best_model('XGBoost', real_train_x , real_train_y, real_valid_x, real_valid_y)
    trtr_models = [dt_model_real,rf_model_real,xgb_model_real]

    return trtr_models


def get_results(original, synthetic):
    train_x, train_y = original
    syn_x, syn_y = synthetic

    syn_train_x, syn_valid_x, syn_train_y, syn_valid_y = train_test_split(syn_x, syn_y, test_size=0.2, random_state=0, stratify=syn_y)

    real_train_x, real_valid_x, real_train_y, real_valid_y = train_test_split(train_x, train_y, test_size = 0.2, stratify=train_y, random_state=0)

    # get train syn - valid real
    dt_model = get_best_model('DecisionTree', syn_train_x , syn_train_y, real_valid_x, real_valid_y)
    rf_model = get_best_model('RandomForest', syn_train_x , syn_train_y, real_valid_x, real_valid_y)
    xgb_model = get_best_model('XGBoost', syn_train_x , syn_train_y, real_valid_x, real_valid_y)
    tstr_models = [dt_model,rf_model,xgb_model]

    # get train syn - valid syn
    dt_model_tsts = get_best_model('DecisionTree', syn_train_x , syn_train_y, syn_valid_x, syn_valid_y)
    rf_model_tsts = get_best_model('RandomForest', syn_train_x , syn_train_y, syn_valid_x, syn_valid_y)
    xgb_model_tsts = get_best_model('XGBoost', syn_train_x , syn_train_y, syn_valid_x, syn_valid_y)
    tsts_models = [dt_model_tsts,rf_model_tsts,xgb_model_tsts]

    # get train real - valid syn
    dt_model_trts = get_best_model('DecisionTree', real_train_x , real_train_y, syn_valid_x, syn_valid_y)
    rf_model_trts = get_best_model('RandomForest', real_train_x , real_train_y, syn_valid_x, syn_valid_y)
    xgb_model_trts = get_best_model('XGBoost', real_train_x , real_train_y, syn_valid_x, syn_valid_y)
    trts_models = [dt_model_trts,rf_model_trts,xgb_model_trts]

    model_arr = [tstr_models,trts_models,tsts_models]
    return model_arr


def test_model(model_list, type, epsilon, test) :
    test_x, test_y = test
    model_names = ["DecisionTree", "RandomForest", "XGBoost"]

    scores = []
    for idx,(model, scaler) in enumerate(model_list):
        pred = model.predict(test_x)
        
        auroc = roc_auc_score(test_y,pred)
        f1score = f1_score(test_y, pred)
        accuracy = accuracy_score(test_y, pred)
        auprc = average_precision_score(test_y, pred)

        book = {'model' : model_names[idx],
                "type": type,
                "epsilon":  epsilon,
                "auroc" : auroc,
                "f1_score": f1score,
                "accuracy": accuracy,
                "auprc" : auprc}

        scores.append(book)
    return scores

#%%
def main() :
    args = argument_parse()
    age = args.age

    train = pd.read_pickle(org_dir.joinpath(f'train_ori_{age}.pkl'))
    train = train.replace(np.NaN, 999)

    test = pd.read_pickle(org_dir.joinpath(f'test_{age}.pkl'))
    test = test.fillna(999)

    train=train.drop(['DEAD_DIFF','PT_SBST_NO','OVR_SURV'], axis=1)

    train_x = train.drop('DEAD',axis=1)
    train_y = train['DEAD']

    test_x = test.drop(['DEAD','DEAD_DIFF','PT_SBST_NO','OVR_SURV'], axis=1)
    test_y = test['DEAD']

    synthetic_data_list = [pd.read_csv(syn_dir.joinpath(f"Synthetic_data_epsilon{eps}_{age}.csv")) for eps in config['epsilon']]

    syn_list = list(map(prepare_data, synthetic_data_list))
    syn_list = list(map(process_synthetic, syn_list))

    rr_models = train_real_real((train_x, train_y))
    rr_results = test_model(rr_models, 'trtr', 0, (test_x, test_y))
    all_test_result = rr_results.copy()
    # rr_result -> list. Thus you need to append to this 
    
    for idx, epsilon in enumerate(config['epsilon']) :

        syn_x, syn_y = syn_list[idx]
        strategy_list = get_results((train_x, train_y), (syn_x, syn_y))

        for j, train_type in enumerate(['tstr', 'trts', 'tsts']) :
            result = test_model(strategy_list[j], train_type, epsilon, (test_x, test_y))
            all_test_result.extend(result)

    test_result = pd.DataFrame(all_test_result)
    # test_result.to_csv(output_path.joinpath(f'training_strategy_{args.age}.csv'),index=False)
    test_result.to_csv(output_path.joinpath(f'training_strategy_{args.age}_no_bind.csv'),index=False)
    
    print("finished calculating the training strategy!!")

#%%

    # import matplotlib.pyplot as plt
    # import scienceplots

    # plt.style.use(['science','grid','no-latex','vibrant'])
    # plt.figure(figsize=(14,8))

    # x1=[1,3,5,7]
    # data1 = np.array(score_by_case).transpose()[0]
    # plt.bar(x1, data1,width=0.4,label='decision tree', color='skyblue')
    # plt.axhline(np.array(score_by_case).transpose()[0][0], 0.05, 0.95, linestyle='--',color='skyblue', linewidth=1)
    # for i, v in enumerate(data1):
    #     plt.text(x1[i], v, str(round(v,2)), ha='center', fontsize=11)
# test_x = test_x.replace(np.NaN,999)



    # X2=[1+0.5,3+0.5,5+0.5,7+0.5]
    # data2 = np.array(score_by_case).transpose()[1]
    # plt.bar(X2, data2,width=0.4,label='Random Forest',color = 'cornflowerblue')
    # plt.axhline(np.array(score_by_case).transpose()[1][0], 0.15, 0.95, color = 'cornflowerblue' ,linestyle='--', linewidth=1)
    # for i, v in enumerate(data2):
    #     plt.text(X2[i], v, str(round(v,3)), ha='center', fontsize=11)


    # X3=[1+1,3+1,5+1,7+1]
    # data3 = np.array(score_by_case).transpose()[2]
    # plt.bar(X3, data3,width=0.4,label='XGBoost',color = 'royalblue')
    # plt.axhline(np.array(score_by_case).transpose()[2][0], 0.2, 0.95, color = 'royalblue',  linestyle='--', linewidth=1)
    # for i, v in enumerate(data3):
    #     plt.text(X3[i], v, str(round(v,2)), ha='center', fontsize=11)



    # plt.legend(loc='upper right',bbox_to_anchor=(1.17, 1))
    # ticklabel=['Train real Valid in real','Train syn Valid in real','Train syn Valid in syn','Train real Valid in syn']
    # plt.xticks(X2,ticklabel, fontsize=11)

    # plt.xlabel('Scenario',fontsize=14)
    # plt.ylabel('AUROC Score',fontsize=14)

    # plt.yticks(np.arange(0,1,0.1))

    # plt.title('Training Strategy results comparison($\epsilon$ = 10,000)', fontsize=14)
    # plt.savefig(figure_path.joinpath("train_strategy.png"), dpi = 1000)

    # plt.show()

if __name__ == "__main__" :
    main()

#%%
