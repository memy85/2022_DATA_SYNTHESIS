from src.MyModule.ml_function import *
from src.MyModule.utils import *
import os
from pathlib import Path
from imblearn.under_sampling import RandomUnderSampler

project_path = Path(__file__).absolute().parents[2]
data_dir = project_path.joinpath('data')


data_dir = parent_dir.joinpath('data')

org_dir = data_dir.joinpath('processed/preprocess_1')
syn_dir = data_dir.joinpath('processed/no_bind/decoded')

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
    epsilons = [0.1,1,10,100,1000,10000]
    fig1score = []

    for epsilon in epsilons:
        syn = pd.read_csv(syn_dir.joinpath(f'Synthetic_data_epsilon{epsilon}_50.csv'),index_col = 0)
        syn_x = syn.drop('DEAD',axis=1)
        syn_y = syn['DEAD']
        syn_train_x, syn_valid_x, syn_train_y, syn_valid_y = train_test_split(syn_x, syn_y, test_size=0.2, random_state=0, stratify=syn_y)

        dt_model = get_best_model('DecisionTree', syn_train_x , syn_train_y, real_valid_x, real_valid_y)
        rf_model = get_best_model('RandomForest', syn_train_x , syn_train_y, real_valid_x, real_valid_y)
        xgb_model = get_best_model('XGBoost', syn_train_x , syn_train_y, real_valid_x, real_valid_y)
        tstr_models = [dt_model,rf_model,xgb_model]

        for model in tstr_models:
            scores = []
            pred = model.predict(test_x)
            score = roc_auc_score(test['DEAD'],pred)
            scores.append(score)
        fig1score.append(scores)


if __name__ == "__main__" :
    main()
