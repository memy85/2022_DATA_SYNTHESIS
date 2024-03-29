#%%
import pandas as pd
from DataSynthesizer.DataGenerator import DataGenerator
import os, sys
import argparse
from itertools import repeat
from pathlib import Path
from pathos.multiprocessing import _ProcessPool
import random

PROJ_PATH = Path(__file__).parents[2]
sys.path.append(PROJ_PATH.joinpath('src').as_posix())

from MyModule.utils import *
config = load_config()

PROJ_PATH = Path(config['path_config']['project_path'])
INPUT_PATH = PROJ_PATH.joinpath('data/processed/1_apply_bayesian/apply_bayesian/out/')
OUTPUT_PATH = PROJ_PATH.joinpath('data/processed/1_apply_bayesian/produce_data')

if not OUTPUT_PATH.exists() :
    OUTPUT_PATH.mkdir(parents=True)


#%%
epsilons = config.get('epsilon')
mean_observation_days = config['bayesian_config'].get('mean_observation_days')
sd_observation_days = config['bayesian_config'].get('sd_observation_days')

def generate_data(epsilon, description_idx, sample_number):
    '''
    epsilon : 0.1 ~ 10000
    description_idx : patient id (1,2,..)
    sample_number : pseudo sample ID
    '''
    output_path = OUTPUT_PATH.joinpath(f'epsilon{epsilon}')
    if not output_path.exist() :
        output_path.mkdir()
        
    num_tuples = -1
    while num_tuples <= 0 :
        num_tuples = round(np.random.normal(mean_observation_days, sd_observation_days))
    
    # the outcome : Recur (DG_RCNF), DEATH (DEAD) should be the same
    original_data_path = PROJ_PATH.joinpath(f'data/processed/1_apply_bayesian/preprocess_data/pt_{description_idx}.csv')
    original_data = pd.read_csv(original_data_path)
    original_data = original_data.reset_index(drop=True)
    time_idx = original_data['TIME']
    max_time = max(time_idx)

    path = INPUT_PATH.joinpath(f'epsilon{epsilon}').joinpath(f'description_{description_idx}.json')
    
    # BN generator 생성
    generator = DataGenerator()
    generator.generate_dataset_in_correlated_attribute_mode(max_time, path)
    df = generator.synthetic_dataset
    
    # time 선별
    df = df[df.TIME.isin(time_idx)].copy().reset_index(drop=True)
    
    # DG_RCNF, DEAD 1 인 기록은 삭제
    df['DEAD_NFRM_DEAD'] = 0
    df['DG_RCNF_RLPS'] = 0 
    
    if original_data['DEAD_NFRM_DEAD'].sum() > 0 :
        death_time = original_data[original_data.DEAD_NFRM_DEAD == 1]['TIME'].to_list()[0]
        death_time = random.sample(range(death_time-5, death_time+5), 1)[0]
        
        new_row = pd.DataFrame([{"PT_SBST_NO": sample_number, "TIME": death_time, "DEAD_NFRM_DEAD": 1.0}])
        df = pd.concat([df, new_row])
    
    if original_data['DG_RCNF_RLPS'].sum() > 0 : 
        relapse_time = original_data[original_data.DG_RCNF_RLPS == 1]['TIME'].to_list()
        for time in relapse_time:
            time = random.sample(range(time-5, time+5), 1)[0]
            new_row = pd.DataFrame([{"PT_SBST_NO": sample_number, "TIME": time, "DG_RCNF_RLPS": 1.0}])
            df = pd.concat([df, new_row])
    
    df = df.sort_values(by="TIME")
    df = df.reset_index(drop=True)
    index = min(df[df.DEAD_NFRM_DEAD == 1].index.values)
    df = df.loc[0:index,].copy()

    df.to_pickle(output_path.joinpath(f'synthetic_data_{sample_number}.pkl'))


#%%
def return_description_files(path : Path):
    import os, sys
    files = os.listdir(path)
    
    f = lambda x : 'description' in x

    files = sorted(list(filter(f, files)))
    return files 


#%%
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--epsilon','-e', type=float, help='choose the epsilon')
    parser.add_argument('--multiplier', '-m', type=int ,help='how much time to resample the data')
    parser.add_argument('--sample', '-s', type=bool, default=False,help='whether you are going to sample')
    parser.add_argument('--sample_number', '-sn', type=int, default=0, help='how much you are going to sample')
    
    args = parser.parse_args()
    
    
    files = return_description_files(INPUT_PATH.joinpath(f'epsilon{args.epsilon}'))
    if args.sample :
        files = random.sample(files, args.sample_number)
        files = files*args.multiplier
        
    pseudo_patient_id = [i for i in range(0, args.multiplier*len(files))]
    
    with _ProcessPool(8) as p:
        p.starmap(generate_data, zip(repeat(args.epsilon), files, pseudo_patient_id))

if __name__ == "__main__":
    main()
    