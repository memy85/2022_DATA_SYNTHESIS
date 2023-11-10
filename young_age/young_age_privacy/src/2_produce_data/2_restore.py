
'''
베이지안 네트워크 돌린 후 복원하는 코드
'''
#%%
import os, sys
from pathlib import Path
import random
import argparse
from sklearn.preprocessing import LabelEncoder

project_path = Path(__file__).absolute().parents[2]
# project_path = Path().cwd()

print(f"this is project_path : {project_path.as_posix()}")
os.sys.path.append(project_path.as_posix())

from src.MyModule.utils import *
#%%

config = load_config()

#%%

import datetime
import pandas as pd
import pickle

def decode(whole_encoded_df, tables, bind_data_columns):
    
    np_encoded = np.array(whole_encoded_df)
    np_encoded = np_encoded.astype(str)
    restored = pd.DataFrame()

    for k in range(len(np_encoded.transpose())):
        temp1 = []
        for i in np_encoded.transpose()[k]:
            temp2=[]
            a =''
            for j in range(len(i)):
                a+=i[j]
                if((j+1)%3==0):
                    temp2.append(a)
                    if(len(a)!=3):
                        print('error')
                    a=''
            
            temp1.append(temp2)
        sep_df = pd.DataFrame(temp1)
        restored = pd.concat([restored,sep_df],axis=1)
        
    cols = []
    for head in tables:
        # originally bind.filter(like=head)
        columns = list(filter(lambda x : head in x, bind_data_columns))
        for col in columns:
            cols.append(col)
    #restored.columns = cols
    
    return restored

#%%

def restore_day(data, target, multiplier):
    days_arr = []
    for days in data[target]:
        restored_days = 0
        if days != 0 and days != 999:
            restored_days = days * multiplier + random.randrange(-int(multiplier/2), int(multiplier/2)+1,1)

        elif days == 0 :
            restore_days = days + random.randrange(0, int(multiplier/2)+1,1)
        elif days == 999 :
            restore_days = np.NaN
        days_arr.append(restored_days)

    data[target] = days_arr
    return data

                                  
#%%

def argument_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--age", default = 50, type = int)
    parser.add_argument("--random_seed", default = 0 , type = int)
    
    args = parser.parse_args()
    return args


def main():
    args = argument_parse()
    random_seed = args.random_seed
    age = args.age

#%%
    # random_seed = 0
    # age = 50
    raw_path = get_path("data/raw/D0_Handmade_ver2.csv")

    project_path = Path(config["project_path"])
    # input_path = get_path("data/processed/no_bind")
    input_path = get_path(f"data/processed/seed{random_seed}/2_produce_data")
    preprocess_1_path = get_path(f"data/processed/seed{random_seed}/1_preprocess")
    output_path = get_path(f"data/processed/seed{random_seed}/2_produce_data")
    # output_path = get_path("data/processed/no_bind/restored")
    if not output_path.exists() : 
        output_path.mkdir(parents=True)

    data = pd.read_csv(raw_path)

    #%% read bind colums
    bind_columns = pd.read_pickle(project_path.joinpath(f"data/processed/seed{random_seed}/1_preprocess/bind_columns_{args.age}.pkl"))
    # bind_columns = pd.read_pickle(project_path.joinpath(f"data/processed/1_preprocess/bind_columns_{50}.pkl"))

    #%%
    tables= []
    for col in bind_columns:
            tables.append('_'.join(col.split('_')[0:1]))

    epsilons = config['epsilon']

    for epsilon in epsilons:

        # syn = pd.read_csv(input_path.joinpath(f'S0_mult_encoded_{epsilon}_{args.age}.csv'))
        syn = pd.read_csv(input_path.joinpath(f'S0_mult_encoded_{epsilon}_{50}.csv'))

        try:
            syn = syn.drop(columns = ['Unnamed: 0'])

        except:
            pass
       #%%
        syn = syn.astype(str)
        # need to comment it if you do it with bind columns
        # syn = syn.replace('nan', 999)

        # uncomment if the columns are binded!!
        for col in syn.iloc[:,11:]:
            syn[col] =syn[col].str.replace('r','')
            
        decoded = decode(syn.iloc[:,11:], tables, bind_columns)
        decoded.columns = bind_columns

        syn = pd.concat([syn.iloc[:,:11],decoded],axis=1)
        syn = syn.rename(columns = {'RLPS DIFF' : 'RLPS_DIFF'})
         
        #%%
        # # continous restore    
        syn['BSPT_IDGN_AGE'] = syn['BSPT_IDGN_AGE'].astype(int)
        ages=[]
        for age in syn['BSPT_IDGN_AGE']:
            if age < 10:
                restored_age = age * 5 + random.randrange(-2,3,1)
            else:
                restored_age = age * 5 + random.randrange(-2,0,1)
            ages.append(restored_age)
            
        syn['BSPT_IDGN_AGE'] =  ages

        days = ['OVR_SURV','RLPS_DIFF','DEAD_DIFF']

        for i in range(1,9):
            days.append(f'REGN_TIME_DIFF_{i}')
            days.append(f'REGN_START_DIFF_{i}')

        for col in days:
            syn[col] = syn[col].astype(float)
            num = 30
            if col == 'OVR_SURV':
                num = 90
            elif col[:4] == 'REGN':
                num = 15
            syn = restore_day(syn, col, num)

        # uncomment if the columns are binded!!
        for col in list(syn.columns)[11:]:
            syn[col] = syn[col].astype(float).astype(int)
        
        # Label Encoding for ml
        ml_data = syn.copy()
        encoder = LabelEncoder() 
        #%%
        
        ####### ML data #############################################################################################################################

        for col in ['BSPT_SEX_CD', 'BSPT_FRST_DIAG_NM', 'BSPT_STAG_CLSF_CD']:
            ml_data[col] = ml_data[col].astype(str)
            encoder.fit(ml_data[col])
            trans = encoder.transform(ml_data[col])
            
            ml_data[col] = trans
        
        ml_data = ml_data.replace(999,np.NaN)
        ml_data = ml_data.replace('999',np.NaN)

        if not output_path.joinpath(f'synthetic_decoded').exists():
            output_path.joinpath(f'synthetic_decoded').mkdir(parents=True)
#%%
                    
        ml_data.to_csv(output_path.joinpath(f'synthetic_decoded/Synthetic_data_epsilon{epsilon}_{args.age}.csv'), index=False)

#%%

        # date time restore with randomly
        start_date = min(data['BSPT_FRST_DIAG_YMD'])
        end_date = max(data['BSPT_FRST_DIAG_YMD'])
        
        date_range = pd.date_range(start_date, end_date,freq='D')

        date = []
        for _ in range(len(syn)):
            date.append(random.choice(date_range))
        
        syn.insert(4,'BSPT_FRST_DIAG_YMD', date)    

        
        for col in ['RLPS', 'DEAD']:
            diff = []
            for i in range(len(syn)):        
                try:
                    diff.append((syn['BSPT_FRST_DIAG_YMD'].iloc[i] + datetime.timedelta(days=syn[col+'_DIFF'].iloc[i])).strftime("%Y-%m-%d"))
                except:
                    diff.append(0)
            syn.insert(list(syn.columns).index(col)+1 , col+'_YMD' , diff)

        for i in range(1,9):
            start_diff = []
            end_diff = []
            for j in range(len(syn)):        
                if syn[f'REGN_START_DIFF_{i}'].iloc[j] != 0:
                    start_day = (syn['BSPT_FRST_DIAG_YMD'].iloc[j] + datetime.timedelta(days=int(syn[f'REGN_START_DIFF_{i}'].iloc[j]))).strftime("%Y-%m-%d")
                    end_day = datetime.datetime.strptime(start_day,"%Y-%m-%d") + datetime.timedelta(days=int(syn[f'REGN_TIME_DIFF_{i}'].iloc[j]))
                    start_diff.append(start_day)
                    end_diff.append(end_day.strftime("%Y-%m-%d"))
                else:
                    start_diff.append(np.NaN)
                    end_diff.append(np.NaN)

            syn[f'REGN_START_DIFF_{i}'] = start_diff

            syn.insert(list(syn.columns).index(f'REGN_START_DIFF_{i}')+1 , f'REGN_END_DAY_{i}' , end_diff)    
            syn = syn.rename({f'REGN_START_DIFF_{i}':f'REGN_START_DAY_{i}'})
         
        # read encoder, list-like
        encoders = pd.read_pickle(preprocess_1_path.joinpath(f"LabelEncoder_{args.age}.pkl"))
        # encoders = pd.read_pickle(preprocess_1_path.joinpath(f"LabelEncoder_{50}.pkl"))

#%%
        # numeric to context
        for col, encoder in encoders:
            try:
                # syn[syn.columns[i+14]] = encoders[i].inverse_transform(syn[syn.columns[i+14]])
                syn[col] = encoder.inverse_transform(syn[col])
            except:
                pass

        syn = syn.rename({'REGN_IMPT_HP2E_RSLT_NM':'IMPT_HP2E_RSLT_NM'})
        syn = syn.replace(999,np.NaN)

        pkl_encode = pd.read_pickle(preprocess_1_path.joinpath(f"label_dict_{args.age}.pkl")) 

        for key, valDict in pkl_encode.items():
            inverse = {}
            for k, v in pkl_encode[key].items():
                inverse[v] = k
            try:
                syn[key] = syn[key].replace(inverse)
            except:
                pass

        save_path = output_path.joinpath("synthetic_restore")
        if not save_path.exists():
            save_path.mkdir(parents=True)

        syn.to_csv(save_path.joinpath(f'Synthetic_data_epsilon{epsilon}_{args.age}.csv'),encoding='cp949', index=False)
#%%

if __name__ == "__main__" : 
    main()


#%%


#import os, sys
#from pathlib import Path
#import random
#import argparse
#from sklearn.preprocessing import LabelEncoder

## project_path = Path(__file__).absolute().parents[2]
#project_path = Path().cwd()

#print(f"this is project_path : {project_path.as_posix()}")
#os.sys.path.append(project_path.as_posix())

#from src.MyModule.utils import *
##%%

#config = load_config()
#project_path = Path(config["project_path"])
#input_path = get_path("data/processed/2_produce_data")
#preprocess_1_path = get_path("data/processed/preprocess_1")
#output_path = get_path("data/processed/2_produce_data")
#if not output_path.exists() : 
#    output_path.mkdir(parents=True)

##%%

#import datetime
#import pandas as pd
#import pickle



#age = 50
##%% read bind colums
#bind_columns = pd.read_pickle(project_path.joinpath(f"data/processed/preprocess_1/bind_columns_{age}.pkl"))

##%%

#bind_columns

##%%
#tables= []
#for col in bind_columns:
#        tables.append('_'.join(col.split('_')[0:1]))

##%%
#epsilons = config['epsilon']
##%%
#syn = pd.read_csv(input_path.joinpath(f'S0_mult_encoded_{epsilons[1]}_{age}.csv'))
##%%

#ori_path = preprocess_1_path.joinpath("encoded_D0_to_syn_50.csv")
#ori = pd.read_csv(ori_path)


##%%
#ori['SGPT'].apply(lambda x : x.replace('r'))
