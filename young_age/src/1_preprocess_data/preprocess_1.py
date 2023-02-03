#%%

from pathlib import Path
import os, sys
import argparse

# project_path = Path(__file__).absolute().parents[2]
project_path = Path("/home/wonseok/projects/2022_DATA_SYNTHESIS/young_age")
print(f"this is project_path : {project_path.as_posix()}")
os.sys.path.append(project_path.as_posix())

from src.MyModule.utils import *
#%%

config = load_config()
project_path = Path(config["project_path"])
input_path = get_path("data/raw")
output_path = get_path("data/processed/preprocess_1")
if not output_path.exists() : 
    output_path.mkdir(parents=True)

import pandas as pd
import numpy as np
import random
import pickle
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt



#%%
def argument_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument("--age", type = int, default = 50)
    args = parser.parse_args()
    return args


#%%
def main():
    args = argument_parse()
    
    data = pd. read_excel(project_path.joinpath('data/raw/D0_Handmade_ver1.1.xlsx'))
    #%% filter patients under age
    data = data[data.BSPT_IDGN_AGE <= args.age].copy()
    data = data.replace('x',999)
    data = data.replace('Not Data', np.NaN)

    # exclude criteria
    # overall observation day is under 30days

    data = data[(data.CENTER_LAST_VST_YMD - data.BSPT_FRST_DIAG_YMD).dt.days >= 30]
    # missing in whole stage value
    cond1 = data['BSPT_STAG_VL'] != 999
    data = data[cond1].dropna(subset=['BSPT_STAG_VL'])

    # Surgical T Stage value is missing, but have operation report
    cond1 = data['OPRT_YMD'].isnull()==True
    cond2 = data['SGPT_PATL_T_STAG_VL'].isnull()==True
    data = data.drop(data[cond1&cond2].index)

    data = data.drop(['OVRL_SRVL_DTRN_DCNT','RLPS_DTRN_DCNT'],axis=1)

    # whole data length : 1501 -> after apply exclude criteria : 1253
    # %%
    # standalone : columns that does not require binding
    # bind : columns that will be binded

    standalone = data.iloc[:,:12]
    bind = data.iloc[:,12:]

    #%%
    # categoriziing continous value
    # standalone : this is a dataset that creates continuous to categorical columns 

    standalone['RLPS DIFF'] =  (((standalone['RLPS_DIAG_YMD'] - standalone['BSPT_FRST_DIAG_YMD']) .dt.days)/30).round()
    standalone['DEAD'] = data['DEAD']
    standalone['BSPT_IDGN_AGE'] = (standalone['BSPT_IDGN_AGE']/5).round().astype(int)
    standalone['DEAD_DIFF'] = (((data['BSPT_DEAD_YMD'] - data['BSPT_FRST_DIAG_YMD']).dt.days)/30).round()
    standalone['OVR_SURV'] = (((data['CENTER_LAST_VST_YMD'] - data['BSPT_FRST_DIAG_YMD']).dt.days)/90).round()
    standalone = standalone.drop(['BSPT_FRST_DIAG_YMD','RLPS_DIAG_YMD','5YR_RLPS','CENTER_LAST_VST_YMD','BSPT_DEAD_YMD'], axis =1)

    # manipulating binds 
    bind = bind.drop(['DEAD','5YR_DEAD','MLPT_ACPT_YMD','BPTH_ACPT_YMD'],axis=1)
    bind['OPRT_YMD'] = (((bind['OPRT_YMD'] - data['BSPT_FRST_DIAG_YMD']).dt.days)/15).round()

    col = list(bind.iloc[:,:26].columns)
    regn_col= list('REGN_' + bind.iloc[:,26:].columns)

    col = col+regn_col
    bind.columns = col

    #%% Here chaning the regimen data
    for i in range(1,9):
        start = pd.to_datetime(bind[f'REGN_TRTM_CASB_STRT_YMD{i}'],format='%Y%m%d')
        end = pd.to_datetime(bind[f'REGN_TRTM_CASB_CSTR_YMD2_{i}'],format='%Y%m%d')
        
        monthly_diff = (((end-start).dt.days)/30).round()
        start_diff = (((start-data['BSPT_FRST_DIAG_YMD']).dt.days)/15).round()
        
        bind[f'REGN_TRTM_CASB_STRT_YMD{i}'] = monthly_diff
        bind[f'REGN_TRTM_CASB_CSTR_YMD2_{i}'] = start_diff
        
        bind.rename(columns= {f'REGN_TRTM_CASB_STRT_YMD{i}':f'REGN_TIME_DIFF_{i}'},inplace=True)
        bind.rename(columns= {f'REGN_TRTM_CASB_CSTR_YMD2_{i}':f'REGN_START_DIFF_{i}'},inplace=True)
        #bind.drop(f'REGN_TRTM_CASB_CSTR_YMD2_{i}',axis=1,inplace = True)

    #%%
    # changning all the data into a encoded form
    # encode_dic == label_dict, they are the same
    # label_dict is for binded columsn
    encoders = []
    encode_dict = dict()
    for col in bind.columns:
        try:
            bind[col].astype(float)
            encoders.append('non')
        except:
            bind[col].astype(str)
            encoder = LabelEncoder()
            encoder.fit(bind[col])
            
            x = {key:i for key, value in dict.fromkeys(encoder.classes_).items()}
            for i, key in enumerate(x.keys()): 
                x[key] = i

            encode_dict[col] = x
            
            encoders.append(encoder)        
            trans = encoder.transform(bind[col])
            bind[col] = trans
            
    #%% save bind_columns information and encodings
    with open(output_path.joinpath(f"label_dict_{args.age}.pkl"), 'wb') as f:
        pickle.dump(encode_dict, f)
        
    with open(output_path.joinpath(f"bind_columns_{args.age}.pkl"), 'wb') as f:
        pickle.dump(bind.columns.to_list(), f)

    #%%
    """
    creates binded data format
    """
    
    tables= []
    for col in bind.columns:
        tables.append('_'.join(col.split('_')[0:1]))
        

    result1 = dict.fromkeys(tables)
    uniq_tables = list(result1)
    print(uniq_tables)

    temp_df=[]
    for uniq in uniq_tables:
        temp_series = []
        for col in bind.columns:
            if uniq == '_'.join(col.split('_')[0:1]):
                temp_series.append(bind[col])
                
        temp_df.append(pd.DataFrame(temp_series))
    #%%

    for i in range(len(temp_df)):
        temp_df[i] = temp_df[i].replace(np.NaN, 999)
        temp_df[i] = temp_df[i].astype(int).astype(str)
        for j in range(10):
            temp_df[i] = temp_df[i].replace(str(j),'00'+str(j))
        for k in range(10,100):
            temp_df[i] = temp_df[i].replace(str(k),'0'+str(k))
            
    #%%
    # joining the splitted table columns 
    results = []
    concated = pd.DataFrame()

    for i in range(len(temp_df)):
        result = temp_df[i].transpose().iloc[:,0]
        for j in range(1,len(temp_df[i])):
            result += temp_df[i].transpose().iloc[:,j]
            
        a = pd.DataFrame(result)
        col = '_'.join(((uniq_tables)[i].split('_'))[0:1])
        a= a.rename(columns = {result.name : col})
        
        results.append(a)
        
    # results captivates values that are split    
    #%%
    # make the results into a one dataframe
    whole_encoded_df = results[0]
    for df in results[1:]:
        whole_encoded_df = pd.concat([whole_encoded_df, df],axis=1)

    #%%
    whole_encoded_df = whole_encoded_df + 'r'

    pd.concat([standalone, whole_encoded_df], axis=1).to_csv(output_path.joinpath(f"encoded_D0_{args.age}.csv"), index_label=False)

    #%%
    # unmodified : Now we make the D0 that is the same format as the input for bayesian
    # For the columns in uD0, we change the variables into strings

    unmodified_D0 = pd.concat([standalone,bind], axis=1)
    # unmodified_D0.to_csv(output_path.joinpath(f'encoded_D0_{args.age}.csv'),index_label=False)

    encoders = []
    for col in unmodified_D0.columns:
        try:
            unmodified_D0[col].astype(float)
        except:
            unmodified_D0[col].astype(str)
            encoder = LabelEncoder()
            encoder.fit(unmodified_D0[col])
            encoders.append(encoder)
            trans = encoder.transform(unmodified_D0[col])
            unmodified_D0[col] = trans
           
    #%% save encoder
    with open(output_path.joinpath(f"LabelEncoder_{args.age}.pkl"), 'wb') as f:
        pickle.dump(encoders, f)

    # save unmodified
    unmodified_D0.to_pickle(output_path.joinpath(f"unmodified_D0_{args.age}.pkl"))
    #%% split train and valid      

    encoded = pd.read_csv(output_path.joinpath(f'encoded_D0_{args.age}.csv'))
    sampled = encoded.sample(frac= 0.8, random_state = 0)
    train_idx = sampled.index

    with open(output_path.joinpath("train_idx.pkl"), 'wb') as f:
        pickle.dump(train_idx, f)

    valid = unmodified_D0.loc[~encoded.index.isin(train_idx)]

    #%%
    valid["RLPS DIFF"] = (data["RLPS_DIAG_YMD"] - data["BSPT_FRST_DIAG_YMD"]).dt.days
    valid["BSPT_IDGN_AGE"]  = data["BSPT_IDGN_AGE"]
    valid["DEAD_DIFF"] = (data["BSPT_DEAD_YMD"] - data["BSPT_FRST_DIAG_YMD"]).dt.days
    valid["OVR_SURV"] = (data["CENTER_LAST_VST_YMD"]- data["BSPT_FRST_DIAG_YMD"]).dt.days
    # valid["OPRT_SURV"] = (data["CENTER_LAST_VST_YMD"]- data["BSPT_FRST_DIAG_YMD"]).dt.days

    for i in range(1,9):
        start = pd.to_datetime(data[f'TRTM_CASB_STRT_YMD{i}'], format = "%Y%m%d")
        end = pd.to_datetime(data[f'TRTM_CASB_CSTR_YMD2_{i}'], format = "%Y%m%d")

        monthly_diff = (end-start).dt.days
        start_diff = (start-data["BSPT_FRST_DIAG_YMD"]).dt.days
        valid[f"REGN_TIME_DIFF_{i}"] = monthly_diff
        valid[f"REGN_START_DIFF_{i}"] = start_diff

    #%% 
  
    sampled.to_csv(output_path.joinpath(f'encoded_D0_to_syn_{args.age}.csv'), index=False)
    valid.to_csv(output_path.joinpath(f'encoded_D0_to_valid_{args.age}.csv'), index=False)





#%%
if __name__ == "__main__" : 
    main()

#%%

# check code part 

#from pathlib import Path
#import os, sys
#import argparse

## project_path = Path(__file__).absolute().parents[2]
#project_path = Path("/home/wonseok/projects/2022_DATA_SYNTHESIS/young_age")
#print(f"this is project_path : {project_path.as_posix()}")
#os.sys.path.append(project_path.as_posix())

#from src.MyModule.utils import *
##%%

#config = load_config()
#project_path = Path(config["project_path"])
#input_path = get_path("data/raw")
#output_path = get_path("data/processed/preprocess_1")
#if not output_path.exists() : 
#    output_path.mkdir(parents=True)

##%%

#df = pd.read_csv(output_path.joinpath("encoded_D0_to_syn_50.csv"))



