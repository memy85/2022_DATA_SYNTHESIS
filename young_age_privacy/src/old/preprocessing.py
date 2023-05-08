


# |%%--%%| <ziib5ebUZy|4hFKh355rw>

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import random
import matplotlib.pyplot as plt

# |%%--%%| <4hFKh355rw|Ppsi8f7uq5>

def decode(whole_encoded_df, tables, data):
    
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
        for col in data.filter(like=head).columns:
            cols.append(col)
    #restored.columns = cols
    
    return restored

# |%%--%%| <Ppsi8f7uq5|HQd4QmjbaE>

def restore_day(data,target,multiplier):

    days_arr=[]
    for days in data[target]:
        restored_days = 0
        if days != 0 and days != 999:
            restored_days = days * multiplier + random.randrange(-int(multiplier/2),int(multiplier/2)+1,1)
        elif days ==0:
            restore_days = days + random.randrange(0, int(multiplier/2)+1,1)
        elif days == 999:
            restore_days = np.NaN
        days_arr.append(restored_days)
        
    data[target] =  days_arr

    return data

# |%%--%%| <HQd4QmjbaE|JWFV0GwNwY>

data = pd. read_excel('/home/wonseok/2022_DATA_SYNTHESIS/young_age/data/raw/D0_Handmade_ver1.1.xlsx')
data = data.replace('x',999)
data = data.replace('Not Data', np.NaN)

# |%%--%%| <JWFV0GwNwY|6l9sdDVpqq>

data.head(2)

# |%%--%%| <6l9sdDVpqq|8AmlUSlzBO>

# exclude creiteria

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

# |%%--%%| <8AmlUSlzBO|uw6eTmg0DY>

standalone = data.iloc[:,:12]
bind = data.iloc[:,12:]

# |%%--%%| <uw6eTmg0DY|AK5vGWqNmC>

# categoriziing continous value

standalone['RLPS DIFF'] =  (((standalone['RLPS_DIAG_YMD'] - standalone['BSPT_FRST_DIAG_YMD']) .dt.days)/30).round()
standalone['DEAD'] = data['DEAD']
standalone['BSPT_IDGN_AGE'] = (standalone['BSPT_IDGN_AGE']/5).round().astype(int)
standalone['DEAD_DIFF'] = (((data['BSPT_DEAD_YMD'] - data['BSPT_FRST_DIAG_YMD']).dt.days)/30).round()
standalone['OVR_SURV'] = (((data['CENTER_LAST_VST_YMD'] - data['BSPT_FRST_DIAG_YMD']).dt.days)/90).round()
standalone = standalone.drop(['BSPT_FRST_DIAG_YMD','RLPS_DIAG_YMD','5YR_RLPS','CENTER_LAST_VST_YMD','BSPT_DEAD_YMD'], axis =1)

# |%%--%%| <AK5vGWqNmC|FcWjdtSpJa>

bind = bind.drop(['DEAD','5YR_DEAD','MLPT_ACPT_YMD','BPTH_ACPT_YMD'],axis=1)
#bind = bind.iloc[:,5:]
bind['OPRT_YMD'] = (((bind['OPRT_YMD'] - data['BSPT_FRST_DIAG_YMD']).dt.days)/15).round()


col = list(bind.iloc[:,:26].columns)
regn_col= list('REGN_' + bind.iloc[:,26:].columns)

col = col+regn_col
bind.columns = col

# |%%--%%| <FcWjdtSpJa|oNb1OBkvXH>

data.columns.__len__()
# 본래 컬럼의 수는 91개

# |%%--%%| <oNb1OBkvXH|44B7pP1vJI>

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

# |%%--%%| <44B7pP1vJI|9BErhQDJBG>

# This encodes the data

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
        

# |%%--%%| <9BErhQDJBG|xUYmag4Mpj>

bind.head()
# 여전히 Nan이 존재하지만 대부분의 값들을 인코딩하였다.
bind.dtypes.unique()

# |%%--%%| <xUYmag4Mpj|WBnkM9Rsin>

# binding columns by the source tables 

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

# |%%--%%| <WBnkM9Rsin|Y6m87p1tqW>

# replacing the weird values in the columns

for i in range(len(temp_df)):
    temp_df[i] = temp_df[i].replace(np.NaN, 999)
    temp_df[i] = temp_df[i].astype(int).astype(str)
    for j in range(10):
        temp_df[i] = temp_df[i].replace(str(j),'00'+str(j))
    for k in range(10,100):
        temp_df[i] = temp_df[i].replace(str(k),'0'+str(k))

# |%%--%%| <Y6m87p1tqW|NZLBM57A0Y>

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

# |%%--%%| <NZLBM57A0Y|ji46egCDSR>

whole_encoded_df = results[0]
for df in results[1:]:
    whole_encoded_df = pd.concat([whole_encoded_df, df],axis=1)

# |%%--%%| <ji46egCDSR|0XT2OI1HSl>

whole_encoded_df = whole_encoded_df +'r'

# |%%--%%| <0XT2OI1HSl|wlRTiByNbf>

pd.concat([standalone, whole_encoded_df],axis=1).to_csv('encoded_D0.csv',index_label=False)

# |%%--%%| <wlRTiByNbf|vt9bhFuJGb>

unmodified_D0 = pd.concat([standalone,bind],axis=1)
encoders = []
for col in unmodified_D0.columns:
    try:
        unmodified_D0[col].astype(float)
        #encoders.append('non')
    except:
        unmodified_D0[col].astype(str)
        encoder = LabelEncoder()
        encoder.fit(unmodified_D0[col])
        encoders.append(encoder)
        trans = encoder.transform(unmodified_D0[col])
        unmodified_D0[col] = trans
        

# |%%--%%| <vt9bhFuJGb|U4dkRUp23y>

data.filter(like="REGN")

# |%%--%%| <U4dkRUp23y|T5VGkOI6kN>

encoded = pd.read_csv('encoded_D0.csv')
sampled = encoded.sample(int(len(encoded)*0.8))
valid = unmodified_D0.drop(sampled.index)


valid['RLPS DIFF'] =  (data['RLPS_DIAG_YMD'] - data['BSPT_FRST_DIAG_YMD']).dt.days
valid['BSPT_IDGN_AGE'] = data['BSPT_IDGN_AGE']
valid['DEAD_DIFF'] = (data['BSPT_DEAD_YMD'] - data['BSPT_FRST_DIAG_YMD']).dt.days
valid['OVR_SURV'] = (data['CENTER_LAST_VST_YMD'] - data['BSPT_FRST_DIAG_YMD']).dt.days

valid['OPRT_YMD'] = ((data['OPRT_YMD'] - data['BSPT_FRST_DIAG_YMD']).dt.days)

for i in range(1,9):
    start = pd.to_datetime(data[f'REGN_TRTM_CASB_STRT_YMD{i}'],format='%Y%m%d')
    end = pd.to_datetime(data[f'REGN_TRTM_CASB_CSTR_YMD2_{i}'],format='%Y%m%d')
    
    monthly_diff = (end-start).dt.days
    start_diff = (start-data['BSPT_FRST_DIAG_YMD']).dt.days
    
    valid[f'REGN_TIME_DIFF_{i}'] = monthly_diff
    valid[f'REGN_START_DIFF_{i}'] = start_diff

sampled.to_csv('encoded_D0_to_syn.csv')

# |%%--%%| <T5VGkOI6kN|CIwfIc0JyW>

pd.read_csv('/home/dogu86/young_age_colon_cancer/final_data/synthetic_decoded/Synthetic_data_epsilon10000.csv')

# |%%--%%| <CIwfIc0JyW|TwnTvhs6xi>

import datetime

epsilons = [10000]
for epsilon in epsilons:
    syn = pd.read_csv(f'/home/dogu86/young_age_colon_cancer/final_data/synthetic/S0_mult_encoded_{epsilon}_degree2.csv')
    try:
        syn = syn.drop('Unnamed: 0', axis=1)
    except:
        pass
    syn = syn.astype(str)
    for col in syn.iloc[:,11:]:
        syn[col] =syn[col].str.replace('r','')
        
        
    decoded = decode(syn.iloc[:,11:], tables, bind)
    decoded.columns = bind.columns
    syn = pd.concat([syn.iloc[:,:11],decoded],axis=1)
    syn = syn.rename(columns = {'RLPS DIFF' : 'RLPS_DIFF'})
    
    # continous restore    
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
    
    
    for col in list(syn.columns)[11:]:
        syn[col] = syn[col].astype(int)
    
    
    # Label Encoding for ml
    ml_data=syn.copy()
    
    for col in ['BSPT_SEX_CD', 'BSPT_FRST_DIAG_NM', 'BSPT_STAG_CLSF_CD']:
        ml_data[col].astype(str)
        encoder.fit(ml_data[col])
        trans = encoder.transform(ml_data[col])
        
        ml_data[col] = trans
    
    ml_data = ml_data.replace(999,np.NaN)
    ml_data = ml_data.replace('999',np.NaN)

    ml_data.to_csv(f'/home/dogu86/young_age_colon_cancer/final_data/synthetic_decoded/Synthetic_data_epsilon{epsilon}.csv')
    
    ####################################################################################################################################
    
    # date time restore with randomly
    start_date = min(data['BSPT_FRST_DIAG_YMD'])
    end_date = max(data['BSPT_FRST_DIAG_YMD'])
    
    date_range = pd.date_range(start_date,end_date,freq='D')

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
        
        
    # numeric to context
    for i in range(len(encoders)):
        try:
            syn[syn.columns[i+14]] = encoders[i].inverse_transform(syn[syn.columns[i+14]])
        except:
            pass

    syn = syn.rename({'REGN_IMPT_HP2E_RSLT_NM':'IMPT_HP2E_RSLT_NM'})
    syn = syn.replace(999,np.NaN)
    #syn = syn.replace(0,'No Data')

    pkl_encode =pd.read_pickle('LabelEncoder.pkl')

    for key in pkl_encode.keys():
        inverse = {}
        for k, v in pkl_encode[key].items():
            inverse[v] = k
        try:
            syn[key] = syn[key].replace(inverse)
        except:
            pass


    syn.to_csv(f'/home/dogu86/young_age_colon_cancer/final_data/synthetic_restore/Synthetic_data_epsilon{epsilon}.csv',encoding='cp949')

# |%%--%%| <TwnTvhs6xi|1OJ3VRkpnN>

import pickle
encode_dict

with open('LabelEncoder.pkl', 'wb') as f:
    pickle.dump(encode_dict, f)

# |%%--%%| <1OJ3VRkpnN|FkfrBFaaTf>

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

input_path = Path("/home/wonseok/projects/2022_DATA_SYNTHESIS/young_age/")
df = pd.read_excel(input_path.joinpath('data/raw/D0_Handmade_ver1.1.xlsx'))

# |%%--%%| <FkfrBFaaTf|vXqJ4VCrix>

df.BSPT_SEX_CD.hist()
plt.show()

# |%%--%%| <vXqJ4VCrix|feg6jDng7X>

df.filter(like="TRTM")
