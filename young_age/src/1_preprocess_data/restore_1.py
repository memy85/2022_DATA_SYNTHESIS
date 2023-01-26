'''
베이지안 네트워크 돌린 후 복원하는 코드
'''
import datetime

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

#%%


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
#%%