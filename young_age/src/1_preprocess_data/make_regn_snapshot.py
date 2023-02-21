import pandas as pd
import numpy as np
from pathlib import Path
import os, sys
import argparse


temp = pd.read_excel('/mnt/synthetic_data/data/raw/CLRC_TRTM_CASB.xlsx')
pt_num = list(pd.read_csv('/home/dogu86/pt_num.csv')['PT_SBST_NO']) # under 50 patients number

print(f'Target patient number : {str(len(pt_num))}')

regn = temp[temp['PT_SBST_NO'].isin(pt_num)].drop(['CENTER_CD','IRB_APRV_NO'],axis =1)

def transform_format(data, syn = False, droped = False):
    
    counts = []  
    counts = list(data.groupby(by ='PT_SBST_NO').count()['CSTR_STRT_YMD'])
    #data =get_days(data)
    if droped == True:
        data.drop('PT_SBST_NO', axis = 1 ,inplace = True)
    np_input = data.to_numpy()
    
    input_form= []

    cur = 0
    for count in list(counts):
        input_form.append(np_input[cur:cur+count])
        cur += count
    input_form = np.array(input_form)
    
    return input_form


def main():
    np_regn = transform_format(regn)

    len_arr = []
    for i in range(len(np_regn)):
        len_arr.append(len(np.unique(np_regn[i].transpose()[4])))

    snapshot = []
    for num in pt_num:
        pt0 = []
        pt0.append(num)
        
        if num in list(temp['PT_SBST_NO'].unique()):
            for i in range(len(np_regn)):
                if(num == np_regn[i].transpose()[0][0]):
                    cycle_col = []
                    cycle = np_regn[i].transpose()[9]

                    cycle_col = []
                    cycle = np_regn[i].transpose()[9]

                    cycle_info = []
                    for j in range(len(cycle)):
                        try:
                            if(cycle[j]-cycle[j+1]) < 0:
                                cycle_info.append(np_regn[i][j])
                                
                            else:
                                cycle_info.append(np_regn[i][j])
                                cycle_col.append(cycle_info)
                                
                                cycle_info =[]
                        except:
                            cycle_info.append(np_regn[i][j])
                            cycle_col.append(cycle_info)
                        
                    for k in range(len(cycle_col)):
                        try:
                            cycle_col = np.array(cycle_col)
                            cycle_col[k] = np.array(cycle_col[k])
                            pt0.append(np.unique(cycle_col[k].transpose()[4]))
                            if len(np.unique(cycle_col[k].transpose()[5])) == 1:
                                pt0.append(np.unique(cycle_col[k].transpose()[5])[0])
                            else:
                                print("ERROR :: Number of Therphy is not unique value where in 1 cylce")
                                break
                            pt0.append(np.unique(cycle_col[k].transpose()[8]))
                            pt0.append(cycle_col[k].transpose()[1][0])
                            pt0.append(cycle_col[k].transpose()[6][-1])
                            pt0.append(cycle_col[k].transpose()[9][-1])
                        except:
                            for _ in range(5):
                                pt0.append(np.NaN)
        else:
            for _ in range(40):
                pt0.append("Not Data")
        
        snapshot.append(pt0)
    new_col = []
    new_col.append('PT_SBST_NO')
    #new_col = []
    for i in range(1,9):
        new_col.append('CSTR_REGN_NM_'+str(i))
        new_col.append('CSTR_PRPS_NT_'+str(i))
        new_col.append('CASB_CSTR_PRPS_NM_'+str(i))
        new_col.append('TRTM_CASB_STRT_YMD'+str(i))
        new_col.append('TRTM_CASB_CSTR_YMD2_'+str(i))
        new_col.append('CSTR_CYCL_VL_END_'+str(i))

    new_d0 = pd.DataFrame(snapshot,columns=new_col)
    new_d0.to_csv('CSTR_originated.csv', encoding='utf-8-sig')