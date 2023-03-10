#%%
from pathlib import Path
import os, sys
import argparse

project_path = Path(__file__).absolute().parents[2]
# project_path = Path().cwd()
os.sys.path.append(project_path.as_posix())

#%%
from src.MyModule.utils import *
import pandas as pd
import numpy as np

config = load_config()
# input_path = get_path("data/processed/2_produce_data")
input_path = get_path("data/processed/no_bind")
output_path = get_path("data/processed/3_evaluate_data/")
figure_path = get_path("figures/")


#%%
class Variable:

    '''
    properties : data type and data with list
    functions : sort values and count values
    '''
    
    def __init__(self,
                 dtype : str,
                 data : list):

        self.dtype = dtype
        self.data = data
        if isinstance(self.data, pd.Series):
            self.data = self.data.tolist()

        # remove nan values
        self.remove_nan_values()

    def __repr__(self):
        print(self.data)
        return self.dtype

    def remove_nan_values(self):
        myarray = np.array(self.data)
        new_array = myarray[~pd.isnull(myarray)]
        self.data = new_array.tolist()

    def count_values(self):
        '''
        count values in a list and returns dictionary
        '''
        NotImplementedError

    def probability_distribution(self):
        '''
        creates probability distribution
        '''
        NotImplementedError

#%%
class ContinuousVariable(Variable):

    def __init__(self, data):
        super().__init__(dtype = 'continuous', data = data)
    
    def count_values(self, bins, range = None, density = False):
        """
        returns count information
        """
        counts, _ = np.histogram(self.data, bins, range = range, density = density)
        return counts

    def probability_distribution(self, bins, range = None):
        """
        returns distribution histogram series
        """
        counts = self.count_values(bins) 
        total = counts.sum()
        return pd.Series(data = counts / total)
#%%

class CategoricalVariable(Variable):

    def __init__(self, data):
        super().__init__(dtype="categorical", data= data)
        self.categories = set(self.data) 

    def count_values(self):
        counts = [self.data.count(element) for element in self.categories]
        return counts
    
    def probability_distribution(self):
        counts = self.count_values()
        data = np.array(counts) / sum(counts)
        return pd.Series(data = data, index=self.categories)


#%%
def hellinger_distance(var1 : Variable,
                       var2 : Variable,
                       bins = None):

    """
    calculates the hellinger distance of two variables
    dtype : categorical or continuous
    var1, var2 are variables
    bins : np.arange shape
    """
    assert var1.dtype == var2.dtype, "the two data type are not the same!"
    
    if var1.dtype == 'categorical' :
        var1prob, var2prob = var1.probability_distribution(), var2.probability_distribution()

        var1prob.name = "var1"
        var2prob.name = "var2"

        df = pd.merge(var1prob, var2prob, left_index=True, right_index=True, how = "outer")
        df = df.fillna(0)

        df['distance'] = df.apply(lambda x : (np.sqrt(x['var1']) - np.sqrt(x['var2']))**2, axis=1)

        return np.round((1/np.sqrt(2))*np.sqrt(df['distance'].sum()), 3)


    else :
        '''dtype is continuous'''
        assert bins is not None, "continous variables require bins"
        var1prob, var2prob = var1.probability_distribution(bins = bins), var2.probability_distribution(bins = bins)

        var1prob.name = "var1"
        var2prob.name = "var2"

        df = pd.merge(var1prob, var2prob, left_index=True, right_index=True, how = "outer")
        df = df.fillna(0.0)

        df['distance'] = df.apply(lambda x : (np.sqrt(x['var1']) - np.sqrt(x['var2']))**2, axis=1)

        return np.round((1/np.sqrt(2))*np.sqrt(df['distance'].sum()), 3)

#%% calculate hellinger distance between variables

#%% load original data and synthetic data

# load original

def load_pickle(path, cols=None) :
    df = pd.read_pickle(path) 

    if cols is None :
        return df
    
    return df[cols].copy()


#%% preprocessing


def get_variable_type(column_name, data) :
    datatype = data[column_name].dtype.__str__()

    if datatype in ['int64', 'float64'] :
        return 'continuous'
    
    return 'categorical'


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

def prepare_original_data(age) :

    original_data_path = get_path(f"data/processed/preprocess_1/encoded_D0_{age}.csv")
    data = pd.read_csv(original_data_path)

    bind_columns = pd.read_pickle(project_path.joinpath(f"data/processed/preprocess_1/bind_columns_{age}.pkl"))

    tables= []
    for col in bind_columns:
            tables.append('_'.join(col.split('_')[0:1]))
    try:
        data = data.drop('Unnamed: 0', axis=1)
    except:
        pass
    data = data.astype(str)

    for col in data.iloc[:,11:]:
        data[col] = data[col].str.replace('r','')
        
    decoded = decode(data.iloc[:,11:], tables, bind_columns)
    decoded.columns = bind_columns

    data.reset_index(drop=True, inplace=True)
    
    data = pd.concat([data.iloc[:,:11],decoded],axis=1)
    data = data.rename(columns = {'RLPS DIFF' : 'RLPS_DIFF'})
    data = data.drop(columns = "PT_SBST_NO")

    return data

def prepare_synthetic_data(age) :

    epsilons = config['epsilon']
    synthetic_data_list = []

    bind_columns = pd.read_pickle(project_path.joinpath(f"data/processed/preprocess_1/bind_columns_{age}.pkl"))

    tables= []
    for col in bind_columns:
            tables.append('_'.join(col.split('_')[0:1]))

    for epsilon in epsilons:

        syn = pd.read_csv(input_path.joinpath(f'S0_mult_encoded_{epsilon}_{age}.csv'))

        try:
            syn = syn.drop('Unnamed: 0', axis=1)
        except:
            pass
        syn = syn.astype(str)

        for col in syn.iloc[:,11:]:
            syn[col] =syn[col].str.replace('r','')
            
        decoded = decode(syn.iloc[:,11:], tables, bind_columns)
        decoded.columns = bind_columns
        
        syn = pd.concat([syn.iloc[:,:11],decoded],axis=1)
        syn = syn.rename(columns = {'RLPS DIFF' : 'RLPS_DIFF'})
        syn = syn.drop(columns = 'PT_SBST_NO')

        synthetic_data_list.append(syn)
    return synthetic_data_list

#%%
def calculate_hellinger_unbinded() :
    args = argument_parser()
    no_bind_path = project_path.joinpath("data/processed/no_bind")
    original = pd.read_csv(no_bind_path.joinpath('encoded_D0_to_syn_50.csv'))
    original = original.drop(columns = 'PT_SBST_NO')

    # synthetic_data_list = [pd.read_csv(no_bind_path.joinpath(f'seed0/S0_mult_encoded_{epsilon}_{args.age}.csv'))
    #                        for epsilon in config['epsilon']]
    synthetic_data_list = [pd.read_csv(no_bind_path.joinpath(f'seed0/S0_mult_encoded_{epsilon}_{50}.csv'))
                           for epsilon in config['epsilon']]
#%%
    def drop_column(data, columns):
        data = data.drop(columns = columns)
        return data

    synthetic_data_list = list(map(lambda x : drop_column(x, 'PT_SBST_NO'), synthetic_data_list))

    columns = original.columns.tolist()
    print(f'columns of original data : {len(columns)}')
    print(f'columns of synthetic data : {len(synthetic_data_list[0].columns)} ')

    hd_list = []
    for idx, epsilon in enumerate(config['epsilon']) :

        hd_dict = {}
        hd_dict['epsilon'] = epsilon
        synthetic = synthetic_data_list[idx]
    
        for col in columns :
            dtype = get_variable_type(col, original) 

            if col == "BSPT_STAG_VL" :
                ori = original[col].astype('float')
                syn = synthetic[col].astype('float')

            else : 
                ori = original[col]
                syn = synthetic[col]

            ori = CategoricalVariable(ori)
            syn = CategoricalVariable(syn)

            hd = hellinger_distance(ori, syn)

            hd_dict[col] = hd
        hd_list.append(hd_dict)
        #%%

    hd_info = pd.DataFrame(hd_list)
    hd_info.to_csv(output_path.joinpath('hd_info_all_columns_no_bind.csv'), index=False)
    print("successfully ended calculatign hellinger distance for all variables")
    return 0
#%%

#%%


def calculate_hellinger_for_all_variables() :
    args = argument_parser()
    
    synthetic_data_list = prepare_synthetic_data(args.age)
    original = prepare_original_data(args.age)
    columns = original.columns.tolist()

    hd_list = []
    for idx, epsilon in enumerate(config['epsilon']) :

        hd_dict = {}
        hd_dict['epsilon'] = epsilon
        synthetic = synthetic_data_list[idx]
    
        for col in columns :
            dtype = get_variable_type(col, original) 
            if col == "BSPT_STAG_VL" :
                ori = original[col].astype('float')
                syn = synthetic[col].astype('float')

            else : 
                ori = original[col]
                syn = synthetic[col]

            ori = CategoricalVariable(ori)
            syn = CategoricalVariable(syn)

            hd = hellinger_distance(ori, syn)

            hd_dict[col] = hd
        hd_list.append(hd_dict)

    hd_info = pd.DataFrame(hd_list)
    hd_info.to_csv(output_path.joinpath('hd_info_all_columns.csv'), index=False)
    print("successfully ended calculatign hellinger distance for all variables")
    return 0

#%%
def argument_parser() :
    parser = argparse.ArgumentParser()
    parser.add_argument('--age', default = 50, type = int)
    args = parser.parse_args()
    return args

def main() :
    args = argument_parser()
    columns = original.columns.tolist()
    age = args.age

    cols = [
        "BSPT_IDGN_AGE",
        "BSPT_SEX_CD",
        "BSPT_FRST_DIAG_NM",
        "SGPT_PATL_T_STAG_VL",
        "SGPT_PATL_N_STAG_VL",
        "SGPT_PATL_STAG_VL",
        "BSPT_STAG_VL",
        "MLPT_KRES_RSLT_NM",
        "IMPT_HM1E_RSLT_NM",
        "IMPT_HS2E_RSLT_NM",
        "IMPT_HS6E_RSLT_NM",
        "IMPT_HP2E_RSLT_NM",
        "DEAD",
        "OVRL_SURV",
        "LNE_CHEMO",
        "ADJ_CNT"
    ]

    bin_dict = {
        "BSPT_IDGN_AGE" : [0,20,30,40,50],
        "OVRL_SURV" : np.arange(0,365*10,1000),
    }

    original_path = get_path(f'data/processed/3_evaluate_data/matched_org_{age}.pkl')


    synthetic_data_path_list = []
    for epsilon in config['epsilon'] : 
        synthetic_path = get_path(f'data/processed/3_evaluate_data/matched_syn_{epsilon}_{age}.pkl')
        synthetic_data_path_list.append(synthetic_path)

    synthetic_data_list = list(map(lambda x : load_pickle(x, cols), synthetic_data_path_list))

    original = pd.read_pickle(original_path) 
    original = original[cols].copy()

    hd_list = []
    for idx, epsilon in enumerate(config['epsilon']) :

        hd_dict = {}
        hd_dict['epsilon'] = epsilon
        synthetic = synthetic_data_list[idx]

        for col in columns :

            dtype = get_variable_type(col, original)

            if col in bin_dict :
                ori = original[col]
                syn = synthetic[col]
                
                ori = ContinuousVariable(ori)
                syn = ContinuousVariable(syn)

                hd = hellinger_distance(ori, syn, bin_dict[col])

            else : 

                ori = original[col]
                syn = synthetic[col]
                
                ori = CategoricalVariable(ori)
                syn = CategoricalVariable(syn)

                hd = hellinger_distance(ori, syn)

            hd_dict[col] = hd
        hd_list.append(hd_dict)

    hd_info = pd.DataFrame(hd_list)
    hd_info.to_csv(output_path.joinpath('hd_info.csv'), index=False)

#%%
if __name__ == "__main__" :
    # main()
    # calculate_hellinger_for_all_variables()
    calculate_hellinger_unbinded()

#%%
