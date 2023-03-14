#%%
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from functools import reduce
import yaml
import pandas as pd
import random

from src.MyModule.utils import *

#%%

class MapColumns :

    def __init__(self, mappings_path) :
        '''
        mappings : dictionary that maps the columns to specific column type
        e.g. 'numeric' : ['hemoglobin'] , 'discrete' : ['gender']
        '''
        self.mappings_path = mappings_path
        self.load_mapping()

    def load_mapping(self) :
        with open(self.mappings_path, 'rb') as f :
            self.mappings = yaml.load(f, yaml.SafeLoader)
        self.index = self.mappings['index']
        self.standard_date = self.mappings['standard_date']
        self.numeric = self.mappings['numeric']
        self.discrete_numeric = self.mappings['discrete_numeric']
        self.discrete = self.mappings['discrete']
        self.date = self.mappings['date']

        self.no_maps = []
        for keys, val in self.mappings.items() :
            if self.mappings[keys] is None  :
                self.no_maps.append(keys)


class ProcessNumeric :

    def __init__(self, column) :
        self.column = column

        if not isinstance(self.column, list) :
            self.column = [self.column]

        self.scaler = StandardScaler()

    def process_numeric(self, data) :
        data = data.copy()
        col = self.column
        self.scaler.fit(data[col])
        data.loc[:,self.column] = self.scaler.transform(data[col])
        data = self.make_na_indicator_column(data)
        data = self.fillna(data)
        return data 

    def make_na_indicator_column(self, data) : 
        data = data.copy()
        for col in self.column : 
            data[col + '_naindicator'] = data[col].isna() * 1
        return data

    def fillna(self, data) :
        data = data.copy()
        for col in self.column :
            meanval = data[col].mean()
            data[col] = data[col].fillna(meanval)
        return data

    def reverse_column(self, data) :
        data = data.copy()
        for col in self.column : 
            tempdf = data.filter(like = col).copy()
            indicator = ~(tempdf[col + '_naindicator'] > 0.5)
            result = tempdf[col].mul(indicator)

            data.loc[:, col] = result
            data.drop(columns = col + '_naindicator', inplace = True)

        data.loc[:, self.column] = self.scaler.inverse_transform(data[self.column])
        return data

class ProcessDiscreteNumeric : 

    def __init__(self, column) :
        self.column = column

        if not isinstance(self.column, list) :
            self.column = [self.column]

    def process_discretenumeric(self, data) :
        data = data.copy()
        self.maxval_list = []
        self.minval_list = []
        for col in self.column :
            maxval = data[col].max()
            minval = data[col].min()
            
            data[col] = (data[col] - minval) / (maxval - minval)

            self.maxval_list.append(maxval)
            self.minval_list.append(minval)

        data = self.make_na_indicator_column(data)
        data = self.fillna(data)

        return data

    def make_na_indicator_column(self, data) : 
        data = data.copy()
        for col in self.column : 
            data[col + '_naindicator'] = data[col].isna() * 1
        return data
    
    def fillna(self, data) :
        data = data.copy()
        for col in self.column : 
            meanval = data[col].mean()
            if np.isnan(meanval) :
                meanval = 0
            data[col] = data[col].fillna(meanval)
        return data


    def reverse_column(self, data) :
        data = data.copy()
        for col in self.column : 
            tempdf = data.filter(like = col).copy()
            indicator = ~(tempdf[col + '_naindicator'] > 0.5)
            result = tempdf[col].mul(indicator)

            data.loc[:, col] = result
            data.drop(columns = col + '_naindicator', inplace = True)

        for idx, col in enumerate(self.column) :
            maxval = self.maxval_list[idx]
            minval = self.minval_list[idx]
            data[col] = data[col] * (maxval - minval) + minval
            data[col] = data[col].round()

        return data

class ProcessDiscrete : 

    def __init__(self, column) :
        self.column = column

        if not isinstance(self.column, list) :
            self.column = [self.column]

    def process_discrete(self, data) :
        data = data.copy()
        data = self.fillna(data)
        dummified = pd.get_dummies(data[self.column])
        data = data.drop(columns = self.column)
        dummified_data = pd.concat([data, dummified], axis = 1)
        return dummified_data 
            

    def fillna(self, data) :
        data = data.copy()
        for col in self.column : 
            data[col] = data[col].fillna('nan')
        return data

    def reverse_column(self, data) :
        data = data.copy()
        for col in self.column : 
            tempd = data.filter(like = col).copy()
            argmax_list = tempd.values.argmax(axis = 1).tolist()

            value_dict = {}
            for idx, tempd_col in enumerate(tempd.columns.tolist()) :
                val = tempd_col.replace(col + '_', '')
                value_dict[idx] = val

            data = data.drop(columns = tempd.columns.tolist())
            data[col] = argmax_list
            data[col] = data[col].replace(value_dict)

        return data

            
class ProcessDate :

    def __init__(self, column, standard_date_column) :
        self.column = column
        self.standard_date_column = standard_date_column

        if not isinstance(self.column, list) :
            self.column = [self.column]

    def process_date(self, data) :
        data = data.copy()

        try :
            for col in self.column :
                data.loc[:, col] = pd.to_datetime(data[col], format= "%Y%m%d")
        except :
            print("the columns does not match the format!")
        
        index = self.column.index(self.standard_date_column)
        column = self.column.copy()

        standard_date_column = column.pop(index)

        for col in column :
            data.loc[:,col] = (data[col] - data[standard_date_column]).dt.days

        data.loc[:, standard_date_column] = 0
        return data

    def reverse_column(self, data) :
        data = data.copy()

        data[self.standard_date_column] = 20000101
        data.loc[:, self.standard_date_column] = pd.to_datetime(data[self.standard_date_column], format = '%Y%m%d')

        for col in self.column : 
            if col == self.standard_date_column :
                continue
            data.loc[:, col] = pd.to_timedelta(data[col],'days')
            data.loc[:, col] = data.apply(lambda x : x[col] + x[self.standard_date_column], axis=1)
            # data.loc[:, col] = pd.to_timedelta(data[col], 'days') + data[self.standard_date_column]

        return data


# preprocessor
class ProcessStatic :

    def __init__(self, 
                 static_data,
                 mapping,
                 ) :

        '''
        static_data : original static data
        mapping : path to the mapping file 
        '''
        self.static_data = static_data.copy()
        self.mapping_path = mapping
        self.mappings = MapColumns(mapping)

        # date will be translated as numeric at first
        self.numeric_processor = ProcessNumeric(self.mappings.numeric)
        self.discrete_numeric_processor = ProcessDiscreteNumeric(self.mappings.discrete_numeric +
                                                                 self.mappings.date)
        self.discrete_processor = ProcessDiscrete(self.mappings.discrete)
        self.date_processor = ProcessDate(self.mappings.date, self.mappings.standard_date)


    def preprocess(self) :
        # process dates first

        data = self.process_date(self.static_data)
        data = self.process_numeric(data)
        data = self.process_discretenumeric(data)
        data = self.process_discrete(data)

        columns = data.columns.tolist()
        columns.remove(self.mappings.index)
        columns = sorted(columns)
        ordered_column = [self.mappings.index] + columns

        return data[ordered_column].copy()

    def reverse(self, data) :
        data = data.copy()
        data = self.reverse_discrete(data)
        data = self.reverse_discretenumeric(data)
        data = self.reverse_numeric(data)
        data = self.reverse_date(data)

        data = self.process_column_order(data)

        return data
    
    def process_column_order(self, data) :
        column_order = self.static_data.columns.tolist()
        data = data[column_order].copy()

        return data

    def process_numeric(self, data) :
        data = data.copy()
        if 'numeric' in self.mappings.no_maps :
            return data
        return self.numeric_processor.process_numeric(data)

    def reverse_numeric(self, data) :
        data = data.copy()
        if 'numeric' in self.mappings.no_maps :
            return data
        return self.numeric_processor.reverse_column(data)

    def process_date(self, data) :
        data = data.copy()
        if 'date' in self.mappings.no_maps :
            return data
        return self.date_processor.process_date(data)

    def reverse_date(self, data) :
        data = data.copy()
        if 'date' in self.mappings.no_maps :
            return data
        return self.date_processor.reverse_column(data)

    def process_discretenumeric(self, data) :
        data = data.copy()
        if 'discrete_numeric' in self.mappings.no_maps :
            return data
        return self.discrete_numeric_processor.process_discretenumeric(data)

    def reverse_discretenumeric(self, data) :
        data = data.copy()
        if 'discrete_numeric' in self.mappings.no_maps :
            return data
        return self.discrete_numeric_processor.reverse_column(data)

    def process_discrete(self, data) :
        data = data.copy()
        if 'discrete' in self.mappings.no_maps :
            return data
        return self.discrete_processor.process_discrete(data)
        
    def reverse_discrete(self, data) :
        data = data.copy()
        if 'discrete' in self.mappings.no_maps :
            return data
        return self.discrete_processor.reverse_column(data)


from torch.utils.data import Dataset, DataLoader
import torch

class StaticDataset(Dataset) :

    def __init__(self, data_path, file_name, mapping_path) :
        self.data_path = data_path
        self.file_name = file_name
        self.mapping_path = mapping_path

        self.cancerlibrary = ProcessCancerLibrary(data_path, file_name, mapping_path)
        self.df = self.cancerlibrary.return_required()

        self.processor = ProcessStatic(self.df, mapping_path)

        processed_df = self.processor.preprocess()
        self.processed_df_columns = processed_df.columns.tolist()
        
        processed_df = processed_df.drop(columns = self.processor.mappings.index)
        self.processed_dataset = torch.Tensor(processed_df.values)
    
    def __len__(self) :
        return len(self.processed_dataset)

    def __getitem__(self, index) :
        return self.processed_dataset[index]

    def get_processor(self) :
        return self.processor

class ProcessCancerLibrary :

    def __init__(self, path, file_name, mapping_path) :
        self.path = path
        self.file_name = file_name
        self.mapping_path = mapping_path
        self.original_file = read_file(path, file_name)

        self.mapping = MapColumns(mapping_path)

        self.required_columns = []

        for key, val in self.mapping.mappings.items() :

            if key in self.mapping.no_maps :
                continue

            if isinstance(val, list) :
                self.required_columns = self.required_columns + val
            else :
                self.required_columns.append(val)

    def return_required(self) :
        filterd_columns = []
        for col in self.original_file.columns.tolist() :
            if col in self.required_columns :
                filterd_columns.append(col)

        required_data = self.original_file[filterd_columns]
        return required_data

def get_static_dataloader(data_path, file_name, mapping_path, batch_size, shuffle, num_workers = 4) :
    dataset = StaticDataset(data_path, file_name, mapping_path)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)


####################
####################     Dynmaic data part !!!!
####################

class MapDynamicColumns :

    def __init__(self, mappings_path) :
        '''
        mappings : dictionary that maps the columns to specific column type
        e.g. 'numeric' : ['hemoglobin'] , 'discrete' : ['gender']
        '''
        self.mappings_path = mappings_path
        self.load_mapping()

    def load_mapping(self) :
        with open(self.mappings_path, 'rb') as f :
            self.mappings = yaml.load(f, yaml.SafeLoader)

        self.index = self.mappings['index']
        self.index_time = self.mappings['index_time']
        self.standard_date = self.mappings['standard_date']

        self.numeric = self.mappings['numeric']
        self.discrete_numeric = self.mappings['discrete_numeric']
        self.discrete = self.mappings['discrete']
        self.date = self.mappings['date']

        self.no_maps = []
        for keys, val in self.mappings.items() :
            if self.mappings[keys] is None  :
                self.no_maps.append(keys)

class ProcessNumerictoDiscrete :

    def __init__(self, column) :
        self.column = column

        if not isinstance(self.column, list) :
            self.column = [self.column]
        self.scaler = StandardScaler()

    def process_numeric(self, data) :
        data = data.copy()
        col = self.column
        self.scaler.fit(data[col])
        data.loc[:,self.column] = self.scaler.transform(data[col])
        data = self.transform_numeric_to_discrete(data)
        data = self.fillna(data)
        # changed to discrete column
        # now process the discrete column

        dummified = pd.get_dummies(data[self.column])
        data = data.drop(columns = self.column)
        dummified_data = pd.concat([data, dummified], axis = 1)
        return dummified_data 

    def transform_numeric_to_discrete(self, data) :
        data = data.copy()
        self.range = []
        self.label = []
        for idx, col in enumerate(self.column) :
            maxval = data[col].max()
            minval = data[col].min()
            meanval = data[col].mean()
            stdval = data[col].std()
            
            self.range.append([minval, meanval-2*stdval, meanval-1*stdval, meanval, 
                               meanval + 1*stdval, meanval + 2*stdval, maxval])

            self.label.append([0, 1, 2, 3, 4, 5])

            data[col] = pd.cut(data[col], self.range[idx], labels=self.label[idx], include_lowest=True)
        return data

    def fillna(self, data) :
        data = data.copy()
        for col in self.column :
            data[col] = data[col].astype('object').fillna('nan')
        return data

    def reverse_column(self, data) :
        data = data.copy()

        for idx, col in enumerate(self.column) : 
            tempd = data.filter(like = col).copy()
            argmax_list = tempd.values.argmax(axis = 1).tolist()

            value_dict = {}
            for idx, tempd_col in enumerate(tempd.columns.tolist()) :
                val = tempd_col.replace(col + '_', '')
                if val != 'nan' :
                    val = int(val)
                value_dict[idx] = val

            data = data.drop(columns = tempd.columns.tolist())
            data[col] = argmax_list
            data[col] = data[col].replace(value_dict)
            data[col] = data[col].replace('nan', np.nan)

            data[col] = data[col].apply(lambda x : self.randomly_reverse(idx, x))
       
        return data

    def randomly_reverse(self, idx, x) :
        if np.isnan(x) :
            return np.nan
        else :
            return random.uniform(self.range[idx][x], self.range[idx][x+1])

class ProcessNumericDiscretetoDiscrete :

    def __init__(self, columns) :
        self.columns = columns

        if not isinstance(self.columns, list) :
            self.columns = [self.columns]

    def process_discretenumeric(self, data) :
        data = data.copy()
        self.maxval_list = []
        self.minval_list = []

        for col in self.columns :
            maxval = data[col].max()
            minval = data[col].min()
            
            data[col] = (data[col] - minval) / (maxval - minval)

            self.maxval_list.append(maxval)
            self.minval_list.append(minval)

        data = self.transform_numeric_to_discrete(data)
        data = self.fillna(data)
        # changed to discrete column
        # now process the discrete column
        if len(self.columns) < 2 : 
            dummified = pd.get_dummies(data[self.columns]).add_prefix(self.columns[0] + '_')

        else :
            dummified = pd.get_dummies(data[self.columns])

        data = data.drop(columns = self.columns)
        dummified_data = pd.concat([data, dummified], axis = 1)

        return dummified_data

    def transform_numeric_to_discrete(self, data) :
        data = data.copy()
        self.range = []
        self.label = []
        for idx, col in enumerate(self.columns) :
            maxval = data[col].max()
            minval = data[col].min()
            meanval = data[col].mean()
            stdval = data[col].std()
            self.range.append([minval, meanval-2*stdval, meanval-1*stdval, meanval, 
                               meanval + 1*stdval, meanval + 2*stdval, maxval])

            self.label.append([0, 1, 2, 3, 4, 5])

            data[col] = pd.cut(data[col], self.range[idx], labels=self.label[idx], include_lowest=True)
        return data

    def fillna(self, data) :
        data = data.copy()
        for col in self.columns :
            data[col] = data[col].astype('object').fillna('nan')
        return data

    def reverse_column(self, data) :
        data = data.copy()
        for idx, col in enumerate(self.columns) : 
            tempd = data.filter(like = col).copy()
            argmax_list = tempd.values.argmax(axis = 1).tolist()

            value_dict = {}
            for jdx, tempd_col in enumerate(tempd.columns.tolist()) :
                val = tempd_col.replace(col + '_', '')
                value_dict[jdx] = val

            data = data.drop(columns = tempd.columns.tolist())
            data[col] = argmax_list
            data[col] = data[col].replace(value_dict)
            data[col] = data[col].apply(lambda x : self.randomly_reverse(idx, x))
            data[col] = data[col].replace('nan', np.nan)

            maxval = self.maxval_list[idx]
            minval = self.minval_list[idx]
            data[col] = data[col] * (maxval - minval) + minval
            data[col] = data[col].round()

        return data

    def randomly_reverse(self, idx, x) :
        if x == 'nan' :
            return 'nan'
        else :
            x = int(x)
            return random.uniform(self.range[idx][x], self.range[idx][x+1])

#%%

class ProcessDynamicDate :

    def __init__(self, column, standard_date_column, index_column) :
        self.column = column
        self.standard_date_column = standard_date_column
        self.index_column = index_column

        if not isinstance(self.column, list) :
            self.column = [self.column]

    def process_date(self, data) :
        data = data.copy()

        try :
            for col in self.column :
                data.loc[:, col] = pd.to_datetime(data[col], format= "%Y%m%d")
                
        except :
            print("the columns does not match the format!")
        
        index = self.column.index(self.standard_date_column)
        column = self.column.copy()

        standard_date_column = column.pop(index)

        for col in column :
            data.loc[:,col] = (data[col] - data[standard_date_column]).dt.days

        data = data.drop(columns = standard_date_column)

        return data

    def reverse_column(self, data, standard_date_column = None, is_synthetic = False) :
        data = data.copy()

        if is_synthetic : 
            
            data[self.standard_date_column] = "20000101"
            data.loc[:, self.standard_date_column] = pd.to_datetime(data[self.standard_date_column], format = '%Y%m%d')

            for col in self.column : 
                if col == self.standard_date_column :
                    continue

                data.loc[:, col] = pd.to_timedelta(data[col],'days')
                data.loc[:, col] = data.apply(lambda x : x[col] + x[self.standard_date_column], axis=1)
                data = data.dropna(subset = self.index_column)

            return data

        else :
            data = data.merge(standard_date_column)

            data.loc[:, self.standard_date_column] = pd.to_datetime(data[self.standard_date_column], 
                                                                    format = '%Y%m%d')

            for col in self.column : 
                if col == self.standard_date_column :
                    continue
                data.loc[:, col] = pd.to_timedelta(data[col],'days')
                data.loc[:, col] = data.apply(lambda x : x[col] + x[self.standard_date_column], axis=1)
                data = data.dropna(subset = self.index_column)

            return data



#%%

class DynamicProcess :

    def __init__(self, dynamic_data, mappings) :
        self.dynamic_data = dynamic_data
        self.mappings = MapDynamicColumns(mappings)
        self.numerictodiscrete_processor = ProcessNumerictoDiscrete(self.mappings.numeric)

        if 'discrete_numeric' not in self.mappings.no_maps :
            self.discretenumerictodiscrete_processor = ProcessNumericDiscretetoDiscrete(self.mappings.discrete_numeric + 
                                                                                      self.mappings.index_time)
            self.mappings.no_maps.remove('discrete_numeric')
        else :
            self.discretenumerictodiscrete_processor = ProcessNumericDiscretetoDiscrete(self.mappings.index_time)
            self.mappings.no_maps.remove('discrete_numeric')

        self.discrete_processor = ProcessDiscrete(self.mappings.discrete)
        self.date_processor = ProcessDynamicDate(self.mappings.date, 
                                                 self.mappings.standard_date, 
                                                 self.mappings.index_time)

    def preprocess(self) :
        # process dates first
        data = self.process_date(self.dynamic_data)
        data = self.process_numeric(data)
        data = self.process_discretenumeric(data)
        data = self.process_discrete(data)

        columns = data.columns.tolist()
        columns.remove(self.mappings.index)

        columns = sorted(columns)
        ordered_column = [self.mappings.index] + columns

        return data[ordered_column].copy()

    def reverse(self, data, is_synthetic=True) :
        data = data.copy()
        data = self.reverse_discrete(data)
        data = self.reverse_discretenumeric(data)
        data = self.reverse_numeric(data)
        data = self.reverse_date(data, is_synthetic)
        #
        data = self.process_column_order(data)

        return data
    
    def process_column_order(self, data) :
        column_order = self.dynamic_data.columns.tolist()
        data = data[column_order].copy()

        return data

    def process_numeric(self, data) :
        data = data.copy()
        if 'numeric' in self.mappings.no_maps :
            return data
        return self.numerictodiscrete_processor.process_numeric(data)

    def reverse_numeric(self, data) :
        data = data.copy()
        if 'numeric' in self.mappings.no_maps :
            return data
        return self.numerictodiscrete_processor.reverse_column(data)

    def process_date(self, data) :
        data = data.copy()
        if 'date' in self.mappings.no_maps :
            return data
        self.standard_date_info = data[[self.mappings.index, self.mappings.standard_date]].copy()
        return self.date_processor.process_date(data)

    def reverse_date(self, data, is_synthetic) :
        data = data.copy()
        if 'date' in self.mappings.no_maps :
            return data

        if is_synthetic :
            return self.date_processor.reverse_column(data, standard_date_column=None, is_synthetic=True)

        else :
            return self.date_processor.reverse_column(data, standard_date_column=self.standard_date_info, 
                                                  is_synthetic=False)

    def process_discretenumeric(self, data) :
        data = data.copy()
        if 'discrete_numeric' in self.mappings.no_maps :
            return data
        return self.discretenumerictodiscrete_processor.process_discretenumeric(data)

    def reverse_discretenumeric(self, data) :
        data = data.copy()
        if 'discrete_numeric' in self.mappings.no_maps :
            return data
        return self.discretenumerictodiscrete_processor.reverse_column(data)

    def process_discrete(self, data) :
        data = data.copy()
        if 'discrete' in self.mappings.no_maps :
            return data
        return self.discrete_processor.process_discrete(data)
        
    def reverse_discrete(self, data) :
        data = data.copy()
        if 'discrete' in self.mappings.no_maps :
            return data
        return self.discrete_processor.reverse_column(data)

#%%
path = Path('/home/wonseok/projects/2022_DATA_SYNTHESIS/data/raw/')
pt_bsnf = read_file(path, 'LUNG_PT_BSNF.xlsx')
dead_mapping_path = '/home/wonseok/projects/2022_DATA_SYNTHESIS/mapping/lung/DEAD_NFRM.yaml'
dead_nfrm = read_file(path, 'LUNG_DEAD_NFRM.xlsx')

#%%
dead_nfrm = pt_bsnf[['PT_SBST_NO','BSPT_BRYM']].merge(dead_nfrm)

#%%
dead_nfrm = dead_nfrm.drop(columns = ['CENTER_CD', 'IRB_APRV_NO', 'CRTN_DT'])

#%%
dead_nfrm = pt_bsnf[['PT_SBST_NO', 'BSPT_BRYM']].merge(dead_nfrm, how='left')

#%%
dp = DynamicProcess(dead_nfrm, dead_mapping_path)


#%%
a  = dp.preprocess()
a
#%%

dp.reverse(a, is_synthetic=True)

#%%
dead_nfrm.columns

#%%
class DynamicDataset(Dataset) :

    def __init__(self, 
                 static_path, static_file_name, static_mapping,
                 dynamic_path, dynamic_file_name, dynamic_mapping) : 

        self.staticdataset = StaticDataset(static_path, static_file_name, static_mapping)
        self.dynamic_data = read_file(dynamic_path, dynamic_file_name)

        self.dynamic_processor = DynamicProcess(dynamic_data, dynamic_mapping)

        pass
    
    def preprocess_data(self) :
        dynamic = self.dynamic_processor.preprocess()
        


    def __len__(self) :

        pass

    def __getitem__(self, index) :

        pass

#%%

class DynamicData : 

    def __init__(self) :
        











