from MyModule.configuration import Configurator

from pathlib import Path
import pandas as pd
import pickle

class Data:
    '''
    class that has the original and can have the synthetic
    '''
    def __init__(self, table_name:str, configurator: Configurator):
        '''
        data : the original data
        table_name : name of the original data
        configurator : configurator object
        '''
        self.table_name = str.upper(table_name)
        self.configurator = configurator
        self.data = self.load_data()
        self.convert_dates()
        
        self.original_columns = self.data.columns.tolist()
        self.required_columns = self.configurator.config['data_config']['required'][self.table_name]
    
    def load_data(self):
        if not self.configurator.input_path.joinpath(self.table_name.upper()+'.xlsx').exists() :
            return pd.read_csv(self.configurator.input_path.joinpath(self.table_name.upper()+'.csv'))
        else :
            return pd.read_excel(self.configurator.input_path.joinpath(self.table_name.upper()+'.xlsx'))
        # convert dates 
        
    def convert_dates(self):
        '''
        convert dates that are in object format to pandas dates
        '''
        def give_ymd_data(required_columns:dict):
            '''
            filter ymd data
            '''
            return [k for k, v in self.configurator.config['data_config']['required'][self.table_name].items() if v == 'datetime64[ns]']
        date_cols = give_ymd_data(list(self.required_columns.keys()))  
        for col in date_cols:
            self.data[col] = pd.to_datetime(self.data[col], format='%Y%m%d')
    
    def return_synthesizable(self):
        NotImplemented

    def return_data_with_required_variables(self):
        '''
        첫번째 필요변수만 추려진 데이터 반환. 아직 timeseries를 index화 하기 전.
        '''
        required_columns = list(self.required_columns.keys())
        self.required_data = self.data[required_columns]
        # save the intermediate data
        self.configurator.save_data('required_variables',self.table_name, self.required_data)
        return self.required_data
        
    def convert_required_data2(self):
        NotImplementedError    
