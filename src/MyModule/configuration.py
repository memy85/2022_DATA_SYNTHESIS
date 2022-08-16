import yaml
from pathlib import Path
import pickle

class Configurator:
    '''
    configuration class that consists information.
    '''
    
    def __init__(self, config_path: dict):
        with open(config_path) as f:
            self.config = yaml.load(f, yaml.SafeLoader)
            
            self.input_path = Path(self.config['path_config']['input_path'])
            self.output_path = Path(self.config['path_config']['output_path'])
            self.epsilon = self.config['epsilon']
            self.random_seed = self.config['random_seed']
    
    def return_required_variables(self, table_name):
        '''
        returns required variables list
        '''
        return self.config['data_config']['required'][table_name]
    
    def return_derivative_variables(self, table_name):
        '''
        returns the parent, child list for derivative variables for the particular data
        '''
        return self.config['derivative_columns'][table_name] 
    
    def return_institution_specific_variables(self):
        '''
        returns institution specific variables
        '''
        return self.config['institution_specific_columns']
    
    def create_medium_output_folder(self, output_type):
        '''
        helper function to make folders for processes
        '''
        path = self.ouput_path.joinpath(output_type)
        if not path.exists():
            path.mkdir(parents=True)
    
    def save_data(self, output_type : str, data_name : str, object):
        '''
        helper function that saves input object(mostly data itself)
        output_type : 중간산물 type
        data_name : data name
        object : the python data object
        '''
        self.create_medium_output_folder(output_type=output_type)
        data_name = data_name + '.pkl'
        with open(self.output_path.joinpath(output_type, data_name),'wb') as f:
            pickle.dump(f, object)
    
    def load_data(self, output_type : str, data_name : str):
        with open(self.output_path.joinpath(output_type, data_name),'rb') as f:
            return pickle.load(f)
