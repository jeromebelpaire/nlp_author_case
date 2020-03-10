import yaml
import os
import sys

class Config:
    def __init__(self):
        self.proj_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        self.config_dir = os.path.join(self.proj_dir, 'config/')
        self.config = yaml.safe_load(open('{}config.yml'.format(self.config_dir)))
        
        #Get paths
        self.raw_data_path = os.path.join(self.proj_dir, self.config['paths']['data']['raw_path'])
        self.interim_data_path = os.path.join(self.proj_dir, self.config['paths']['data']['interim_path'])
        self.processed_data_path = os.path.join(self.proj_dir, self.config['paths']['data']['processed_path'])
        self.temp_data_path = os.path.join(self.proj_dir, self.config['paths']['data']['temp_path'])
        self.models_path = os.path.join(self.proj_dir, self.config['paths']['models_path'])

        #Get Api settings
        self.api_address = self.config['api']['address']
        self.api_port = self.config['api']['port']

    def get_raw_data_path(self):
        return self.raw_data_path

    def get_interim_data_path(self):
        return self.interim_data_path        

    def get_processed_data_path(self):
        return self.processed_data_path

    def get_temp_data_path(self):
        return self.temp_data_path
        
    def get_models_path(self):
        return self.models_path

    def get_api_address(self):
        return self.api_address

    def get_api_port(self):
        return self.api_port