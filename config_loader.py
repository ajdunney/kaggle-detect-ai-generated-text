import json
import yaml
import os


class ProjectConfig():
    def __init__(self, config_path='proj_config.yaml'):
        with open(config_path, 'r') as file:
            self.config = yaml.safe_load(file)

    def get_config(self, section_name: str) -> dict:
        return self.config[section_name]

    def get_data_config(self) -> dict:
        return self.config['data']
    
    def get_training_config(self) -> dict:
        return self.config['training']

    def get_kaggle_config(self) -> dict:
        return self.config['kaggle']

    def set_kaggle_credentials(self) -> None:
        kaggle_config = self.get_kaggle_config()
        creds = ProjectConfig.load_kaggle_credentials(
            kaggle_config['credentials_path']
            )
        os.environ["KAGGLE_USERNAME"] = creds['username']
        os.environ["KAGGLE_KEY"] = creds['key']

    @staticmethod
    def load_kaggle_credentials(kaggle_json_path) -> dict:
        with open(kaggle_json_path, 'r') as file:
            credentials = json.load(file)
        return credentials
