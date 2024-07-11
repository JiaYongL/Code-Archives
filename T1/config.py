import yaml

# This file is for parsing configs generated from config_generator.py.
# No need to run this file.

class Config:
    def __init__(self, config_file):
        config_dict = yaml.safe_load(open(config_file, 'r'))
        for key, value in config_dict.items():
            if isinstance(value, dict):
                setattr(self, key, Config(value))
            else:
                setattr(self, key, value)