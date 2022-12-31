import argparse
import os
import yaml

def parse_args():
    parser = argparse.ArgumentParser(description='Train model')
    
    parser.add_argument("--config", type=str, nargs="*", default=None, help="If specified, a config file with the parameters will be used instead of the arguments given via the command line")

    args = parser.parse_args()
    
    return args

def parse_dict(res, dictionary):
    if dictionary is None:
        return
    
    for key, val in dictionary.items():
        if isinstance(val, dict):
            if key in res:
                parse_dict(res[key], val)
            else:
                res[key] = val
        else:
            res[key] = val

def overwrite_with_config_args(config):
    # Load config from yaml if available
    res = {'config': config}
    if config:
        for config_path in config:
            with open(os.getcwd() + '/config/' + config_path + '.yaml', 'r') as stream:
                parsed_yaml = yaml.safe_load(stream)
                parse_dict(res, parsed_yaml)  
    
    return res