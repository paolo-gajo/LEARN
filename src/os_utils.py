from datetime import datetime
import argparse
import json

def get_time():
    return str(datetime.now()).split('.')[0].replace(' ', '').replace('-', '').replace(':', '')[2:]

def setup_config(namespace: argparse.Namespace, default_cfg: dict):
    args = vars(namespace)
    config = default_cfg
    for k, v in args.items():
        config[k] = v
    if not config['suffix']:
        config['suffix'] = get_time()
    config['tag_dict'] = json.load(open(config['tag_dict_path'], 'r'))
    return config
