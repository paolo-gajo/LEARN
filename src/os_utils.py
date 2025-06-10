from datetime import datetime
import argparse

def get_time():
    return str(datetime.now()).split('.')[0].replace(' ', '').replace('-', '').replace(':', '')[2:]

def setup_config(namespace: argparse.Namespace, default_cfg: dict):
    args = vars(namespace)
    config = default_cfg
    for k, v in args.items():
        config[k] = v
    return config
