import json


def log_config(config_class):
    ret = {k: v for k, v in vars(config_class).items() if not k.startswith('__')}
    return json.dumps(ret, indent=4)
