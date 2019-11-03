config = {
        'data_type': 'normal',
        'exp_order': ['micro'],
        'nsga_strategy': 'micro',
        'INPUT_WIDTH': 1,
        'problem': 'regression'
        }


def set_config(key, value):
    global config
    config[key] = value


def config_dict():
    global config
    return config