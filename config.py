config = {
        'data_type': 'normal',
        'INPUT_WIDTH': 1,
        'problem': 'classification',
        'performance_measure': 'acc'
        }


def set_config(key, value):
    global config
    config[key] = value


def config_dict():
    global config
    return config