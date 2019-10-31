config = {
        'INPUT_HEIGHT': 1125,
        'INPUT_WIDTH': 1,
        'dataset': 'BCI_IV_2a',
        'data_type': 'normal',
        'nsga_strategy': 'macro',
        'arg_string': '--search_space micro --init_channels 24 --n_gens 30',
        'custom_cell_for_macro': True
        }


def set_config(key, value):
    global config
    config[key] = value


def config_dict():
    global config
    return config