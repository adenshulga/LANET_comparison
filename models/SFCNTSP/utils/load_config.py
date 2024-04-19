import os
import json
import torch
import itertools

from utils.load_config import config

import numpy as np
import pickle


def get_attribute(attribute_name: str, default_value=None):
    """
    get configs
    :param attribute_name: config key
    :param default_value: None
    :return:
    """
    try:
        return getattr(config, attribute_name)
    except KeyError:
        return default_value


config.data_path = config.data

# dataset specified settings
# config.update(config[f"{get_attribute('dataset_name')}"])
# config.pop('JingDong')
# config.pop('DC')
# config.pop('TaoBao')
# config.pop('TMS')
# config.pop('JingDong_inductive')
# config.pop('DC_inductive')
# config.pop('TaoBao_inductive')
# config.pop('TMS_inductive')
# config.pop('mimic3_preprocessed')

def unpickle_file(path, type_of_split, data, prefix, encoding):
    tmp_path = os.path.join(path, prefix + type_of_split + '.pkl')
    with open(tmp_path, 'rb') as file:
        data[type_of_split] = pickle.load(file, encoding=encoding)[type_of_split]
    return data

def retrieve_dict(path, prefix='', encoding='ASCII'):
    tmp_path = os.path.join(path, prefix + 'dev.pkl')
    # print(encoding)
    with open(tmp_path, 'rb') as file:
        data = pickle.load(file, encoding=encoding)
    unpickle_file(path, 'train', data, prefix, encoding=encoding)
    unpickle_file(path, 'test', data, prefix, encoding=encoding)
    return data

def from_one_hot(one_hot_encoded_basket: list) -> list:
    return np.nonzero(one_hot_encoded_basket)[0].tolist()

def convert_dataset_from_pickle(unpickled_data) -> None:
    converted_data = {'train': [], 'validate': [], 'test': []}  # Assuming you might have 'validate' and 'test' sets as well
    
    # Convert each dataset section (train, validate, test)
    help_dict = {'train' : 'train', 'dev' : 'validate', 'test' : 'test'}
    for section in help_dict:
        for user_id, baskets in enumerate(unpickled_data[section]):
            user_baskets = []

            # if isinstance(user_id, str):
            #     if user_id.startswith('user'):
            #         user_id = int(user_id[4:])
            #     else:
            #         user_id = int(user_id)
                

            set_time = 1  # Initialize a time counter; you might have real timestamps to use instead
            for basket in baskets:
                if ~(np.all(basket['type_event'] == 0)):
                    user_baskets.append({
                        "user_id": user_id,  # Convert user ID to an integer
                        "items_id": from_one_hot(basket['type_event']),       # The list of item IDs in the basket
                        "set_time": basket['time_since_start']      # The time the basket was 'purchased'
                    })

            # Add the converted user baskets to the respective section of the dataset
            converted_data[help_dict[section]].append(user_baskets)

    return converted_data

def get_users_items_num_and_max_seq_length(data_path):
            

    data = retrieve_dict(data_path)
    data_dict = convert_dataset_from_pickle(data)

    max_seq_length = -1
    # get users and items num
    user_ids_set, item_ids_set = set(), set()
    for data_type in data_dict:
        for user_sets in data_dict[data_type]:
            user_ids_set = user_ids_set.union({user_sets[0]['user_id']})
            item_ids_set = item_ids_set.union(set(itertools.chain.from_iterable([user_set['items_id'] for user_set in user_sets])))

            if len(user_sets) - 1 > max_seq_length:
                max_seq_length = len(user_sets) - 1

    num_users, num_items = len(user_ids_set), len(item_ids_set)

    return num_users, num_items, max_seq_length


config.num_users, config.num_items, config.max_seq_length = get_users_items_num_and_max_seq_length(config.data_path)
config.device = f'cuda:{get_attribute("cuda")}' if torch.cuda.is_available() and get_attribute("cuda") >= 0 else 'cpu'

print(f'Config.num_items :{config.num_items}')