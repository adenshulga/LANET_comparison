import json
import copy
import itertools
from torch.utils.data import Dataset, DataLoader
from functools import partial
import torch
import pickle
import os

import numpy as np

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


class CustomizedDataset(Dataset):
    """
    Customized Dataset
    """
    def __init__(self, data_path: str, data_type: str):
        """
        :param data_path: json file
        :param data_type: str, train, validate or test
        """

        self.data_path = data_path

        data = retrieve_dict(data_path)
        converted_dataset = convert_dataset_from_pickle(data)


        self.data_list = converted_dataset[data_type]

    def __getitem__(self, index: int):
        """
        :param index:
        :return:  user_id, int
                sets_item_list, list of sets(each set is a list of items)
        """
        # list of sets of a specific user
        user_sets = self.data_list[index]

        # int
        user_id = user_sets[0]['user_id']
        # list, shape (seq_len, items_num)
        sets_items_list = [user_set['items_id'] for user_set in user_sets]

        return user_id, sets_items_list

    def __len__(self):
        return len(self.data_list)


def pad_sequence_scalar(seq_to_pad: list, length: int, pad_value: int = -1):
    """
    padding seq_to_pad to specific length with pad_value
    :return: list, seq with len(seq)==length
    """
    if len(seq_to_pad) < length:
        seq_to_pad += [pad_value for _ in range(length - len(seq_to_pad))]

    return seq_to_pad


def pad_sequence_list(seq_to_pad: list, length: int, each_list_length: int, pad_value: int = -1):
    """
    padding seq_to_pad to specific length with list of pad_value
    :return: seq with len(seq)==length
    """
    if len(seq_to_pad) < length:
        seq_to_pad += [[pad_value for _ in range(each_list_length)] for _ in range(length - len(seq_to_pad))]

    return seq_to_pad


def get_k_hot_encoding(id_list: list, num_classes: int):
    """
    get k-hot encoding based on the input ids
    :param id_list: list, list of ids, shape (input_items_num, )
    :param num_classes:
    :return:
        k_hot_encoding[i] = 1 if i in id_list, else 0
    """
    k_hot_encoding = torch.zeros(num_classes)
    if len(id_list) > 0:
        k_hot_encoding[id_list] = 1
    return k_hot_encoding


def collate_fn(batch_data: list, max_seq_length: int, num_items: int):
    """
    :param batch_data: list, shape (batch_size, XXX), XXX is the tuple of user_id and sets_items_list
    :param max_seq_length: int, the max length of sequence in the dataset
    :param num_items: int, number of items in the dataset
    Returns:
        batch_user_id, batch_items_id, batch_seq_length, batch_set_size, batch_input_data, batch_truth_data
    """
    # user_id, sets_items_list
    # zip * -> unpack
    ret = list()
    for idx, b_data in enumerate(zip(*batch_data)):
        # user_id
        if isinstance(b_data[0], int) and idx == 0:
            # user id, list shape -> (batch_size, )
            ret.append(b_data)

        elif isinstance(b_data[0], list) and idx == 1:
            # need to exclude the last set in data, which is the objective to predict
            # list of sets, b_data -> [[set_1, set_2, ...], [set_1, set_2, ...], ...]

            # batch_items_id, list of items id in each set, shape (batch_size, user_seq_len, each_set_size)
            batch_items_id = [data[:-1] for data in b_data]
            ret.append(batch_items_id)

            # batch_seq_length, list of sequence length, shape (batch_size, )
            batch_seq_length = [len(data[:-1]) for data in b_data]
            ret.append(batch_seq_length)

            # list of list, each list contains the number of items in each set
            # batch_set_size, list of each set size, shape (batch_size, user_seq_len)
            batch_set_size = [[len(d) for d in data[:-1]] for data in b_data]
            batch_max_set_size = max(list(itertools.chain.from_iterable(batch_set_size)))
            ret.append(batch_set_size)

            # ensure that the padding operation not affect the original data
            # assert -1 not in list(set(itertools.chain.from_iterable([set(itertools.chain.from_iterable(data)) for data in b_data])))

            # list, shape -> (batch_size, max_seq_len, max_set_size)
            batch_input_data = []

            # pad across seq level and set level
            for index, data in enumerate(b_data):
                input_data = data[:-1]
                for idx, d in enumerate(input_data):
                    input_data[idx] = pad_sequence_scalar(seq_to_pad=copy.deepcopy(d), length=batch_max_set_size, pad_value=-1)
                batch_input_data.append(pad_sequence_list(seq_to_pad=input_data, length=max_seq_length, each_list_length=batch_max_set_size, pad_value=-1))

            # tensor, shape -> (batch_size, max_seq_len, max_set_size)
            ret.append(torch.tensor(batch_input_data))

            batch_truth_data = torch.stack([get_k_hot_encoding(id_list=data[-1], num_classes=num_items) for data in b_data], dim=0)

            ret.append(batch_truth_data)
        else:
            raise ValueError(f'wrong data type -> {type(b_data[0])}')

    # (batch_user_id, batch_items_id, batch_seq_length, batch_set_size, batch_input_data, batch_truth_data)
    return tuple(ret)


def get_data_loader(data_path: str, data_type: str, batch_size: int, max_seq_length: int, num_items: int, num_workers: int = 0):
    """
    Args:
        data_path: str
        data_type: str, 'train'/'validate'/'test'
        batch_size: int
        max_seq_length: int, the max length of sequence in the dataset
        num_items: int, number of items in the dataset
        num_workers: int
    Returns:
        data_loader: DataLoader
    """

    dataset = CustomizedDataset(data_path=data_path, data_type=data_type)

    data_loader = DataLoader(dataset=dataset,
                             batch_size=batch_size,
                             shuffle=True,
                             drop_last=False,
                             num_workers=num_workers,
                             collate_fn=partial(collate_fn, max_seq_length=max_seq_length, num_items=num_items))
    return data_loader
