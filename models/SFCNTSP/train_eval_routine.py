from  tqdm import tqdm
# from .utils.util import convert_all_data_to_gpu, convert_to_gpu
from utils.utils import save_to_csv
from .utils.utils import convert_to_gpu
from .utils.load_config import get_attribute


import torch.nn as nn
import torch

import numpy as np
from sklearn.metrics import roc_auc_score

def train_epoch(model, training_data, optimizer, opt):
    """ Epoch operation in training phase. """

    model.train()

    loss_func = nn.MultiLabelSoftMarginLoss(reduction="mean")
    # loss_func = convert_to_gpu(loss_func)

    for batch in tqdm(training_data, mininterval=2,
                      desc='  - (Training)   ', leave=False):
        """ prepare data """
        _, batch_items_id, batch_seq_length, batch_set_size, batch_input_data, batch_truth_data = batch
        batch_input_data, batch_truth_data = convert_to_gpu(batch_input_data, batch_truth_data,
                                                                device=get_attribute('device'))

        
        batch_output = model(batch_seq_length=batch_seq_length, batch_items_id=batch_items_id,
                                batch_set_size=batch_set_size, batch_input_data=batch_input_data)

        loss = loss_func(batch_output, batch_truth_data)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return loss

def eval_epoch(model, validation_data, opt):
    """ Epoch operation in evaluation phase. """

    model.eval()
    
    total_ll = 0
    pred_label = []
    true_label = []

    loss_func = nn.MultiLabelSoftMarginLoss(reduction="mean")
    # loss_func = convert_to_gpu(loss_func)
    with torch.no_grad():
        for batch in tqdm(validation_data, mininterval=2,
                        desc='  - (Training)   ', leave=False):
            """ prepare data """
            _, batch_items_id, batch_seq_length, batch_set_size, batch_input_data, batch_truth_data = batch
            batch_input_data, batch_truth_data = convert_to_gpu(batch_input_data, batch_truth_data,
                                                                    device=get_attribute('device'))

            
            batch_output = model(batch_seq_length=batch_seq_length, batch_items_id=batch_items_id,
                                    batch_set_size=batch_set_size, batch_input_data=batch_input_data)

            loss = loss_func(batch_output, batch_truth_data)

            pred_label.append(batch_output.detach().cpu())
            true_label.append(batch_truth_data.detach().cpu())

            total_ll += loss.cpu().data.numpy()

        true_label = torch.cat(true_label, dim=0)
        pred_label = torch.cat(pred_label, dim=0)

    tasks_with_non_trivial_targets = np.where(true_label.sum(axis=0) != 0)[0]
    y_pred_copy = pred_label[:, tasks_with_non_trivial_targets].numpy()
    y_true_copy = true_label[:, tasks_with_non_trivial_targets].numpy()
    roc_auc = roc_auc_score(y_true=y_true_copy, y_score=y_pred_copy, average='weighted')
    
    return total_ll, roc_auc

def evaluate(model, dataloader, opt, type) -> None:

    model.eval()

    y_true = []
    y_pred = []

    with torch.no_grad():

        tqdm_loader = tqdm(dataloader, ncols=175)

        for batch, (_, batch_items_id, batch_seq_length, batch_set_size, batch_input_data, batch_truth_data) in enumerate(tqdm_loader):

            batch_input_data, batch_truth_data = convert_to_gpu(batch_input_data, batch_truth_data,
                                                                device=get_attribute('device'))

            batch_output = model(batch_seq_length=batch_seq_length, batch_items_id=batch_items_id,
                                 batch_set_size=batch_set_size, batch_input_data=batch_input_data)
            
            y_true.append(batch_truth_data.detach().cpu())
            y_pred.append(batch_output.detach().cpu())
            
        y_pred = torch.cat(y_pred, dim=0)
        y_true = torch.cat(y_true, dim=0)
        
        save_to_csv(y_pred, f'pred_{type}', opt)
        save_to_csv(y_true, f'gt_{type}', opt)

    
