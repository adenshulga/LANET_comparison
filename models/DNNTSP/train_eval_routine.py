from  tqdm import tqdm
from .utils.util import convert_all_data_to_gpu, convert_to_gpu
from utils.utils import save_to_csv

import torch.nn as nn
import torch

import numpy as np
from sklearn.metrics import roc_auc_score

def train_epoch(model, training_data, optimizer, opt):
    """ Epoch operation in training phase. """

    model.train()

    loss_func = nn.MultiLabelSoftMarginLoss(reduction="mean")
    loss_func = convert_to_gpu(loss_func)

    for batch in tqdm(training_data, mininterval=2,
                      desc='  - (Training)   ', leave=False):
        """ prepare data """
        g, nodes_feature, edges_weight, lengths, nodes, truth_data, users_frequency = batch
        g, nodes_feature, edges_weight, lengths, nodes, truth_data, users_frequency = \
                    convert_all_data_to_gpu(g, nodes_feature, edges_weight, lengths, nodes, truth_data, users_frequency)


        output = model(g, nodes_feature, edges_weight, lengths, nodes, users_frequency)
        loss = loss_func(output, truth_data.float())
        # total_loss += loss.cpu().data.numpy()
    
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
    loss_func = convert_to_gpu(loss_func)

    s = np.zeros(232)

    with torch.no_grad():
        for batch in tqdm(validation_data, mininterval=2,
                          desc='  - (Validation) ', leave=False):
            g, nodes_feature, edges_weight, lengths, nodes, truth_data, users_frequency = batch
            g, nodes_feature, edges_weight, lengths, nodes, truth_data, users_frequency = \
                    convert_all_data_to_gpu(g, nodes_feature, edges_weight, lengths, nodes, truth_data, users_frequency)


            output = model(g, nodes_feature, edges_weight, lengths, nodes, users_frequency)
            loss = loss_func(output, truth_data.float())
            pred_label.append(output.detach().cpu())
            true_label.append(truth_data.detach().cpu())

            # print(truth_data.detach().cpu().sum(axis=0))

            total_ll += loss.cpu().data.numpy()

        # print(s)

        true_label = torch.cat(true_label, dim=0)
        pred_label = torch.cat(pred_label, dim=0)

    tasks_with_non_trivial_targets = np.where(true_label.sum(axis=0) != 0)[0]
    y_pred_copy = pred_label[:, tasks_with_non_trivial_targets].numpy()
    y_true_copy = true_label[:, tasks_with_non_trivial_targets].numpy()
    roc_auc = roc_auc_score(y_true=y_true_copy, y_score=y_pred_copy, average='weighted')
    
    # print(" weighted roc_auc{}".format(roc_auc_score(y_true=true_label, y_score=pred_label, average='weighted')))
    # roc_auc_mean = np.mean(roc_auc) 
    
    return total_ll, roc_auc

def evaluate(model, dataloader, opt, type) -> None:
    
    model.eval()

    y_true = []
    y_pred = []
    with torch.no_grad():
        for step, (g, nodes_feature, edges_weight, lengths, nodes, truth_data, users_frequency) in enumerate(
                tqdm(dataloader)):
            g, nodes_feature, edges_weight, lengths, nodes, truth_data, users_frequency = \
                convert_all_data_to_gpu(g, nodes_feature, edges_weight, lengths, nodes, truth_data, users_frequency)

            predict_data = model(g, nodes_feature, edges_weight, lengths, nodes, users_frequency)

            # predict_data shape (batch_size, baskets_num, items_total)
            # truth_data shape (batch_size, baskets_num, items_total)
            y_pred.append(predict_data.detach().cpu())
            y_true.append(truth_data.detach().cpu())

        y_pred = torch.cat(y_pred, dim=0)
        y_true = torch.cat(y_true, dim=0)

        save_to_csv(y_pred, f'pred_{type}', opt)
        save_to_csv(y_true, f'gt_{type}', opt)
    
