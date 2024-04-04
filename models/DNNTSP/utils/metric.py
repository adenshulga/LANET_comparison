import numpy as np
import torch
from tqdm import tqdm
from utils.util import convert_all_data_to_gpu
import datetime
import os

from utils.load_config import get_attribute
from sklearn.metrics import f1_score, hamming_loss, roc_auc_score



def recall_score(y_true, y_pred, top_k=5):
    """
    Args:
        y_true (Tensor): shape (batch_size, items_total)
        y_pred (Tensor): shape (batch_size, items_total)
        top_k (int):
    Returns:
        output (float)
    """
    # start_time = datetime.datetime.now()
    print(y_pred.shape)
    _, predict_indices = y_pred.topk(k=top_k)
    predict, truth = y_pred.new_zeros(y_pred.shape).scatter_(dim=1, index=predict_indices,
                                                             value=1).long(), y_true.long()
    tp, t = ((predict == truth) & (truth == 1)).sum(-1), truth.sum(-1)
    # end_time = datetime.datetime.now()
    # print("recall_score cost %d seconds" % (end_time - start_time).seconds)
    return (tp.float() / t.float()).mean().item()


def dcg(y_true, y_pred, top_k):
    """
    Discounted Cumulative Gain
    Args:
        y_true: (batch_size, items_total)
        y_pred: (batch_size, items_total)
        top_k (int):

    Returns:

    """
    _, predict_indices = y_pred.topk(k=top_k)
    gain = y_true.gather(-1, predict_indices)  # (batch_size, top_k)
    return (gain.float() / torch.log2(torch.arange(top_k, device=y_pred.device).float() + 2)).sum(-1)  # (batch_size,)


def ndcg_score(y_true, y_pred, top_k):
    """
    Normalized Discounted Cumulative Gain
    Args:
        y_true: (batch_size, items_total)
        y_pred: (batch_size, items_total)
        top_k (int):
    Returns:

    """
    # start_time = datetime.datetime.now()
    dcg_score = dcg(y_true, y_pred, top_k)
    idcg_score = dcg(y_true, y_true, top_k)
    # end_time = datetime.datetime.now()
    # print("ndcg cost %d seconds" % (end_time - start_time).seconds)
    return (dcg_score / idcg_score).mean().item()


def PHR(y_true, y_pred, top_k=5):
    """
    Precision at Rank k
    Args:
        y_true (Tensor): shape (batch_size, items_total)
        y_pred (Tensor): shape (batch_size, items_total)
        top_k (int):
    Returns:
        output (float)
    """
    # start_time = datetime.datetime.now()
    _, predict_indices = y_pred.topk(k=top_k)
    predict, truth = y_pred.new_zeros(y_pred.shape).scatter_(dim=1, index=predict_indices,
                                                             value=1).long(), y_true.long()
    hit_num = torch.mul(predict, truth).sum(dim=1).nonzero().shape[0]
    # end_time = datetime.datetime.now()
    # print("PHR cost %d seconds" % (end_time - start_time).seconds)
    return hit_num / truth.shape[0]

def compute_f1(y_true, y_pred, threshold=0.5):
    y_pred_binarized = (y_pred > threshold).float()
    return f1_score(y_true.numpy(), y_pred_binarized.numpy(), average='weighted')

def compute_hamming_loss(y_true, y_pred, threshold=0.5):
    y_pred_binarized = (y_pred > threshold).float()
    return hamming_loss(y_true.numpy(), y_pred_binarized.numpy())

def compute_roc_auc(y_true, y_pred):
    return roc_auc_score(y_true.numpy(), y_pred.numpy(), average='weighted')

def get_best_metric(y_true, y_pred, metric_func, min_or_max):
    thr_list = np.arange(100)/100
    metric_list = []
    for thr in list(thr_list):
        metric_list.append(metric_func(y_true, y_pred, thr))
    if min_or_max == 'max':
        value = max(metric_list)
    else:
        value = min(metric_list)
    return value


def get_metric(y_true, y_pred, top_k_arr, eval_mode='train'):
    """
        Args:
            y_true: tensor (samples_num, items_total)
            y_pred: tensor (samples_num, items_total)
        Returns:
            scores: dict
    """
    result = {}
    tasks_with_non_trivial_targets = np.where(y_true.sum(axis=0) != 0)[0]
    y_pred_copy = y_pred[:, tasks_with_non_trivial_targets]
    y_true_copy = y_true[:, tasks_with_non_trivial_targets]
    for top_k in top_k_arr:
        result.update({
            f'recall_{top_k}': recall_score(y_true, y_pred, top_k=top_k),
            f'ndcg_{top_k}': ndcg_score(y_true, y_pred, top_k=top_k),
            f'PHR_{top_k}': PHR(y_true, y_pred, top_k=top_k)
        })
    if eval_mode == 'valid':
        result.update({
        'roc_auc': compute_roc_auc(y_true_copy, y_pred_copy),
        'f1_score': get_best_metric(y_true_copy, y_pred_copy, compute_f1, 'max'),
        'hamming_loss': get_best_metric(y_true, y_pred, compute_hamming_loss, 'min')
        })
    
        return result
    

    result.update({
        'roc_auc': compute_roc_auc(y_true_copy, y_pred_copy),
        'f1_score': compute_f1(y_true_copy, y_pred_copy),
        'hamming_loss': compute_hamming_loss(y_true, y_pred)
    })
    return result

def save_to_csv(y, type):
    'type is pred_test or pred_valid'
    path_to_save = f'model_pred_and_gt/{get_attribute("data")}/run_{get_attribute("seed")}/{type}'
    os.makedirs(path_to_save, exist_ok=True)
    y_true_np = np.savetxt(os.path.join(path_to_save, 'data.csv'), y.numpy(), delimiter=',')


def evaluate(model, data_loader, top_k):
    """
    Args:
        model: nn.ModuleSjeqgJ5L6BSjeqgJ5L6B
        data_loader: DataLoader
    Returns:
        scores: dict
    """
    model.eval()

    y_true = []
    y_pred = []
    with torch.no_grad():
        for step, (g, nodes_feature, edges_weight, lengths, nodes, truth_data, users_frequency) in enumerate(
                tqdm(data_loader)):
            g, nodes_feature, edges_weight, lengths, nodes, truth_data, users_frequency = \
                convert_all_data_to_gpu(g, nodes_feature, edges_weight, lengths, nodes, truth_data, users_frequency)

            predict_data = model(g, nodes_feature, edges_weight, lengths, nodes, users_frequency)

            # predict_data shape (batch_size, baskets_num, items_total)
            # truth_data shape (batch_size, baskets_num, items_total)
            y_pred.append(predict_data.detach().cpu())
            y_true.append(truth_data.detach().cpu())

        y_pred = torch.cat(y_pred, dim=0)
        y_true = torch.cat(y_true, dim=0)

        save_to_csv(y_pred, 'pred_test')
        save_to_csv(y_true, 'gt_test')

        print(f' Sum: {np.sum(y_true.numpy(), axis = 0)}, Shape:{y_true.numpy().shape}')

        return get_metric(y_true=y_true, y_pred=y_pred, top_k_arr=top_k)
    
def evaluate_valid(model, data_loader):
    """
    Args:
        model: nn.Module
        data_loader: DataLoader
    Returns:
        None
    """
    model.eval()

    y_true = []
    y_pred = []
    with torch.no_grad():
        for step, (g, nodes_feature, edges_weight, lengths, nodes, truth_data, users_frequency) in enumerate(
                tqdm(data_loader)):
            g, nodes_feature, edges_weight, lengths, nodes, truth_data, users_frequency = \
                convert_all_data_to_gpu(g, nodes_feature, edges_weight, lengths, nodes, truth_data, users_frequency)

            predict_data = model(g, nodes_feature, edges_weight, lengths, nodes, users_frequency)

            # predict_data shape (batch_size, baskets_num, items_total)
            # truth_data shape (batch_size, baskets_num, items_total)
            y_pred.append(predict_data.detach().cpu())
            y_true.append(truth_data.detach().cpu())

        y_pred = torch.cat(y_pred, dim=0)
        y_true = torch.cat(y_true, dim=0)

        save_to_csv(y_pred, 'pred_valid')
        save_to_csv(y_true, 'gt_valid')
