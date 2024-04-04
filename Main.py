import argparse
import numpy as np
import pickle
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, f1_score, hamming_loss
import os

# from transformer.Models import Transformer
# from lanet.LANET import TransformerEncoder
from tqdm import tqdm
from copy import deepcopy

from utils.utils import set_random_seed, import_by_name
from utils.load_config import config


def train(model, training_data, validation_data, test_data,  optimizer, scheduler, opt):
    """ Start training. """
    """ TODO: write train epoch and eval epoch for each model"""
    best_auc_roc = 0
    impatient = 0 
    best_model = deepcopy(model.state_dict())

    for epoch_i in range(opt.epoch):
        epoch = epoch_i + 1
        print('[ Epoch', epoch, ']')

        start = time.time()

        train_epoch = import_by_name(f'models.{opt.model_name}.train_eval_routine', 'train_epoch')
        eval_epoch = import_by_name(f'models.{opt.model_name}.train_eval_routine', 'eval_epoch')

        train_event = train_epoch(model, training_data, optimizer, opt)
        print('  - (Train)    negative loglikelihood: {ll: 8.4f}, '
              'elapse: {elapse:3.3f} min'
              .format(ll=train_event, elapse=(time.time() - start) / 60))

        start = time.time()
        valid_event, valid_roc_auc = eval_epoch(model, validation_data, opt)
        print('  - (dev)    nll: {ll: 8.4f}, '
              ' roc auc : {type:8.4f},'
              'elapse: {elapse:3.3f} min'
              
              .format(ll=valid_event, type=valid_roc_auc, elapse=(time.time() - start) / 60))
        
        start = time.time()
        test_event, test_roc_auc = eval_epoch(model, test_data, opt)
        print('  - (test)    nll: {ll: 8.4f}, '
              ' roc auc :{type:8.4f},'
              'elapse: {elapse:3.3f} min'
              
              .format(ll=test_event, type=test_roc_auc, elapse=(time.time() - start) / 60))


        if (valid_roc_auc - best_auc_roc ) < 1e-5:
            impatient += 1
            if best_auc_roc < valid_roc_auc:
                best_auc_roc = valid_roc_auc
                best_model = deepcopy(model.state_dict())
  
        else:
            best_auc_roc = valid_roc_auc
            best_model = deepcopy(model.state_dict())
            impatient = 0
        
            
        if impatient >= 20:
            print(f'Breaking due to early stopping at epoch {epoch}')
            break

        scheduler.step()

    return best_model


def main():
    """ Main function. 
    Parse config file, create model, create dataloader for model
    """

    opt = config

    opt.device = torch.device(f'cuda:{opt.cuda}')

    print('[Info] parameters: {}'.format(opt))
    
    set_random_seed(opt.seed)

    """ prepare model """
    create_model = import_by_name(f'models.{opt.model_name}.model_creation', 'create_model')
    model = create_model(opt)

    model.to(opt.device)
    opt.model = model

    """ prepare dataloader """
    prepare_dataloader = import_by_name(f'models.{opt.model_name}.prepare_dataloader', 'prepare_dataloader')
    trainloader, devloader, testloader = prepare_dataloader(opt)

    """ optimizer and scheduler """
    # TODO: parse parameters from config
    optimizer = optim.Adam(filter(lambda x: x.requires_grad, model.parameters()),
                           opt.lr, betas=opt.betas, eps=opt.eps)
    scheduler = optim.lr_scheduler.StepLR(optimizer, opt.scheduler_step, gamma=opt.gamma)


    """ number of parameters """
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print('[Info] Number of parameters: {}'.format(num_params))

    """ train the model """
    # TODO: write common train function 
    best_model = train(model, trainloader, devloader, testloader, optimizer, scheduler, opt)
    
    model.load_state_dict(best_model)
    model.eval()
    # save the model
    model_save_path = f"saved_models/{opt.model_name}/{opt.dataset_name}"
    os.makedirs(model_save_path, exist_ok=True)
    torch.save(model.state_dict(), model_save_path + f'/run_{opt.seed}')












import time
start = time.time()

if __name__ == '__main__':
    main()
end= time.time()
print("total training time is {}".format(end-start))

