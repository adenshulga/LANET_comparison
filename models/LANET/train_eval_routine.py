import torch
from tqdm import tqdm
import numpy as np
from sklearn.metrics import roc_auc_score
from utils.utils import save_to_csv

def multilabel_celoss(x, y):
    mean0 = 1 - x + 10**(-9)
    mean1 = x + 10**(-9)
    mean1log = torch.log(mean1)
    mean0log = torch.log(mean0)
    
    logProbTerm1 = y * mean1log
    logProbTerm2 = (1-y) * mean0log
    
    logterms = logProbTerm1 + logProbTerm2 
    loss = -torch.sum(logterms, dim=1)
    return loss    

def train_epoch(model, training_data, optimizer, opt):
    """ Epoch operation in training phase. """

    model.train()


    for batch in tqdm(training_data, mininterval=2,
                      desc='  - (Training)   ', leave=False):
        """ prepare data """
        event_time, time_gap, event_type = map(lambda x: x.to(opt.device), batch)
        

        """ forward """
        optimizer.zero_grad()
        
        enc_out, non_pad_mask = model(event_type, event_time)
        
        a, b, c = enc_out[:, :-1, :].shape[0], enc_out[:,:-1,:].shape[1], enc_out[:,:-1,:].shape[2]
        

        """ backward """

        # calculate P*(y_i+1) by mbn: 
        log_loss_type = multilabel_celoss(enc_out[:, :-1, :].reshape(a*b, c), event_type[:, 1: ,:].reshape(a*b, model.num_types) )

        # sum log loss
        loss = torch.sum(log_loss_type.reshape(a, b) * non_pad_mask[:, 1:, 0]) 
              
        loss.backward()

        """ update parameters """
        optimizer.step()

        """ note keeping """

    return loss

def eval_epoch(model, validation_data, opt):
    """ Epoch operation in evaluation phase. """

    # model.eval()
    
    # total_ll = 0
    # total_event = 0
    # total_time_nll = 0
    # pred_label = []
    # true_label = []

    # with torch.no_grad():
    #     for batch in tqdm(validation_data, mininterval=2,
    #                       desc='  - (Validation) ', leave=False):
    #         """ prepare data """

    #         event_time, time_gap, event_type = map(lambda x: x.to(opt.device), batch)

    #         enc_out, non_pad_mask = model(event_type, event_time)
            
    #         a, b, c = enc_out[:, :-1, :].shape[0], enc_out[:,:-1,:].shape[1], enc_out[:,:-1,:].shape[2]
            

    #         """ backward """

    #         # calculate P*(y_i+1) by mbn: 
    #         log_loss_type = multilabel_celoss(enc_out[:, :-1, :].reshape(a*b, c), event_type[:, 1: ,:].reshape(a*b, model.num_types) )
            
    #         pred_type = (enc_out[:, :-1, :][(non_pad_mask[:,1:,:].repeat(1,1, model.num_types)==1)]).reshape(-1, model.num_types)
            
            
    #         pred_label += list(pred_type.cpu().numpy())

    #         true_type = (event_type[:, 1:, :][(non_pad_mask[:,1:,:].repeat(1,1, model.num_types)==1)]).reshape(-1, model.num_types)
            
                      
    #         true_label += list(true_type.cpu().numpy())
           
            
    #         #  log loss
    #         loss = torch.sum(log_loss_type.reshape(a,b) * non_pad_mask[:,1:,0])

    #         """ note keeping """
    #         total_ll += loss.item()

             
    # roc_auc = roc_auc_score(y_true=true_label, y_score=pred_label, average=None)
    
    # print(" weighted roc_auc{}".format(roc_auc_score(y_true=true_label, y_score=pred_label, average='weighted')))
    
    
    # roc_auc_mean = np.mean(roc_auc) 
    
    
    # return total_ll, roc_auc_mean

    model.eval()
    
    pred_label = []
    true_label = []

    with torch.no_grad():
        for batch in tqdm(validation_data, mininterval=2,
                          desc=f'  - ({type}) ', leave=False):
            """ prepare data """

            event_time, time_gap, event_type = map(lambda x: x.to(opt.device), batch)

            enc_out, non_pad_mask = model(event_type, event_time)

            last_pred_idx = (non_pad_mask.sum(dim=1) - 1).long()
            batch_indices = torch.arange(enc_out.size(0)).to(enc_out.device) 
            pred_type = enc_out[batch_indices, last_pred_idx.squeeze(), :]

            # Store the last prediction for each sequence
            # pred_label += list(pred_type.detach().cpu())
            pred_label.append(pred_type.detach().cpu())

            # Extract the true value for the corresponding last predictions
            true_type = event_type[batch_indices, last_pred_idx.squeeze(), :]
            
            # Store the last true value for each sequence
            # true_label += list(true_type.detach().cpu())
            true_label.append(true_type.detach().cpu())


        true_label = torch.cat(true_label, dim=0)
        pred_label = torch.cat(pred_label, dim=0)
         
    tasks_with_non_trivial_targets = np.where(true_label.sum(axis=0) != 0)[0]
    y_pred_copy = pred_label[:, tasks_with_non_trivial_targets].numpy()
    y_true_copy = true_label[:, tasks_with_non_trivial_targets].numpy()
    roc_auc = roc_auc_score(y_true=y_true_copy, y_score=y_pred_copy, average='weighted')

    return 0, roc_auc

def evaluate(model, dataloader, opt, type) -> None:
    
    model.eval()
    
    pred_label = []
    true_label = []

    with torch.no_grad():
        for batch in tqdm(dataloader, mininterval=2,
                          desc=f'  - ({type}) ', leave=False):
            """ prepare data """

            event_time, time_gap, event_type = map(lambda x: x.to(opt.device), batch)

            enc_out, non_pad_mask = model(event_type, event_time)

            last_pred_idx = (non_pad_mask.sum(dim=1) - 1).long()
            batch_indices = torch.arange(enc_out.size(0)).to(enc_out.device) 
            pred_type = enc_out[batch_indices, last_pred_idx.squeeze(), :]

            # Store the last prediction for each sequence
            pred_label += list(pred_type.detach().cpu())

            # Extract the true value for the corresponding last predictions
            true_type = event_type[batch_indices, last_pred_idx.squeeze(), :]
            
            # Store the last true value for each sequence
            true_label += list(true_type.detach().cpu())
         
    y_pred = torch.cat(pred_label, dim=0).reshape(-1, model.num_types)
    y_true = torch.cat(true_label, dim=0).reshape(-1, model.num_types)
    


    save_to_csv(y_pred, f'pred_{type}', opt)
    save_to_csv(y_true, f'gt_{type}', opt)
