import torch
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
import numpy as np


from utils.utils import save_to_csv


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
        
        a,b,c = enc_out[:,:-1,:].shape[0],enc_out[:,:-1,:].shape[1],enc_out[:,:-1,:].shape[2]
        

        """ backward """

        # calculate P*(y_i+1) by mbn: 
        log_loss_type = model.MBN.loss(enc_out[:,:-1,:].reshape(a*b,c), event_type[:,1:,:].reshape(a*b,model.num_types) )
        
        pred_type = (model.MBN.predict(enc_out[:,:-1,:].reshape(a*b,c) )[(non_pad_mask[:,1:,:].repeat(1,1, model.num_types)==1).reshape(a*b,model.num_types)]).reshape(-1,model.num_types)
        
        #bk loss
        #defi : 0:5, synthetic 0:2
        constraint_loss1 = torch.sum(torch.square( torch.sum(pred_type[:,0:2],dim=1)-1))
        constraint_loss2 = torch.sum(torch.square( torch.sum(pred_type[:,2:],dim=1)-1))
        
        combined_constraint_loss = constraint_loss1 + constraint_loss2
        
        # sum log loss
        loss = torch.sum(log_loss_type.reshape(a,b)  * non_pad_mask[:,1:,0]) # + 0.001*combined_constraint_loss #
              
        loss.backward()

        """ update parameters """
        optimizer.step()

        """ note keeping """

    return loss

def eval_epoch(model, validation_data, opt):
    """ Epoch operation in evaluation phase. """

    model.eval()
    
    total_ll = 0
    total_event = 0
    total_time_nll = 0
    pred_label = []
    true_label = []

    with torch.no_grad():
        for batch in tqdm(validation_data, mininterval=2,
                          desc='  - (Validation) ', leave=False):
            """ prepare data """

            event_time, time_gap, event_type = map(lambda x: x.to(opt.device), batch)

            enc_out, non_pad_mask = model(event_type, event_time)

            a,b,c = enc_out[:,:-1,:].shape[0],enc_out[:,:-1,:].shape[1],enc_out[:,:-1,:].shape[2]
            

            # calculate P*(y_i+1) by mbn: 
            log_loss_type = model.MBN.loss(enc_out[:,:-1,:].reshape(a*b,c), event_type[:,1:,:].reshape(a*b,model.num_types))
            
            pred_type = (model.MBN.predict(enc_out[:,:-1,:].reshape(a*b,c) )[(non_pad_mask[:,1:,:].repeat(1,1, model.num_types)==1).reshape(a*b,model.num_types)]).reshape(-1,model.num_types)
            
            
            pred_label += list(pred_type.cpu().numpy())

            true_type = (event_type[:,1:,:][(non_pad_mask[:,1:,:].repeat(1,1, model.num_types)==1)]).reshape(-1,model.num_types)
            
                      
            true_label += list(true_type.cpu().numpy())
           
            
            #  log loss
            loss = torch.sum(log_loss_type.reshape(a,b) * non_pad_mask[:,1:,0])

            """ note keeping """
            total_ll += loss.item()

             
    roc_auc = roc_auc_score(y_true=true_label, y_score=pred_label, average=None)
    
    print(" weighted roc_auc{}".format(roc_auc_score(y_true=true_label, y_score=pred_label, average='weighted')))
    
    
    roc_auc_mean = np.mean(roc_auc) 
    
    
    return total_ll, roc_auc_mean

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

            last_non_pad_idx = (non_pad_mask.squeeze(-1).sum(dim=1) - 1).long()
            batch_indices = torch.arange(enc_out.size(0), device=opt.device)

            # Extract the last valid encoder output for each sequence
            last_enc_out = enc_out[batch_indices, last_non_pad_idx]

            # Assuming the existence of a function to predict types based on the last encoder output
            pred_type = model.MBN.predict(last_enc_out)

            pred_label.append(pred_type.cpu())
            # Extract the corresponding true types for the last position
            last_true_type = event_type[batch_indices, last_non_pad_idx]
            true_label.append(last_true_type.cpu())

    # Concatenate all the stored last predictions and true labels
    y_pred = torch.cat(pred_label, dim=0)
    y_true = torch.cat(true_label, dim=0)

    save_to_csv(y_pred, f'pred_{type}', opt)
    save_to_csv(y_true, f'gt_{type}', opt)