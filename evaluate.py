
import time
import torch


from utils.load_config import config
from utils.utils import set_random_seed, import_by_name

    


def main():
    """ Main function. """
    opt = config

    # default device is CUDA
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


    """ evaluate on test set"""
    model.load_state_dict( torch.load(f"saved_models/{opt.model_name}/{opt.dataset_name}/run_{opt.seed}") ) 
    model.eval()

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print('[Info] Number of parameters: {}'.format(num_params))
    # opt_tau = eval_epoch(model, devloader, opt)
    # test_epoch(model, testloader, opt, opt_tau)
    evaluate = import_by_name(f'models.{opt.model_name}.train_eval_routine', 'evaluate')
    evaluate(model, devloader, opt, 'valid')
    evaluate(model, testloader, opt, 'test')

start = time.time()

if __name__ == '__main__':
    main()
end= time.time()
print("total training time is {}".format(end-start))

