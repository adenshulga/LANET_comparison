
import time
import torch


from utils.load_config import config, Config
from utils.utils import set_random_seed, import_by_name, append_to_txt

from copy import deepcopy    


def main(config):
    """ Main function. """
    opt = config

    # default device is CUDA
    opt.device = torch.device(f'cuda:{opt.cuda}')


    set_random_seed(opt.seed)

    """ prepare model """
    create_model = import_by_name(f'models.{opt.model_name}.model_creation', 'create_model')

    model = create_model(opt)
    model.to(opt.device)
    opt.model = model

    print('[Info] parameters: {}'.format(opt))

    """ prepare dataloader """
    prepare_dataloader = import_by_name(f'models.{opt.model_name}.prepare_dataloader', 'prepare_dataloader')

    trainloader, devloader, testloader = prepare_dataloader(opt)


    """ evaluate on test set"""
    if config.ablation:
        model.load_state_dict( torch.load(f"saved_models/{opt.model_name}_ablation_{config.ablation_note}/{opt.dataset_name}/run_{opt.seed}") ) 
    else:
        model.load_state_dict( torch.load(f"saved_models/{opt.model_name}/{opt.dataset_name}/run_{opt.seed}"))

    model.eval()

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print('[Info] Number of parameters: {}'.format(num_params))
    # opt_tau = eval_epoch(model, devloader, opt)
    # test_epoch(model, testloader, opt, opt_tau)
    evaluate = import_by_name(f'models.{opt.model_name}.train_eval_routine', 'evaluate')
    evaluate(model, devloader, opt, 'valid')
    evaluate(model, testloader, opt, 'test')

    append_to_txt(f'---------------------------------------')
    append_to_txt(f'Evaluated model with following config')
    append_to_txt(str(config))
    append_to_txt(f'Number of parameters: {num_params}')

start = time.time()

def get_list(obj):
    if isinstance(obj, list):
        return obj
    else:
        return [obj]

model_names = deepcopy(get_list(config.model_name))
dataset_names = deepcopy(get_list(config.dataset_name))
seeds = deepcopy(get_list(config.seed))

if __name__ == '__main__':
    for dataset in dataset_names:
        for model in model_names:
            for seed in seeds:
                config.modify_config(model_name=model, dataset_name=dataset, seed=seed)
                main(config=config)

# if __name__ == '__main__':
#     main(config)
    
end= time.time()
print("total training time is {}".format(end-start))

