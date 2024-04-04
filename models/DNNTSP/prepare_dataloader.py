from .utils.data_container import get_data_loader

def prepare_dataloader(opt):
    """ Load data and prepare dataloader. """

        
    trainloader = get_data_loader(data_path=opt.data,
                                        data_type='train',
                                        batch_size=opt.batch_size,
                                        item_embedding_matrix=opt.model.item_embedding)
    devloader = get_data_loader(data_path=opt.data,
                                        data_type='validate',
                                        batch_size=opt.batch_size,
                                        item_embedding_matrix=opt.model.item_embedding)
    testloader = get_data_loader(data_path=opt.data,
                                       data_type='test',
                                       batch_size=opt.batch_size,
                                       item_embedding_matrix=opt.model.item_embedding)
    
    return trainloader, devloader, testloader