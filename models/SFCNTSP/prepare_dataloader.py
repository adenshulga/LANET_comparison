from .utils.data_loader import get_data_loader
from .utils.load_config import get_attribute

def prepare_dataloader(opt):
    """ Load data and prepare dataloader. """

        
    trainloader = get_data_loader(data_path=get_attribute('data_path'), data_type='train',
                                        batch_size=get_attribute('batch_size'), max_seq_length=get_attribute('max_seq_length'),
                                        num_items=get_attribute('num_items'), num_workers=4)

    devloader = get_data_loader(data_path=get_attribute('data_path'), data_type='validate',
                                      batch_size=get_attribute('batch_size'), max_seq_length=get_attribute('max_seq_length'),
                                      num_items=get_attribute('num_items'), num_workers=4)

    testloader = get_data_loader(data_path=get_attribute('data_path'), data_type='test',
                                       batch_size=get_attribute('batch_size'), max_seq_length=get_attribute('max_seq_length'),
                                       num_items=get_attribute('num_items'), num_workers=4)    
    
    return trainloader, devloader, testloader