from .utils.load_config import get_attribute

from .model.SFCNTSP import SFCNTSP

def create_model(opt):
    # print(f"{get_attribute('data')}/{get_attribute('save_model_folder')}")

    model = SFCNTSP(num_items=get_attribute('num_items'), max_seq_length=get_attribute('max_seq_length'),
                    embedding_channels=get_attribute('embedding_channels'), dropout=get_attribute('dropout'),
                    bias=get_attribute('bias'), alpha=get_attribute('alpha'), beta=get_attribute('beta'))
    return model