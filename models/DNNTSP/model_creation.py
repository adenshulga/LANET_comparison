from .utils.load_config import get_attribute
from .model.temporal_set_prediction import temporal_set_prediction

def create_model(opt):
    # print(f"{get_attribute('data')}/{get_attribute('save_model_folder')}")

    model = temporal_set_prediction(items_total=get_attribute('items_total'),
                                    item_embedding_dim=get_attribute('item_embed_dim'))

    return model