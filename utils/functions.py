import torch
import numpy as np

def str_config(config):
    config_string = ""
    for tpl in config:
        tpl_str = "-".join(str(x) for x in tpl)
        config_string += tpl_str + "_"

    config_string = config_string.rstrip("_")
    return(config_string)



def count_parameters(model):
    return np.sum(np.prod(v.size()) for name, v in model.named_parameters() if "auxiliary" not in name)


def save(model, model_path):
    torch.save(model.state_dict(), model_path)
