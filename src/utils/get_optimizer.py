import torch.optim as optim
from transformers import Adafactor
import re
from collections import defaultdict
from deepspeed.ops.adam import FusedAdam, DeepSpeedCPUAdam

from src.utils.Config import Config

def get_optimizer(model, config: Config):
    """
    Construct optimizer based on config

    :param model:
    :param config:
    :return:
    """
    optim_name = config.optimizer
    use_deepspeed: bool = config.use_deepspeed

    def param_name_to_group_name(param_name):
        if False:
            return ".".join(param_name.split(".")[:3])
            # only needed when the model has many trainable parameters, disabled in our expeirments
        else:
            return "."

    param_groups = defaultdict(lambda: {"params": []})
    trainable_param_names = set()
    for (param_name, param) in model.named_parameters():
        if re.fullmatch(config.trainable_param_names, param_name):
            param_groups[param_name_to_group_name(param_name)]["params"].append(param)
            trainable_param_names.add(param_name)
        else:
            param.requires_grad = False

    param_groups = param_groups.values()
    if optim_name.lower() == "adam":
        if use_deepspeed:
            if config.ds_offload_optimizer:
                optimizer = DeepSpeedCPUAdam(param_groups, lr=config.lr, adamw_mode=False)
            else:
                optimizer = FusedAdam(param_groups, lr=config.lr, adam_w_mode=False)
        else:
            optimizer = optim.Adam(param_groups, lr=config.lr)
    elif optim_name.lower() == "sgd":
        optimizer = optim.SGD(param_groups, lr=config.lr, weight_decay=config.weight_decay)
    elif optim_name.lower() == "adamw":
        optimizer = optim.AdamW(param_groups, lr=config.lr, weight_decay=config.weight_decay, eps=1e-8)
        if config.ds_offload_optimizer:
            optimizer = DeepSpeedCPUAdam(param_groups, lr=config.lr, weight_decay=config.weight_decay, adamw_mode=True)
        else:
            optimizer = FusedAdam(param_groups, lr=config.lr, weight_decay=config.weight_decay, adam_w_mode=True)
    elif optim_name.lower() == "adafactor":
        optimizer = Adafactor(
            param_groups,
            lr=config.lr,
            weight_decay=config.weight_decay,
            scale_parameter=config.scale_parameter,
            relative_step=False,
            warmup_init=False,
        )
    else:
        raise ValueError("Invalid Optimizer name %s" % optim_name)

    return optimizer, trainable_param_names
