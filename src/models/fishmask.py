import os
import torch


def fishmask_plugin_on_init(pl_module):
    if pl_module.config.fishmask_mode == "apply":
        print(f"Load gradient mask from {pl_module.config.fishmask_path}")
        mask_dict = torch.load(pl_module.config.fishmask_path)
        for param_name, param in pl_module.model.named_parameters():
            param.stored_mask = mask_dict[param_name].to("cuda")


def fishmask_plugin_on_optimizer_step(pl_module):
    if pl_module.config.fishmask_mode == "create":
        for name, param in pl_module.model.named_parameters():
            if not hasattr(param, "stored_grad"):
                param.stored_grad = torch.zeros_like(param.data)
            param.stored_grad += torch.square(param.grad) / pl_module.config.num_shot
            param.grad.zero_()
    elif pl_module.config.fishmask_mode == "apply":
        for name, param in pl_module.model.named_parameters():
            param.grad.data *= param.stored_mask
    else:
        raise ValueError(f"Invalid fishmask_mode {pl_module.config.fishmask_mode}")


def fishmask_plugin_on_end(pl_module):
    if pl_module.config.fishmask_mode == "create":
        sizes = {}
        tensors = []
        all_params_size = 0
        for param_name, param in pl_module.model.named_parameters():
            sizes[param_name] = param.size()
            tensors.append(param.stored_grad.view(-1))
            all_params_size += param.numel()
        tensors = torch.cat(tensors, 0).to("cpu")
        keep_num = int(all_params_size * pl_module.config.fishmask_keep_ratio)
        assert keep_num > 0
        top_pos = torch.topk(tensors, keep_num)[1]
        masks = torch.zeros(tensors.shape, dtype=torch.bool)
        masks[top_pos] = True
        del tensors
        assert masks.long().sum() == len(top_pos)
        mask_dict = {}

        now_idx = 0
        for param_name, param_size in sizes.items():
            end_idx = now_idx + param_size.numel()
            mask_dict[param_name] = masks[now_idx:end_idx].reshape(param_size)
            now_idx = end_idx
        assert now_idx == len(masks)

        all_params_size = 0
        trainable_weight_size = 0
        for param_name, param_mask in mask_dict.items():
            trainable_weight_size += param_mask.long().sum().item()
            all_params_size += param_mask.numel()

        print(f"Trainable parameters: {(trainable_weight_size) / all_params_size * 100:.3f} %")
        fishmask_path = os.path.join(pl_module.config.exp_dir, "mask.bin")
        torch.save(mask_dict, fishmask_path)
        print(f"Save gradient mask to {fishmask_path}")
