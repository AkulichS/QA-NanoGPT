import torch


def build_optimizer(model, cfg_optim):
    return torch.optim.AdamW(
        model.parameters(),
        lr=cfg_optim.lr,
        betas=cfg_optim.betas,
        weight_decay=cfg_optim.weight_decay
    )