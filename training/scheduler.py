import torch


def build_scheduler(optimizer, cfg_sched):

    warmup_steps = int(cfg_sched.warmup_ratio * cfg_sched.total_steps)
    decay_steps = cfg_sched.total_steps - warmup_steps

    warmup = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=cfg_sched.start_factor,
        end_factor=cfg_sched.end_factor,
        total_iters=warmup_steps
    )

    eta_min = optimizer.param_groups[0]['lr'] * cfg_sched.eta_min_ratio

    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=decay_steps,
        eta_min=eta_min
    )

    return torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[warmup, cosine],
        milestones=[warmup_steps]
    )