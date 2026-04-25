import torch


def build_scheduler(optimizer, total_steps, eta_min_ratio=0.1):

    warmup_steps = int(0.01 * total_steps)
    decay_steps = total_steps - warmup_steps

    warmup = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=1/4,
        end_factor=1.0,
        total_iters=warmup_steps
    )

    eta_min = optimizer.param_groups[0]['lr'] * eta_min_ratio

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