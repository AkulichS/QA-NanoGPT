import torch


def build_scheduler(optimizer, total_steps):

    warmup_steps = int(0.01 * total_steps)
    decay_steps = total_steps - warmup_steps

    warmup = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=1/4,
        end_factor=1.0,
        total_iters=warmup_steps
    )

    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=decay_steps,
        eta_min=1e-5
    )

    return torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[warmup, cosine],
        milestones=[warmup_steps]
    )