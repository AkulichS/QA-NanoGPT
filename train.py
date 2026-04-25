import json
import torch
import hydra
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

from data.pretrain.lm_dataset import LMDataset
from tokenizer import BPETokenizer
from training import Trainer, build_model, build_optimizer, build_scheduler


@hydra.main(version_base="1.3", config_path="configs", config_name="gpt_L12_finetune")
def main(cfg):
    
    # --- device ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # --- tokenizer ---
    bpe_tokenizer = BPETokenizer().from_file(cfg.tokenizer.save_dir)

    # --- dataset ---
    train_ds = LMDataset(
        path=cfg.stages.data.train_path,
        block_size=cfg.models.model.max_seq_len,
        dtype=np.uint16
    )

    valid_ds = LMDataset(
        path=cfg.stages.data.valid_path,
        block_size=cfg.models.model.max_seq_len,
        dtype=np.uint16
    )

    # --- dataloader ---
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.stages.train.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=False
    )

    valid_loader = DataLoader(
        valid_ds,
        batch_size=cfg.stages.train.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=False
    )

    # --- model ---
    model = build_model(cfg, device)

     # --- optimizer ---
    optimizer = build_optimizer(model, cfg.stages.optim.lr)

    # --- scheduler ---
    scheduler = build_scheduler(optimizer, cfg.stages.train.total_steps)

    # --- log writer ---
    run_name = cfg.log.name + "_" + datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    writer = SummaryWriter(log_dir=f"{cfg.log.dir}/{run_name}")

    # --- training ---
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        train_loader=train_loader,
        valid_loader=valid_loader,
        device=device,
        writer=writer,
        grad_accum_steps=cfg.stages.train.grad_accum_steps,
        early_stopping=cfg.stages.train.early_stopping,
    )

    trainer.train(cfg.stages.train.total_steps)
    

if __name__ == "__main__":
    main()