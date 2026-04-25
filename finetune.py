# fine_tuning.py

import json
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

from data.finetune.qa_dataset import QADataset, qa_collate_fn
from tokenizer import BPETokenizer
from training import Trainer, build_optimizer, build_scheduler
from model import NanoGPT

MAX_STEPS = 500

def main():
    # --- load dataset ---
    # with open("data/qa_datasets/qa_samples.json", "r", encoding="utf-8") as f:
    #     qa_dataset = json.load(f)

    with open("data/qa_datasets/qa_samples_val.json", "r", encoding="utf-8") as f:
        qa_dataset_val = json.load(f)

    # --- tokenizer ---
    bpe_tokenizer = BPETokenizer().from_file("tokenizer/bpe_tokenizer_v2/tokenizer.json")

    qa_dataset = qa_dataset_val

    # --- dataset ---
    train_ds = QADataset(
        data=qa_dataset,
        tokenizer=bpe_tokenizer,
        max_seq_len=1024
    )

    valid_ds = QADataset(
        data=qa_dataset_val,
        tokenizer=bpe_tokenizer,
        max_seq_len=1024
    )

    pad_id = bpe_tokenizer.token_to_id("<PAD>")

    # --- dataloader ---
    train_loader = DataLoader(
        train_ds,
        batch_size=2,
        shuffle=True,
        collate_fn=lambda x: qa_collate_fn(x, pad_id)
    )

    valid_loader = DataLoader(
        valid_ds,
        batch_size=2,
        shuffle=True,
        collate_fn=lambda x: qa_collate_fn(x, pad_id)
    )

    # --- log writer ---
    run_name = "run_" + datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    writer = SummaryWriter(log_dir=f"utils/log/tensorboard/{run_name}")

    # --- device ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # --- model ---
    # model = NanoGPT(bpe_tokenizer.get_vocab_size(), 384, 8, 6, 4*384, 1024).to(device)
    checkpoint = torch.load("utils/checkpoints/best_checkpoint4.pt")   # ckpt_pretrained_gpt.pt")  best_checkpoint4.pt
    model = NanoGPT(
        checkpoint["config"]["vocab_size"],
        checkpoint["config"]["d_model"],
        checkpoint["config"]["n_layers"],
        checkpoint["config"]["n_heads"],
        checkpoint["config"]["d_ff"],
        checkpoint["config"]["max_seq_len"],
    ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])

     # --- optimizer ---
    optimizer = build_optimizer(model, lr=3e-5)

    # --- scheduler ---
    scheduler = build_scheduler(optimizer, MAX_STEPS)

    # --- training ---
    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        scheduler=scheduler,
        train_loader=train_loader,
        valid_loader=valid_loader,
        device=device,
        writer=writer,
        grad_accum_steps=32,
        early_stopping=50,
    )

    trainer.train(MAX_STEPS)


if __name__ == "__main__":
    main()