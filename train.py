import torch
from ..model.gpt import NanoGPT
from ..tokenization.bpe_tokenizer import BPETokenizer
from ..training.trainer import Trainer
from ..training.optimizer import build_optimizer
from ..training.scheduler import build_scheduler


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

bpe_tokenizer = BPETokenizer().from_file("../tokenization/bpe_tokenizer_v3/tokenizer.json")

model = NanoGPT(bpe_tokenizer.get_vocab_size(), 384, 8, 6, 4*384, 1024).to(device)

optimizer = build_optimizer(model)
scheduler = build_scheduler(optimizer, total_steps)

trainer = Trainer(
    model=model,
    optimizer=optimizer,
    scheduler=scheduler,
    train_loader=train_loader,
    valid_loader=valid_loader,
    device=device,
    writer=writer,
    grad_accum_steps=16,
    task="lm"
)

trainer.train(NUM_EPOCHS)