import torch
from model import NanoGPT   


def build_model(cfg, device: torch.device) -> NanoGPT:
    """
    Build NanoGPT from config, optionally loading weights from a checkpoint.
    Two modes:
      - resume_from is None  → fresh model using cfg.models.model params
      - resume_from is a path → load arch + weights from that checkpoint
    """
    resume_path = cfg.checkpoint.get("resume_from", None)

    if resume_path is not None:
        return _load_from_checkpoint(resume_path, device)
    else:
        return _build_fresh(cfg, device)


def _build_fresh(cfg, device: torch.device) -> NanoGPT:
    m = cfg.models.model
    model = NanoGPT(
        vocab_size=cfg.tokenizer.vocab_size,
        d_model=m.d_model,
        n_layers=m.n_layers,
        n_heads=m.n_heads,
        d_ff=m.d_ff,
        max_seq_len=m.max_seq_len,
    )
    return model.to(device)


def _load_from_checkpoint(path: str, device: torch.device) -> NanoGPT:
    checkpoint = torch.load(path, map_location=device)
    ckp = checkpoint["config"]

    model = NanoGPT(
        vocab_size=ckp["vocab_size"],
        d_model=ckp["d_model"],
        n_layers=ckp["n_layers"],
        n_heads=ckp["n_heads"],
        d_ff=ckp["d_ff"],
        max_seq_len=ckp["max_seq_len"],
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    
    return model.to(device)