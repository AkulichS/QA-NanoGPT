import torch
from model import NanoGPT   


def build_model(cfg, device: torch.device) -> NanoGPT:
    """
    Build NanoGPT from config, optionally loading weights from a checkpoint.
    Two modes:
      - resume_from is None  → fresh model using cfg.models params
      - resume_from is a path → load weights from that checkpoint
    """

    model = NanoGPT(
        vocab_size=cfg.tokenizer.vocab_size,
        d_model=cfg.models.d_model,
        n_layers=cfg.models.n_layers,
        n_heads=cfg.models.n_heads,
        d_ff=cfg.models.d_ff,
        max_seq_len=cfg.models.max_seq_len,
    )

    checkpoint_path = cfg.checkpoint.get("resume_from", None)

    if checkpoint_path is not None:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        
    return model.to(device)
