import torch.nn as nn
from .block import TransformerBlock


class NanoGPT(nn.Module):
    def __init__(
        self,
        vocab_size,
        d_model,
        n_layers,
        n_heads,
        d_ff,
        max_seq_len,
        return_hidden=False
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.return_hidden = return_hidden
        self.token_emb = nn.Embedding(self.vocab_size, d_model)

        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, d_ff, max_seq_len)
            for _ in range(n_layers)
        ])

        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, self.vocab_size, bias=False)
        self.lm_head.weight = self.token_emb.weight # weight tying

    def forward(self, input_ids, labels):
        x = self.token_emb(input_ids)  # (batch, seq_len, d_model)

        for block in self.blocks:
            x = block(x)

        x = self.ln_f(x)
        logits = self.lm_head(x)  # (batch, seq_len, vocab_size)

        if self.return_hidden:
            return logits, x

        loss = None
        shift_logits = logits[:, :-1, :].contiguous()  # (batch, seq_len, vocab_size)
        shift_labels = labels[:, 1:].contiguous()      # (batch, seq_len)

        loss = nn.functional.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=-100
            )

        return logits, loss