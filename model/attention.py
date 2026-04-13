import torch
import torch.nn as nn
import torch.nn.functional as F
from .rope import RotaryEmbedding, apply_rope

import math

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model, n_heads, max_seq_len, dropout_rate=0.1):
        super().__init__()
        self.n_heads = n_heads
        
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.head_dim = d_model // n_heads

        # Positional embedding RoPE
        self.rope = RotaryEmbedding(self.head_dim)

        # Attantion layer
        self.qkv = nn.Linear(d_model, 3 * d_model)
        # Output layer
        self.out = nn.Linear(d_model, d_model)
        # Dropout layer 
        self.attn_drop = dropout_rate  # nn.Dropout(p=dropout_rate)
        self.dropout = nn.Dropout(p=dropout_rate)

        # self.register_buffer(
        #     "mask", 
        #     torch.tril(torch.ones(max_seq_len, max_seq_len)).bool()
        # )

    def forward(self, x):
        batch_size, seq_len, d_model = x.shape
        qkv = self.qkv(x) # (batch, seq_len, 3*d_model)

        # Splits the last dimension into (n_heads, head_dim) for (q, k, v)
        qkv = qkv.view(batch_size, seq_len, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)  # (batch, seq_len, n_heads, head_dim)

        # Reshape for multi-head attention.
        q = q.transpose(1, 2)  # (batch, n_heads, seq_len, head_dim)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Apply RoPE 
        cos, sin = self.rope(seq_len, x.dtype)
        q, k = apply_rope(q, k, cos, sin)
        
        # # Dot-product attention logits  (batch, n_heads, seq_len, seq_len)
        # logits = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        # # Apply causal (look-ahead) mask.  
        # mask = self.mask[:seq_len, :seq_len]
        # logits_masked = logits.masked_fill(~mask, -torch.inf) 

        # # Apply softmax to get attention weights.
        # attention_weights = torch.softmax(logits_masked, dim=-1)

        # # Apply attention dropout
        # attention_weights = self.attn_drop(attention_weights)

        # # Apply attention weights to values.
        # attn_out = attention_weights @ v  # (batch, n_heads, seq_len, head_dim)

        attn_out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=None,
            dropout_p=self.attn_drop if self.training else 0.0,
            is_causal=True
        )

        # Concatenate heads and transpose back to (batch, seq_len, d_model)  
        attn_out = attn_out.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)

        # output projection (mix heads)
        out = self.out(attn_out)
        
        # Apply dropout
        out = self.dropout(out)

        return out 