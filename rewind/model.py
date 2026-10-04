"""Sequence model for click prediction.

The design follows the Behavior Sequence Transformer (Chen et al., 2019, "Behavior Sequence
Transformer for E-commerce Recommendation in Alibaba"): embed the user's past actions, append the
candidate item as the final token, run a small transformer, and read the click prediction off that
final token. Nothing here is novel on purpose; the project's contribution is the comparison.

Input layout, for a batch of B examples with history length L:

    hist_items   (B, L)  item IDs, oldest -> newest, LEFT-padded with 0
    hist_actions (B, L)  action type per step (1=view, 2=cart, 3=fav, 4=buy), 0 = padding
    target_item  (B,)    the candidate item we want a click probability for

Left padding keeps the newest action at a fixed position (L-1) and the target at L, so position
embeddings mean "how many steps before now", whatever the user's history length.
"""

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

PAD = 0


@dataclass
class SeqConfig:
    n_items: int = 2**18  # item IDs are hashed into this many buckets
    n_actions: int = 4  # view, cart, fav, buy (0 is reserved for padding)
    max_len: int = 64  # history length L; the target adds one more token
    d_model: int = 64  # width of every token vector
    n_heads: int = 2
    n_layers: int = 2
    dropout: float = 0.1


class Block(nn.Module):
    """One transformer layer: attention (tokens look at each other), then a per-token MLP.

    Written out by hand rather than using nn.TransformerEncoderLayer so the attention call is in
    one visible place; a later experiment swaps it for other implementations.
    """

    def __init__(self, cfg: SeqConfig):
        super().__init__()
        self.n_heads = cfg.n_heads
        self.norm1 = nn.LayerNorm(cfg.d_model)
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model)
        self.proj = nn.Linear(cfg.d_model, cfg.d_model)
        self.norm2 = nn.LayerNorm(cfg.d_model)
        self.mlp = nn.Sequential(
            nn.Linear(cfg.d_model, 4 * cfg.d_model),
            nn.GELU(),
            nn.Linear(4 * cfg.d_model, cfg.d_model),
        )
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor, allowed: torch.Tensor) -> torch.Tensor:
        B, T, D = x.shape
        q, k, v = self.qkv(self.norm1(x)).split(D, dim=-1)
        # (B, T, D) -> (B, heads, T, D/heads): each head attends independently.
        q, k, v = (t.view(B, T, self.n_heads, D // self.n_heads).transpose(1, 2) for t in (q, k, v))
        p = self.dropout.p if self.training else 0.0
        att = F.scaled_dot_product_attention(q, k, v, attn_mask=allowed, dropout_p=p)
        att = att.transpose(1, 2).reshape(B, T, D)
        x = x + self.dropout(self.proj(att))
        x = x + self.dropout(self.mlp(self.norm2(x)))
        return x


class SequenceCTR(nn.Module):
    def __init__(self, cfg: SeqConfig):
        super().__init__()
        self.cfg = cfg
        self.item_emb = nn.Embedding(cfg.n_items, cfg.d_model, padding_idx=PAD)
        self.action_emb = nn.Embedding(cfg.n_actions + 1, cfg.d_model, padding_idx=PAD)
        self.pos_emb = nn.Embedding(cfg.max_len + 1, cfg.d_model)
        self.blocks = nn.ModuleList(Block(cfg) for _ in range(cfg.n_layers))
        self.norm = nn.LayerNorm(cfg.d_model)
        self.head = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_model), nn.ReLU(), nn.Linear(cfg.d_model, 1)
        )

    def forward(
        self, hist_items: torch.Tensor, hist_actions: torch.Tensor, target_item: torch.Tensor
    ) -> torch.Tensor:
        """Returns click logits, shape (B,). Apply sigmoid to get probabilities."""
        B, L = hist_items.shape
        T = L + 1

        hist = self.item_emb(hist_items) + self.action_emb(hist_actions)
        # The target has no action embedding: its action is exactly what we are predicting.
        target = self.item_emb(target_item).unsqueeze(1)
        x = torch.cat([hist, target], dim=1)
        x = x + self.pos_emb(torch.arange(T, device=x.device))

        x = x.masked_fill(_pad_mask(hist_items).unsqueeze(-1), 0.0)
        allowed = _attention_mask(hist_items)
        for block in self.blocks:
            x = block(x, allowed)

        return self.head(self.norm(x[:, -1])).squeeze(-1)


def _pad_mask(hist_items: torch.Tensor) -> torch.Tensor:
    """(B, T) True where the token is padding. The target token is never padding."""
    B = hist_items.shape[0]
    target_col = torch.zeros(B, 1, dtype=torch.bool, device=hist_items.device)
    return torch.cat([hist_items == PAD, target_col], dim=1)


def _attention_mask(hist_items: torch.Tensor) -> torch.Tensor:
    """(B, 1, T, T) boolean, True where query position i may look at key position j.

    Two rules combined:
      causal  - a token sees only itself and earlier tokens, never later ones
      padding - nobody looks at padding
    Every token may always see itself, so a padding row is never fully masked; a fully masked row
    makes softmax return NaN, and that NaN would leak into real tokens in the next layer.
    """
    T = hist_items.shape[1] + 1
    device = hist_items.device
    causal = torch.ones(T, T, dtype=torch.bool, device=device).tril()
    key_is_real = ~_pad_mask(hist_items)  # (B, T)
    self_ok = torch.eye(T, dtype=torch.bool, device=device)
    allowed = causal & (key_is_real[:, None, :] | self_ok)
    return allowed.unsqueeze(1)


def count_params(model: nn.Module) -> dict[str, int]:
    """Parameter counts, split so embedding tables (which dominate) are visible separately."""
    emb = sum(p.numel() for m in model.modules() if isinstance(m, nn.Embedding) for p in m.parameters())
    total = sum(p.numel() for p in model.parameters())
    return {"total": total, "embedding": emb, "dense": total - emb}
