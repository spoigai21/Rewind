"""Phase 3 sequence model: the Phase 0 transformer with inputs built for this data.

Behavior Sequence Transformer (Chen et al., 2019): embed the user's past impressions, append the
candidate ad as the last token, run a small transformer, and read the prediction off the
candidate's output, combined with the same side inputs the tabular models get.

One history token = the sum of
    proj(concat of 16-wide embeddings for ad, category, brand, campaign, advertiser, placement)
    + action embedding     shown-not-clicked / clicked     (the candidate gets its own value)
    + time-gap embedding   how long before the question     (13 buckets; none for the candidate)
    + position embedding   steps back from the candidate    (0 = candidate, 1 = newest, ...)

Positions count BACKWARDS from the candidate, so dropping leading padding (SequenceData trim) does
not change any token's position, and so changes no prediction.

The ID embedding tables are the same vocabularies at the same width (16) as DCN's, so the two
families have identical embedding tables by construction. The dense part is sized to DCN's
(`size_head`). The attention block and mask are the tested Phase 0 code (rewind/model.py).
"""

from dataclasses import dataclass, field

import torch
from torch import nn

from rewind.data.sequences import PROFILE_COLS
from rewind.model import Block, SeqConfig, attention_mask
from rewind.sequence.data import N_GAP_BUCKETS, TOKEN_FIELDS

CANDIDATE_ACTION = 3  # action ids: 0 pad, 1 shown-not-clicked, 2 clicked, 3 candidate


@dataclass
class SequenceModelConfig:
    cardinalities: dict[str, int]  # all ID fields: TOKEN_FIELDS + PROFILE_COLS
    n_context: int = 7
    emb_dim: int = 16
    d_model: int = 64
    n_heads: int = 2
    n_layers: int = 2
    head_hidden: int = 256
    max_len: int = 64
    extra: dict = field(default_factory=dict)  # free-form notes saved with checkpoints


class SequenceModel(nn.Module):
    def __init__(self, cfg: SequenceModelConfig):
        super().__init__()
        self.cfg = cfg
        d = cfg.d_model
        self.field_emb = nn.ModuleDict(
            {f: nn.Embedding(cfg.cardinalities[f], cfg.emb_dim) for f in TOKEN_FIELDS + PROFILE_COLS}
        )
        for e in self.field_emb.values():
            nn.init.normal_(e.weight, std=0.01)  # same initialisation as the DCN embeddings
        self.token_proj = nn.Linear(len(TOKEN_FIELDS) * cfg.emb_dim, d)
        self.action_emb = nn.Embedding(CANDIDATE_ACTION + 1, d)
        self.gap_emb = nn.Embedding(N_GAP_BUCKETS + 1, d)
        self.pos_emb = nn.Embedding(cfg.max_len + 1, d)
        block_cfg = SeqConfig(d_model=d, n_heads=cfg.n_heads, n_layers=cfg.n_layers, dropout=0.0)
        self.blocks = nn.ModuleList(Block(block_cfg) for _ in range(cfg.n_layers))
        self.norm = nn.LayerNorm(d)
        head_in = d + len(PROFILE_COLS) * cfg.emb_dim + cfg.n_context
        h = cfg.head_hidden
        self.head = nn.Sequential(
            nn.Linear(head_in, h), nn.ReLU(), nn.Linear(h, h // 2), nn.ReLU(), nn.Linear(h // 2, 1)
        )

    def embed_ads(self, tokens: torch.Tensor) -> torch.Tensor:
        """(..., 6) ID codes -> (..., d_model)."""
        parts = [self.field_emb[f](tokens[..., i]) for i, f in enumerate(TOKEN_FIELDS)]
        return self.token_proj(torch.cat(parts, dim=-1))

    def forward(self, b: dict[str, torch.Tensor]) -> torch.Tensor:
        """Click logits, shape (B,). Apply sigmoid for probabilities."""
        real_hist = b["real"]
        B, L = real_hist.shape
        if L > self.cfg.max_len:
            raise ValueError(f"history length {L} exceeds max_len {self.cfg.max_len}")

        hist = self.embed_ads(b["hist_tokens"]) + self.action_emb(b["hist_action"]) + self.gap_emb(b["hist_gap"])
        cand = self.embed_ads(b["cand_tokens"]) + self.action_emb.weight[CANDIDATE_ACTION]
        x = torch.cat([hist, cand.unsqueeze(1)], dim=1)
        steps_back = torch.arange(L, -1, -1, device=x.device)  # L, L-1, ..., 1, 0 (candidate)
        x = x + self.pos_emb(steps_back)

        real = torch.cat([real_hist, torch.ones(B, 1, dtype=torch.bool, device=x.device)], dim=1)
        x = x * real.unsqueeze(-1)  # padding tokens carry nothing
        allowed = attention_mask(real)
        for block in self.blocks:
            x = block(x, allowed)
        seq_out = self.norm(x[:, -1])

        profile = torch.cat([self.field_emb[f](b["profile"][:, i]) for i, f in enumerate(PROFILE_COLS)], dim=-1)
        return self.head(torch.cat([seq_out, profile, b["context"]], dim=-1)).squeeze(-1)


def param_counts(model: nn.Module) -> dict[str, int]:
    """Split as in Phase 2. `id_embedding` counts only the ID tables shared in design with DCN;
    the small action / time-gap / position tables are reported separately as `aux_embedding`."""
    total = sum(p.numel() for p in model.parameters())
    emb = sum(p.numel() for m in model.modules() if isinstance(m, nn.Embedding) for p in m.parameters())
    out = {"total": total, "embedding": emb, "dense": total - emb}
    if isinstance(model, SequenceModel):
        ids = sum(p.numel() for p in model.field_emb.parameters())
        out.update(id_embedding=ids, aux_embedding=emb - ids)
    return out


def size_head(cfg: SequenceModelConfig, target_dense: int) -> SequenceModelConfig:
    """Pick head_hidden so the model's dense parameter count is as close as possible to
    target_dense. Dense size does not depend on the vocabularies, so a tiny-vocabulary copy is
    counted instead of allocating the real tables."""
    tiny = {k: 2 for k in cfg.cardinalities}

    def dense(h: int) -> int:
        m = SequenceModel(SequenceModelConfig(**{**vars(cfg), "cardinalities": tiny, "head_hidden": h}))
        return param_counts(m)["dense"]

    lo, hi = 2, 8192
    while hi - lo > 2:  # dense(h) increases with h
        mid = (lo + hi) // 2 // 2 * 2  # keep h even so h // 2 is exact
        lo, hi = (mid, hi) if dense(mid) < target_dense else (lo, mid)
    best = min(range(lo, hi + 1, 2), key=lambda h: abs(dense(h) - target_dense))
    return SequenceModelConfig(**{**vars(cfg), "head_hidden": best})
