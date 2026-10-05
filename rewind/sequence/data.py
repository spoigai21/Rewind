"""Phase 3 data feeder: builds sequence-model batches directly on the GPU.

Everything one history item needs is packed into a few compact tables, moved to the device once
(~2.3 GB for all 26.5M impressions), and every batch is then a handful of indexing operations on
the device instead of thousands of CPU reads:

    tokens   (N, 6) int32   ad, category, brand, campaign, advertiser, placement codes
    clk      (N,)   int8    clicked? (history items only; a question's own label is never an input)
    t        (N,)   int32   seconds since the first impression
    start    (N,)   int32   Phase 1 bookmark: user's first row
    end      (N,)   int32   Phase 1 bookmark: first row of this row's own second
    profile  (N, 8) int16   user profile codes
    context  (N, 7) float32 the shared context features from Phase 2 (ctx_*: global popularity
                            of the ad / category / advertiser so far, and price)

The tabular history totals (hist_*) are deliberately NOT loaded: the sequence model gets the user's
own past only as the ordered timeline. That is the one variable the comparison tests (DESIGN.md).

Batch layout, history length L, oldest -> newest, LEFT-padded:

    hist_tokens (B, L, 6)   0 where padding
    hist_action (B, L)      0 pad, 1 shown-not-clicked, 2 clicked
    hist_gap    (B, L)      0 pad, 1.. time-gap bucket (how long before the question)
    real        (B, L)      True for real history items
    cand_tokens (B, 6)      the ad being scored
    profile, context, label

`trim=True` drops leading columns that are padding in every row of the batch. The model counts
positions from the candidate backwards, so trimming changes no prediction, only the compute.
"""

import json
from pathlib import Path

import numpy as np
import torch

from rewind.data.sequences import PROFILE_COLS, SPLITS

TOKEN_FIELDS = ["adgroup_id", "cate_id", "brand", "campaign_id", "customer", "pid"]

# Time-gap bucket edges in seconds: 1 min, 5 min, 15 min, 30 min, 1 h, 2 h, 4 h, 8 h, 12 h, 1 d,
# 2 d, 4 d. Gaps are always > 0 (history is strictly earlier). Bucket k+1 holds
# edges[k-1] <= gap < edges[k]; bucket 1 is under a minute, the last is 4 days or more.
GAP_EDGES = [60, 300, 900, 1800, 3600, 7200, 14400, 28800, 43200, 86400, 172800, 345600]
N_GAP_BUCKETS = len(GAP_EDGES) + 1  # real buckets are 1..N_GAP_BUCKETS; 0 = padding / candidate


class SequenceData:
    def __init__(self, device: torch.device, seq_dir: Path = Path("data/sequences"),
                 feature_dir: Path = Path("data/features"), user_fraction: float = 1.0):
        """user_fraction < 1 loads only the first share of USERS (rows are stored user by user), each
        with their complete history, so every bookmark stays valid. For laptop trials only."""
        meta = json.loads((feature_dir / "meta.json").read_text())
        names = meta["num_features"]
        self.context_names = meta["num_groups"]["context"]
        ctx_idx = [names.index(n) for n in self.context_names]
        self.cardinalities = dict(zip(meta["cat_features"], meta["cat_cardinalities"]))

        cut = None
        if user_fraction < 1.0:
            starts = np.load(seq_dir / "user_start.npy", mmap_mode="r")
            k = int(len(starts) * user_fraction)
            cut = int(starts[k])  # first row of the user at that point: whole users only

        def arr(name):
            return np.load(seq_dir / f"{name}.npy", mmap_mode="r")[:cut]

        t = np.asarray(arr("time_stamp"))
        self.split = np.asarray(arr("split"))
        self.prior = np.asarray(arr("hist_end")) - np.asarray(arr("user_start"))
        self.user = np.asarray(arr("user"))
        num = np.load(feature_dir / "num.npy", mmap_mode="r")[:cut]

        def dev(a, dtype):
            # np.array(...) always COPIES. Without it, on the CPU the tensor would share memory with
            # the read-only memory-mapped file on disk (BUGLOG #12).
            return torch.from_numpy(np.array(a)).to(device=device, dtype=dtype)

        self.device = device
        self.tokens = dev(np.stack([arr(c) for c in TOKEN_FIELDS], axis=1), torch.int32)
        self.clk = dev(arr("clk"), torch.int8)
        self.t = dev(t - t.min(), torch.int32)
        self.start = dev(arr("user_start"), torch.int32)
        self.end = dev(arr("hist_end"), torch.int32)
        self.profile = dev(np.stack([arr(c) for c in PROFILE_COLS], axis=1), torch.int16)
        self.context = dev(num[:, ctx_idx], torch.float32)
        self.gap_edges = torch.tensor(GAP_EDGES, device=device, dtype=torch.int32)

    def rows(self, split: str) -> np.ndarray:
        return np.flatnonzero(self.split == SPLITS[split])

    def batch(self, rows: torch.Tensor, length: int, trim: bool = True) -> dict[str, torch.Tensor]:
        """Model inputs for question rows (a 1-D long tensor on the device)."""
        end = self.end[rows].long()
        start = self.start[rows].long()
        pos = end[:, None] - length + torch.arange(length, device=self.device)[None, :]
        real = pos >= start[:, None]
        if trim:
            keep = int(real.any(0).nonzero().min()) if real.any() else length - 1
            pos, real = pos[:, keep:], real[:, keep:]
        safe = torch.where(real, pos, torch.zeros_like(pos))

        hist_tokens = self.tokens[safe] * real[..., None]
        hist_action = torch.where(real, self.clk[safe].long() + 1, 0)
        gap = self.t[rows][:, None] - self.t[safe]
        hist_gap = torch.where(real, torch.bucketize(gap, self.gap_edges, right=True) + 1, 0)
        return {
            "hist_tokens": hist_tokens.long(),
            "hist_action": hist_action,
            "hist_gap": hist_gap,
            "real": real,
            "cand_tokens": self.tokens[rows].long(),
            "profile": self.profile[rows].long(),
            "context": self.context[rows],
            "label": self.clk[rows].float(),
        }
