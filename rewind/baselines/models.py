"""Tabular (feature-crossing) baselines. Standard designs, implemented plainly:

    LR      logistic regression: one learned weight per categorical value, one per numeric
            feature, summed. No feature interactions.
    DeepFM  Guo et al., 2017, "DeepFM: A Factorization-Machine based Neural Network for CTR
            Prediction". LR + a factorization machine (every pair of features interacts through
            the dot product of their embeddings) + an MLP over all embeddings, summed.
    DCN     Wang et al., 2021, "DCN V2: Improved Deep & Cross Network and Practical Lessons for
            Web-scale Learning to Rank Systems". Explicit cross layers x_{l+1} = x0 * (W x_l + b)
            + x_l, in parallel with an MLP, combined by a final linear layer.

All three take the same inputs: `cat` (B, n_cat) integer codes and `num` (B, n_num) standardized
numbers, and return click logits (B,).
"""

import torch
from torch import nn


class LR(nn.Module):
    def __init__(self, cardinalities: list[int], n_num: int, **_):
        super().__init__()
        self.weights = nn.ModuleList(nn.Embedding(c, 1) for c in cardinalities)
        self.linear = nn.Linear(n_num, 1)
        for w in self.weights:
            nn.init.zeros_(w.weight)

    def forward(self, cat: torch.Tensor, num: torch.Tensor) -> torch.Tensor:
        first_order = sum(w(cat[:, i]) for i, w in enumerate(self.weights))
        return (first_order + self.linear(num)).squeeze(-1)


class FieldEmbeddings(nn.Module):
    """One embedding per categorical field, plus one learned vector per numeric field scaled by
    its value, so numeric features can take part in feature crossing too. Output (B, fields, dim)."""

    def __init__(self, cardinalities: list[int], n_num: int, dim: int):
        super().__init__()
        self.cat = nn.ModuleList(nn.Embedding(c, dim) for c in cardinalities)
        self.num = nn.Parameter(torch.randn(n_num, dim) * 0.01)
        for e in self.cat:
            nn.init.normal_(e.weight, std=0.01)

    def forward(self, cat: torch.Tensor, num: torch.Tensor) -> torch.Tensor:
        c = torch.stack([e(cat[:, i]) for i, e in enumerate(self.cat)], dim=1)
        n = num.unsqueeze(-1) * self.num.unsqueeze(0)
        return torch.cat([c, n], dim=1)


def mlp(d_in: int, hidden: list[int], dropout: float) -> nn.Sequential:
    layers: list[nn.Module] = []
    for h in hidden:
        layers += [nn.Linear(d_in, h), nn.ReLU(), nn.Dropout(dropout)]
        d_in = h
    return nn.Sequential(*layers)


class DeepFM(nn.Module):
    def __init__(self, cardinalities: list[int], n_num: int, dim: int = 16,
                 hidden: tuple[int, ...] = (256, 128), dropout: float = 0.0, **_):
        super().__init__()
        self.lr = LR(cardinalities, n_num)
        self.emb = FieldEmbeddings(cardinalities, n_num, dim)
        n_fields = len(cardinalities) + n_num
        self.deep = mlp(n_fields * dim, list(hidden), dropout)
        self.out = nn.Linear(hidden[-1], 1)

    def forward(self, cat: torch.Tensor, num: torch.Tensor) -> torch.Tensor:
        e = self.emb(cat, num)  # (B, F, k)
        # Sum over all pairs i<j of <e_i, e_j>, computed in O(F*k): ((sum e)^2 - sum e^2) / 2.
        fm = 0.5 * (e.sum(1).pow(2) - e.pow(2).sum(1)).sum(-1)
        deep = self.out(self.deep(e.flatten(1))).squeeze(-1)
        return self.lr(cat, num) + fm + deep


class DCN(nn.Module):
    def __init__(self, cardinalities: list[int], n_num: int, dim: int = 16, n_cross: int = 3,
                 hidden: tuple[int, ...] = (256, 128), dropout: float = 0.0, **_):
        super().__init__()
        self.emb = FieldEmbeddings(cardinalities, n_num, dim)
        d = (len(cardinalities) + n_num) * dim
        self.cross = nn.ModuleList(nn.Linear(d, d) for _ in range(n_cross))
        self.deep = mlp(d, list(hidden), dropout)
        self.out = nn.Linear(d + hidden[-1], 1)

    def forward(self, cat: torch.Tensor, num: torch.Tensor) -> torch.Tensor:
        x0 = self.emb(cat, num).flatten(1)
        x = x0
        for layer in self.cross:
            x = x0 * layer(x) + x
        return self.out(torch.cat([x, self.deep(x0)], dim=1)).squeeze(-1)


MODELS = {"lr": LR, "deepfm": DeepFM, "dcn": DCN}


def param_counts(model: nn.Module) -> dict[str, int]:
    """Embedding-table parameters reported separately from dense ones: the tables dominate, and
    Phase 3 must match both (PREDICTION.md, section 4)."""
    emb = sum(p.numel() for m in model.modules() if isinstance(m, nn.Embedding) for p in m.parameters())
    total = sum(p.numel() for p in model.parameters())
    return {"total": total, "embedding": emb, "dense": total - emb}
