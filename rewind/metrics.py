"""Evaluation metrics, reported together for every model.

    auc          P(a random click is scored above a random non-click). 0.5 = chance.
    gauc         AUC within each user, averaged weighted by the user's impressions; users whose
                 impressions are all clicks or all non-clicks have no AUC and are skipped.
    ne           normalized entropy: log loss divided by the log loss of always predicting the
                 evaluation set's own click rate. Below 1 = better than that constant.
    ece          expected calibration error over equal-count bins of the prediction: the
                 impression-weighted average gap between predicted and observed click rate.
    calibration  mean prediction / observed click rate. 1.0 = no overall over- or under-prediction.

Ranking quality (AUC, GAUC) and probability quality (NE, ECE, calibration) are different things:
a model can rank perfectly and still be 30% too high, which mis-prices every auction.
"""

import numpy as np
import polars as pl

EPS = 1e-7
ECE_BINS = 20


def auc(y: np.ndarray, p: np.ndarray) -> float:
    """Rank-sum (Mann-Whitney) AUC; tied scores get their average rank."""
    y = np.asarray(y, dtype=bool)
    n_pos, n_neg = int(y.sum()), int((~y).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = pl.Series(np.asarray(p, dtype=np.float64)).rank("average").to_numpy()
    return float((ranks[y].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def gauc(y: np.ndarray, p: np.ndarray, group: np.ndarray) -> float:
    df = pl.DataFrame({"g": group, "y": np.asarray(y, dtype=np.int64), "p": np.asarray(p, dtype=np.float64)})
    per = (
        df.with_columns(pl.col("p").rank("average").over("g").alias("r"))
        .group_by("g")
        .agg(
            pl.len().alias("n"),
            pl.col("y").sum().alias("pos"),
            pl.col("r").filter(pl.col("y") == 1).sum().alias("rpos"),
        )
        .with_columns((pl.col("n") - pl.col("pos")).alias("neg"))
        .filter((pl.col("pos") > 0) & (pl.col("neg") > 0))
        .with_columns(((pl.col("rpos") - pl.col("pos") * (pl.col("pos") + 1) / 2)
                       / (pl.col("pos") * pl.col("neg"))).alias("auc"))
    )
    if per.height == 0:
        return float("nan")
    return float((per["auc"] * per["n"]).sum() / per["n"].sum())


def log_loss(y: np.ndarray, p: np.ndarray) -> float:
    p = np.clip(np.asarray(p, dtype=np.float64), EPS, 1 - EPS)
    y = np.asarray(y, dtype=np.float64)
    return float(-(y * np.log(p) + (1 - y) * np.log(1 - p)).mean())


def ne(y: np.ndarray, p: np.ndarray) -> float:
    base = float(np.mean(y))
    return log_loss(y, p) / log_loss(y, np.full(len(y), base))


def ece(y: np.ndarray, p: np.ndarray, bins: int = ECE_BINS) -> float:
    """Equal-count bins: click predictions cluster near 0-10%, so equal-WIDTH bins would put almost
    everything in the first bin and hide miscalibration."""
    p = np.asarray(p, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    order = np.argsort(p, kind="stable")
    gap = 0.0
    for chunk in np.array_split(order, bins):
        if len(chunk):
            gap += len(chunk) * abs(p[chunk].mean() - y[chunk].mean())
    return float(gap / len(p))


def calibration(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.mean(p) / np.mean(y))


def evaluate(y: np.ndarray, p: np.ndarray, user: np.ndarray) -> dict[str, float]:
    return {
        "auc": auc(y, p),
        "gauc": gauc(y, p, user),
        "ne": ne(y, p),
        "ece": ece(y, p),
        "calibration": calibration(y, p),
        "log_loss": log_loss(y, p),
        "n": int(len(y)),
        "click_rate": float(np.mean(y)),
    }
