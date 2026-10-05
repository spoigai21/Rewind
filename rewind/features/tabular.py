"""Phase 2, step 1: one row of features per impression, for the tabular (feature-crossing) models.

Two groups, kept separate on purpose:

    context  - the candidate ad's price, and how often it (and its category and advertiser) was
               shown and clicked across ALL users so far. Shared by every model family, including
               the Phase 3 sequence model.
    history  - the user's own past compressed into counts, click rates and recency: how a tabular
               model sees history. The Phase 3 sequence model gets the ordered timeline instead.

Categorical inputs (ad, category, brand, campaign, advertiser, placement, profile fields) are the
Phase 1 codes, used as-is.

Every count uses only impressions from STRICTLY EARLIER SECONDS than the impression it describes,
the same rule as the Phase 1 histories. For user-level counts that includes other ads in the same
page load (same second: excluded). For global counts it means other users' impressions in the same
second are excluded too.

    uv run python -m rewind.features.tabular
"""

import argparse
import json
from pathlib import Path

import numpy as np
import polars as pl

from rewind.data.sequences import AD_ID_COLS, PROFILE_COLS, SPLITS, load

CAT_FEATURES = AD_ID_COLS + PROFILE_COLS
WINDOWS = {"1h": 3600, "24h": 86400}
SMOOTHING = 10  # pseudo-impressions at the training click rate, so 1 click in 2 views is not "50%"


def raw_features(cols: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Untransformed counts, click totals and recency for every row. All leak-free by construction."""
    t = np.asarray(cols["time_stamp"], dtype=np.int64)
    clk = np.asarray(cols["clk"], dtype=np.int64)
    user = np.asarray(cols["user"], dtype=np.int64)
    us = np.asarray(cols["user_start"])
    he = np.asarray(cols["hist_end"])
    # csum[k] = clicks in rows 0..k-1, so clicks in rows a..b-1 = csum[b] - csum[a].
    csum = np.concatenate([[0], np.cumsum(clk)])
    f: dict[str, np.ndarray] = {}

    # --- user's own history, from the Phase 1 bookmarks: rows us..he-1 are strictly earlier ---
    f["hist_user_n"] = he - us
    f["hist_user_clk"] = csum[he] - csum[us]

    # Recent windows: rows sorted by (user, time), so one sorted key finds each window's start.
    t0 = t.min()
    key = (user << 31) | (t - t0)
    for name, w in WINDOWS.items():
        lo = np.maximum(np.searchsorted(key, key - w, side="left"), us)
        f[f"hist_user_n_{name}"] = he - lo
        f[f"hist_user_clk_{name}"] = csum[he] - csum[lo]

    has = he > us
    prev = np.maximum(he - 1, 0)
    f["hist_secs_since_last"] = np.where(has, t - t[prev], -1)

    # Time of the user's latest click before this second. Encode a click as (t - t0 + 1), no click
    # as 0, offset by user so a running maximum never carries over from the previous user.
    enc = np.where(clk == 1, t - t0 + 1, 0) + (user << 31)
    last_click = np.maximum.accumulate(enc)[prev] - (user << 31)
    last_click = np.where(has, last_click, 0)
    f["hist_secs_since_last_click"] = np.where(last_click > 0, t - (last_click - 1 + t0), -1)

    # --- user x (category / brand / ad / advertiser): same user, same key, strictly earlier ---
    for key_col, short in (("raw_cate_id", "cate"), ("raw_brand", "brand"),
                           ("raw_adgroup_id", "ad"), ("raw_customer", "advertiser")):
        n, c = prior_counts([user, np.asarray(cols[key_col])], t, clk)
        f[f"hist_user_{short}_n"], f[f"hist_user_{short}_clk"] = n, c

    # --- context: the candidate across all users, strictly earlier seconds ---
    for key_col, short in (("raw_adgroup_id", "ad"), ("raw_cate_id", "cate"), ("raw_customer", "advertiser")):
        n, c = prior_counts([np.asarray(cols[key_col])], t, clk)
        f[f"ctx_{short}_n"], f[f"ctx_{short}_clk"] = n, c
    f["ctx_price"] = np.asarray(cols["price"], dtype=np.float64)
    # 4-byte storage: ~30 columns x 26.5M rows would otherwise hold ~6 GB at once.
    return {k: v.astype(np.float32 if k == "ctx_price" else np.int32) for k, v in f.items()}


def prior_counts(keys: list[np.ndarray], t: np.ndarray, clk: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """For each row: how many rows with the same key values came in strictly earlier seconds, and
    how many of those were clicked. Returned in the input row order."""
    names = [f"k{i}" for i in range(len(keys))]
    df = pl.DataFrame({**dict(zip(names, keys)), "t": t, "clk": clk}).with_row_index("row")
    df = df.sort([*names, "t"]).with_row_index("pos")
    # Two separate steps: a per-group expression nested inside another per-group expression in
    # one call silently returned zeros (BUGLOG #10).
    df = df.with_columns(
        # Rows before this one in the same key group, counted at the first row of this second.
        (pl.col("pos").min().over([*names, "t"]) - pl.col("pos").min().over(names)).alias("n"),
        # Clicks in the key group before this row.
        (pl.col("clk").cum_sum().over(names) - pl.col("clk")).alias("c_row"),
    ).with_columns(
        # The minimum within a second is the value at the second's first row: clicks strictly
        # before the second.
        pl.col("c_row").min().over([*names, "t"]).alias("c"),
    ).sort("row")
    return df["n"].to_numpy().astype(np.int64), df["c"].to_numpy().astype(np.int64)


def transform(raw: dict[str, np.ndarray], base_rate: float) -> tuple[np.ndarray, list[str]]:
    """Turn raw counts into model-friendly numbers: log counts, smoothed click rates, log recency
    with a separate "never" flag. Returns an (n, k) float32 matrix and its column names."""
    rules = []  # (output name, function producing that column)
    for name in raw:
        if "_clk" in name and not name.startswith("hist_secs"):
            continue  # click totals are folded into the matching click rate below
        if name.endswith("_n") or "_n_" in name:
            clk_name = name.replace("_n", "_clk", 1)
            rules.append((f"log_{name}", lambda n=name: np.log1p(raw[n].astype(np.float32))))
            rules.append((name.replace("_n", "_ctr", 1), lambda n=name, c=clk_name:
                          (raw[c] + np.float32(SMOOTHING * base_rate)) / (raw[n] + np.float32(SMOOTHING))))
        elif name.startswith("hist_secs"):
            rules.append((f"log_{name}", lambda n=name: np.log1p(np.maximum(raw[n], 0).astype(np.float32))))
            rules.append((f"{name}_never", lambda n=name: raw[n] < 0))
        elif name == "ctx_price":
            rules.append(("log_ctx_price", lambda: np.log1p(raw["ctx_price"])))
        else:
            raise ValueError(f"no transform rule for {name}")
    # Fill one preallocated float32 matrix column by column instead of stacking copies.
    num = np.empty((len(next(iter(raw.values()))), len(rules)), dtype=np.float32)
    for j, (_, fn) in enumerate(rules):
        num[:, j] = fn()
    return num, [name for name, _ in rules]


def standardize(num: np.ndarray, train: np.ndarray) -> tuple[np.ndarray, dict]:
    """Scale each column to mean 0, std 1 using TRAINING rows only."""
    mean = num[train].mean(axis=0, dtype=np.float64)
    std = num[train].std(axis=0, dtype=np.float64)
    std[std == 0] = 1.0
    num -= mean.astype(np.float32)  # in place: no second copy of the matrix
    num /= std.astype(np.float32)
    return num, {"mean": mean.tolist(), "std": std.tolist()}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seq-dir", type=Path, default=Path("data/sequences"))
    ap.add_argument("--out-dir", type=Path, default=Path("data/features"))
    args = ap.parse_args()
    num_names = build_features(args.seq_dir, args.out_dir)
    print(f"{len(num_names)} numeric + {len(CAT_FEATURES)} categorical features")
    for n in num_names:
        print(" ", n)
    print(f"wrote {args.out_dir}/")


def build_features(seq_dir: Path, out_dir: Path) -> list[str]:
    """Build and save num.npy, cat.npy and meta.json from a Phase 1 sequence directory."""
    cols = load(seq_dir)
    train = np.asarray(cols["split"]) == SPLITS["train"]
    base_rate = float(np.asarray(cols["clk"])[train].mean())

    raw = raw_features(cols)
    num, num_names = transform(raw, base_rate)
    del raw
    num, norm = standardize(num, train)
    cat = np.stack([np.asarray(cols[c]) for c in CAT_FEATURES], axis=1).astype(np.int32)

    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "num.npy", num)
    np.save(out_dir / "cat.npy", cat)
    meta = {
        "rows": int(len(num)),
        "num_features": num_names,
        "num_groups": {"context": [n for n in num_names if "ctx_" in n],
                       "history": [n for n in num_names if "hist_" in n]},
        "cat_features": CAT_FEATURES,
        "cat_cardinalities": [int(np.asarray(cols[c]).max()) + 1 for c in CAT_FEATURES],
        "train_base_rate": base_rate,
        "smoothing": SMOOTHING,
        "normalization_from": "training rows only",
        "normalization": norm,
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    return num_names


if __name__ == "__main__":
    main()
