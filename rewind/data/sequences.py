"""Phase 1, step 3: give every impression its history, and split by day.

Layout. All impressions are sorted by (user, time, ad) and stored once, one array per column, in
data/sequences/. Each row is both a question (predict its `clk`) and, for that user's later rows,
a history item. Instead of copying a history into every question, each row stores two bookmarks:

    user_start[i]  index of this user's first impression
    hist_end[i]    index of the first impression in row i's own second (its page load)

Row i's full history is rows user_start[i] .. hist_end[i]-1: the same user, strictly earlier
seconds. Ads shown in the same second are one page load and are never each other's history.
history_window() cuts the most recent L of those for any length L, so one build serves the whole
length sweep.

ID columns are re-coded to small integers with vocabularies built from TRAINING days only, so an
ad first seen on the validation or test day looks new to the model, as it would in production:

    0 = padding    1 = missing (e.g. brand "NULL", no profile row)    2 = not seen in training
    3.. = values seen in training

    uv run python -m rewind.data.sequences
"""

import argparse
import json
import subprocess
from datetime import date, datetime, timezone
from pathlib import Path

import numpy as np
import polars as pl

from rewind.data.raw_tables import TZ

PAD, MISSING, UNSEEN, FIRST_ID = 0, 1, 2, 3

TRAIN_DAYS = (date(2017, 5, 6), date(2017, 5, 11))  # inclusive
VAL_DAY = date(2017, 5, 12)
TEST_DAY = date(2017, 5, 13)
SPLITS = {"train": 0, "val": 1, "test": 2}

AD_ID_COLS = ["adgroup_id", "cate_id", "brand", "campaign_id", "customer", "pid"]
PROFILE_COLS = ["cms_segid", "cms_group_id", "final_gender_code", "age_level", "pvalue_level",
                "shopping_level", "occupation", "new_user_class_level"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--parquet-dir", type=Path, default=Path("data/parquet"))
    ap.add_argument("--out-dir", type=Path, default=Path("data/sequences"))
    ap.add_argument("--summary", type=Path, default=Path("results/phase1/sequences.json"))
    args = ap.parse_args()

    git = git_info()
    tables = {s: pl.scan_parquet(args.parquet_dir / f"{s}.parquet")
              for s in ("raw_sample", "ad_feature", "user_profile")}
    cols, vocab_sizes = build(tables["raw_sample"], tables["ad_feature"], tables["user_profile"])

    check_invariants(cols)
    save(cols, args.out_dir)
    summary = summarize(cols, vocab_sizes)
    summary["git"] = git
    summary["created_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    print(f"\nwrote {args.out_dir}/ and {args.summary}")


def build(raw: pl.LazyFrame, ads: pl.LazyFrame, profiles: pl.LazyFrame) -> tuple[dict[str, np.ndarray], dict]:
    """Returns one numpy array per column, rows sorted by (user, time, ad), plus vocabulary sizes."""
    day = (pl.from_epoch("time_stamp", time_unit="s").dt.replace_time_zone("UTC")
           .dt.convert_time_zone(TZ).dt.date())
    split = (pl.when(day.is_between(*TRAIN_DAYS)).then(SPLITS["train"])
             .when(day == VAL_DAY).then(SPLITS["val"])
             .when(day == TEST_DAY).then(SPLITS["test"])
             .otherwise(None))

    ev = (
        raw.join(ads.with_columns(pl.col("brand").cast(pl.Int64)), on="adgroup_id", how="left")
        .join(profiles.rename({"userid": "user"}), on="user", how="left")
        .with_columns(split.alias("split"))
        .sort(["user", "time_stamp", "adgroup_id"])
        .collect()
    )
    if ev["split"].null_count():
        raise SystemExit(f"{ev['split'].null_count()} impressions fall outside the split days")

    is_train = ev["split"] == SPLITS["train"]
    vocab_sizes = {}
    for col in AD_ID_COLS:
        ev = ev.with_columns(encode(ev[col], ev[col].filter(is_train)).alias(col))
        vocab_sizes[col] = int(ev[col].max()) + 1
    for col in PROFILE_COLS:
        # Profiles are a static per-user table, not events, so every value is "seen"; only
        # missing (no profile row, or a null field) gets its own code.
        ev = ev.with_columns(encode(ev[col], ev[col]).alias(col))
        vocab_sizes[col] = int(ev[col].max()) + 1

    # Number the rows once, globally, THEN take each group's smallest number. (Generating the row
    # numbers inside .over() would restart them at 0 in every group.)
    ev = ev.with_row_index("row").with_columns(
        pl.col("row").min().over("user").alias("user_start"),
        pl.col("row").min().over(["user", "time_stamp"]).alias("hist_end"),
    )

    cols = {
        "user": ev["user"].to_numpy().astype(np.int64),
        "time_stamp": ev["time_stamp"].to_numpy().astype(np.int64),
        "split": ev["split"].to_numpy().astype(np.int8),
        "clk": ev["clk"].to_numpy().astype(np.int8),
        "user_start": ev["user_start"].to_numpy().astype(np.int64),
        "hist_end": ev["hist_end"].to_numpy().astype(np.int64),
        **{c: ev[c].to_numpy().astype(np.int32) for c in AD_ID_COLS + PROFILE_COLS},
    }
    return cols, vocab_sizes


def encode(values: pl.Series, known: pl.Series) -> pl.Series:
    """Map raw values to codes: null -> MISSING, values in `known` -> FIRST_ID.., others -> UNSEEN."""
    vocab = known.drop_nulls().unique().sort()
    mapping = pl.DataFrame({"v": vocab, "code": pl.int_range(FIRST_ID, FIRST_ID + len(vocab), eager=True)})
    return (
        pl.DataFrame({"v": values})
        .join(mapping, on="v", how="left", maintain_order="left")
        .select(
            pl.when(pl.col("v").is_null()).then(MISSING)
            .otherwise(pl.col("code").fill_null(UNSEEN))
            .cast(pl.Int32)
        )
        .to_series()
    )


def history_window(cols: dict[str, np.ndarray], rows: np.ndarray, length: int) -> tuple[np.ndarray, np.ndarray]:
    """For each question row, the row indices of its most recent `length` history items.

    Returns (positions, real): both (len(rows), length), oldest -> newest, LEFT-padded.
    real[k, j] is False for padding; positions there are meaningless and must be masked.
    """
    end = cols["hist_end"][rows][:, None]
    positions = end - length + np.arange(length)[None, :]
    real = positions >= cols["user_start"][rows][:, None]
    return np.where(real, positions, 0), real


def check_invariants(cols: dict[str, np.ndarray]) -> None:
    """Prove on the full data that every history is the same user's strictly-earlier impressions.

    Because rows are sorted and each history is the contiguous block user_start..hist_end-1, it
    is enough to check the sort order and the two block edges; every row inside a block then
    belongs to the same user and is strictly earlier.
    """
    user, t = cols["user"], cols["time_stamp"]
    us, he = cols["user_start"], cols["hist_end"]
    n = len(user)
    i = np.arange(n)

    same_user_next = user[1:] == user[:-1]
    assert (user[1:] >= user[:-1]).all(), "rows not sorted by user"
    assert (t[1:][same_user_next] >= t[:-1][same_user_next]).all(), "rows not sorted by time within user"

    assert (us <= he).all() and (he <= i).all(), "bookmarks out of order"
    assert (user[us] == user).all(), "user_start points at another user"
    assert ((us == 0) | (user[np.maximum(us - 1, 0)] != user)).all(), "user_start is not the user's first row"
    assert (t[he] == t).all(), "hist_end not in the question's own second"
    has_hist = he > us
    last = he[has_hist] - 1
    assert (user[last] == user[has_hist]).all(), "history crosses into another user"
    assert (t[last] < t[has_hist]).all(), "history contains an item from the same second or later"
    assert len(np.unique(i)) == n


def summarize(cols: dict[str, np.ndarray], vocab_sizes: dict) -> dict:
    prior = cols["hist_end"] - cols["user_start"]
    out = {"rows": int(len(prior)), "vocab_sizes": vocab_sizes, "splits": {}}
    for name, code in SPLITS.items():
        m = cols["split"] == code
        p = prior[m]
        out["splits"][name] = {
            "impressions": int(m.sum()),
            "users": int(len(np.unique(cols["user"][m]))),
            "click_rate": float(cols["clk"][m].mean()),
            "prior_impressions": {
                "median": float(np.median(p)),
                "p90": float(np.quantile(p, 0.9, method="nearest")),
                "share_with_0": float((p == 0).mean()),
                **{f"share_with_at_least_{k}": float((p >= k).mean()) for k in (16, 64, 256, 512)},
            },
            "share_ad_unseen_in_training": float((cols["adgroup_id"][m] == UNSEEN).mean()),
            "share_no_profile": float((cols["age_level"][m] == MISSING).mean()),
        }
    out["training_sparsity"] = sparsity(cols)
    out["prior_impressions_all"] = {
        "median": float(np.median(prior)),
        "p90": float(np.quantile(prior, 0.9, method="nearest")),
        "p99": float(np.quantile(prior, 0.99, method="nearest")),
        "max": int(prior.max()),
        "share_with_0": float((prior == 0).mean()),
    }
    return out


def sparsity(cols: dict[str, np.ndarray]) -> dict:
    """How often each ad and each user appears on training days. An ad seen a handful of times
    gives any model little to learn from; this is the long tail every model has to cope with."""
    train = cols["split"] == SPLITS["train"]
    out = {}
    for col in ("adgroup_id", "user"):
        counts = np.unique(cols[col][train], return_counts=True)[1]
        out[col] = {
            "distinct": int(len(counts)),
            "median_impressions": float(np.median(counts)),
            "share_seen_once": float((counts == 1).mean()),
            "share_seen_under_10": float((counts < 10).mean()),
            # Share of training IMPRESSIONS that belong to rarely-seen ads/users.
            "impression_share_from_seen_under_10": float(counts[counts < 10].sum() / counts.sum()),
        }
    return out


def save(cols: dict[str, np.ndarray], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, arr in cols.items():
        np.save(out_dir / f"{name}.npy", arr)


def load(out_dir: Path = Path("data/sequences")) -> dict[str, np.ndarray]:
    """Memory-mapped: arrays are read from disk on demand, so all 26.5M rows need not fit in RAM."""
    return {p.stem: np.load(p, mmap_mode="r") for p in sorted(out_dir.glob("*.npy"))}


def git_info() -> dict:
    def run(cmd):
        try:
            return subprocess.run(cmd, capture_output=True, text=True, check=True).stdout.strip()
        except (OSError, subprocess.CalledProcessError):
            return None
    status = run(["git", "status", "--porcelain", "--", ".", ":!results"])
    return {"commit": run(["git", "rev-parse", "--short", "HEAD"]), "dirty": bool(status)}


if __name__ == "__main__":
    main()
