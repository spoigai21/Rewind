"""Phase 1, step 1: convert the raw log to Parquet and count what is actually in it.

Nothing here trusts the dataset's documentation. Row counts, date ranges, click rates, ID
overlaps between tables and history lengths are all measured, and the output becomes the dataset
card (DATASET.md).

    uv run python -m rewind.data.count                    # expects CSVs in data/raw/
    uv run python -m rewind.data.count --raw-dir PATH
"""

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import polars as pl

from rewind.data.raw_tables import SCHEMAS, TZ, to_parquet

QUANTILES = [0.1, 0.25, 0.5, 0.75, 0.9, 0.99]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    ap.add_argument("--parquet-dir", type=Path, default=Path("data/parquet"))
    ap.add_argument("--out", type=Path, default=Path("results/phase1/counts.json"))
    args = ap.parse_args()

    tables = {}
    for stem in SCHEMAS:
        out = args.parquet_dir / f"{stem}.parquet"
        if out.exists():
            print(f"{stem}: using existing {out}")
        else:
            t0 = time.perf_counter()
            to_parquet(args.raw_dir, args.parquet_dir, stem)
            print(f"{stem}: converted to Parquet in {time.perf_counter() - t0:.0f}s")
        tables[stem] = pl.scan_parquet(out)

    counts = count_all(tables)
    counts["created_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(counts, indent=2, default=str) + "\n")
    print(json.dumps(counts, indent=2, default=str))
    print(f"\nwrote {args.out}")


def count_all(t: dict[str, pl.LazyFrame]) -> dict:
    return {
        "timezone_for_days": TZ,
        "raw_sample": count_impressions(t["raw_sample"]),
        "ad_feature": count_table(t["ad_feature"], key="adgroup_id"),
        "user_profile": count_table(t["user_profile"], key="userid"),
        "behavior_log": count_behaviour(t["behavior_log"]),
        "overlap": count_overlap(t),
    }


def collect(lf: pl.LazyFrame) -> pl.DataFrame:
    # The streaming engine processes the data in chunks, so 700M rows fit in laptop memory.
    return lf.collect(engine="streaming")


def count_table(lf: pl.LazyFrame, key: str | None = None) -> dict:
    """Rows, nulls per column, distinct values per column, and whether `key` is really unique."""
    cols = lf.collect_schema().names()
    stats = collect(lf.select(
        pl.len().alias("rows"),
        *[pl.col(c).null_count().alias(f"null:{c}") for c in cols],
        *[pl.col(c).n_unique().alias(f"distinct:{c}") for c in cols],
    )).row(0, named=True)
    out = {
        "rows": stats["rows"],
        "nulls": {c: stats[f"null:{c}"] for c in cols},
        "distinct": {c: stats[f"distinct:{c}"] for c in cols},
    }
    if key:
        out["key"] = key
        out["key_is_unique"] = stats[f"distinct:{key}"] == stats["rows"]
    return out


def with_day(lf: pl.LazyFrame) -> pl.LazyFrame:
    return lf.with_columns(
        pl.from_epoch("time_stamp", time_unit="s").dt.replace_time_zone("UTC")
        .dt.convert_time_zone(TZ).dt.date().alias("day")
    )


def time_range(lf: pl.LazyFrame) -> dict:
    r = collect(lf.select(pl.col("time_stamp").min().alias("lo"), pl.col("time_stamp").max().alias("hi"))).row(0)
    fmt = lambda s: datetime.fromtimestamp(s, ZoneInfo(TZ)).isoformat() if s is not None else None
    return {"min_unix": r[0], "max_unix": r[1], "min_shanghai": fmt(r[0]), "max_shanghai": fmt(r[1])}


def count_impressions(lf: pl.LazyFrame) -> dict:
    out = count_table(lf)
    out["time_range"] = time_range(lf)
    out["exact_duplicate_rows"] = out["rows"] - collect(lf.unique().select(pl.len())).item()
    out["clk_plus_nonclk_not_1"] = collect(lf.filter(pl.col("clk") + pl.col("nonclk") != 1).select(pl.len())).item()
    out["click_rate"] = collect(lf.select(pl.col("clk").mean())).item()
    per_day = collect(
        with_day(lf).group_by("day").agg(pl.len().alias("impressions"), pl.col("clk").mean().alias("click_rate"))
        .sort("day")
    )
    out["per_day"] = per_day.to_dicts()
    out["pid_counts"] = collect(lf.group_by("pid").len().sort("pid")).to_dicts()
    return out


def count_behaviour(lf: pl.LazyFrame) -> dict:
    out = count_table(lf)
    out["time_range"] = time_range(lf)
    out["btag_counts"] = collect(lf.group_by("btag").len().sort("len", descending=True)).to_dicts()
    out["per_day"] = collect(with_day(lf).group_by("day").len().sort("day")).to_dicts()
    per_user = collect(lf.group_by("user").len())["len"]
    out["actions_per_user"] = {
        "users": per_user.len(),
        "mean": per_user.mean(),
        **{f"p{int(q * 100)}": per_user.quantile(q, "nearest") for q in QUANTILES},
        "max": per_user.max(),
    }
    return out


def count_overlap(t: dict[str, pl.LazyFrame]) -> dict:
    """How many impressions can be joined to each other table. An impression whose user has no
    behaviour history cannot get a sequence at all; one whose ad has no features loses its
    category and brand."""
    imps = t["raw_sample"].select("user", "adgroup_id")
    n = collect(imps.select(pl.len())).item()

    def share(other: pl.LazyFrame, left: str, right: str) -> float:
        keys = other.select(pl.col(right).alias(left)).unique()
        return collect(imps.join(keys, on=left, how="semi").select(pl.len())).item() / n

    return {
        "impressions": n,
        "share_with_ad_features": share(t["ad_feature"], "adgroup_id", "adgroup_id"),
        "share_with_user_profile": share(t["user_profile"], "user", "userid"),
        "share_with_any_behaviour": share(t["behavior_log"], "user", "user"),
    }


if __name__ == "__main__":
    main()
