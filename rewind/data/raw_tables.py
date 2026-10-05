"""The four raw tables of the Taobao display-ad log, and conversion from CSV to Parquet.

The column lists below are what the dataset's documentation says. They are checked against each
file's real header before anything is read, and a mismatch stops the run with both lists printed:
the documentation is a claim to verify, not a fact.

    raw_sample    one row per ad impression: who saw which ad, when, and whether they clicked
    ad_feature    one row per ad: its category, brand, campaign, advertiser and price
    user_profile  one row per user: coarse demographic and shopping-level buckets
    behavior_log  one row per user action: browse / cart / favourite / buy, on a category+brand

Timestamps are Unix seconds. The site operates in China, so calendar days are cut at midnight
Asia/Shanghai (UTC+8). Cutting at UTC midnight would put 8 hours of each day on the wrong side
of the train/validation/test boundary.
"""

from pathlib import Path

import polars as pl

TZ = "Asia/Shanghai"

# file stem -> {column: dtype}
SCHEMAS: dict[str, dict[str, pl.DataType]] = {
    "raw_sample": {
        "user": pl.Int64,
        "time_stamp": pl.Int64,
        "adgroup_id": pl.Int64,
        "pid": pl.Utf8,
        "nonclk": pl.Int8,
        "clk": pl.Int8,
    },
    "ad_feature": {
        "adgroup_id": pl.Int64,
        "cate_id": pl.Int64,
        "campaign_id": pl.Int64,
        "customer": pl.Int64,
        "brand": pl.Utf8,  # documented as an ID, but kept as text until counted: may contain "NULL"
        "price": pl.Float64,
    },
    "user_profile": {
        "userid": pl.Int64,
        "cms_segid": pl.Int64,
        "cms_group_id": pl.Int64,
        "final_gender_code": pl.Int64,
        "age_level": pl.Int64,
        "pvalue_level": pl.Float64,  # documented as having missing values
        "shopping_level": pl.Int64,
        "occupation": pl.Int64,
        "new_user_class_level ": pl.Float64,  # trailing space is in the documented header
    },
    "behavior_log": {
        "user": pl.Int64,
        "time_stamp": pl.Int64,
        "btag": pl.Utf8,
        "cate": pl.Int64,
        "brand": pl.Int64,
    },
}


def find_csv(raw_dir: Path, stem: str) -> Path:
    """The CSV for a table. Tolerates names like raw_behavior_log.csv; refuses archives."""
    matches = sorted(p for p in raw_dir.iterdir() if stem in p.name)
    csvs = [p for p in matches if p.suffix == ".csv"]
    if len(csvs) == 1:
        return csvs[0]
    if not csvs and matches:
        names = ", ".join(p.name for p in matches)
        raise SystemExit(f"{stem}: found {names} but no .csv; extract the archive into {raw_dir}")
    if not csvs:
        raise SystemExit(f"{stem}: no file containing '{stem}' in {raw_dir}")
    raise SystemExit(f"{stem}: several candidates {[p.name for p in csvs]}; keep exactly one")


def check_header(path: Path, stem: str) -> list[str]:
    """Compare the file's real header to the documented one. Returns the real header."""
    with path.open() as f:
        header = f.readline().rstrip("\r\n").split(",")
    expected = list(SCHEMAS[stem])
    if [h.strip() for h in header] != [e.strip() for e in expected]:
        raise SystemExit(f"{path.name}: header differs from documentation\n"
                         f"  documented: {expected}\n  actual:     {header}")
    return header


def to_parquet(raw_dir: Path, out_dir: Path, stem: str) -> Path:
    """Stream one CSV into Parquet without loading it all into memory.

    Values that fail to parse as their documented type become nulls rather than crashing, and the
    counting step reports how many there are; silently dropping rows here would hide the problem.
    Column names are stripped of stray whitespace.
    """
    path = find_csv(raw_dir, stem)
    header = check_header(path, stem)
    dtypes = dict(zip(header, SCHEMAS[stem].values()))
    out = out_dir / f"{stem}.parquet"
    out_dir.mkdir(parents=True, exist_ok=True)
    (
        pl.scan_csv(path, schema=dtypes, null_values=["NULL", ""], ignore_errors=True)
        .rename({h: h.strip() for h in header})
        .sink_parquet(out)
    )
    return out
