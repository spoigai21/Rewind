"""The counting step on a tiny hand-made log where every right answer is known in advance,
including planted problems the real data might contain."""

from pathlib import Path

import polars as pl
import pytest

from rewind.data.count import count_all
from rewind.data.raw_tables import to_parquet

# 2017-05-06 00:30 and 23:30 in Shanghai. In UTC both fall on 2017-05-05 / 2017-05-06, so a
# UTC day cut would split them; the Shanghai cut must put both on 2017-05-06.
EARLY = 1494001800  # 2017-05-06 00:30 +08:00
LATE = 1494084600  # 2017-05-06 23:30 +08:00
NEXT = 1494088200  # 2017-05-07 00:30 +08:00

FILES = {
    "raw_sample.csv": [
        "user,time_stamp,adgroup_id,pid,nonclk,clk",
        f"1,{EARLY},10,430548_1007,1,0",
        f"1,{EARLY},10,430548_1007,1,0",  # exact duplicate
        f"2,{LATE},11,430539_1007,0,1",
        f"3,{NEXT},99,430548_1007,1,1",  # clk + nonclk == 2: inconsistent label; ad 99 has no features
    ],
    "ad_feature.csv": [
        "adgroup_id,cate_id,campaign_id,customer,brand,price",
        "10,5,100,1000,NULL,9.9",
        "11,6,101,1001,7,19.0",
    ],
    "user_profile.csv": [
        "userid,cms_segid,cms_group_id,final_gender_code,age_level,pvalue_level,shopping_level,occupation,new_user_class_level ",
        "1,0,5,2,5,,3,0,3",
        "2,0,5,2,5,2,3,0,",
    ],
    "behavior_log.csv": [
        "user,time_stamp,btag,cate,brand",
        f"1,{EARLY - 60},pv,5,7",
        f"1,{EARLY - 30},cart,5,7",
        f"1,{LATE},pv,6,8",
        f"2,{EARLY},fav,6,8",
    ],
}


@pytest.fixture
def counts(tmp_path: Path) -> dict:
    raw = tmp_path / "raw"
    raw.mkdir()
    for name, lines in FILES.items():
        (raw / name).write_text("\n".join(lines) + "\n")
    tables = {stem: pl.scan_parquet(to_parquet(raw, tmp_path / "pq", stem))
              for stem in ("raw_sample", "ad_feature", "user_profile", "behavior_log")}
    return count_all(tables)


def test_rows_and_planted_problems(counts):
    imp = counts["raw_sample"]
    assert imp["rows"] == 4
    assert imp["exact_duplicate_rows"] == 1
    assert imp["clk_plus_nonclk_not_1"] == 1


def test_days_are_cut_in_shanghai_time(counts):
    days = {str(d["day"]): d["impressions"] for d in counts["raw_sample"]["per_day"]}
    assert days == {"2017-05-06": 3, "2017-05-07": 1}


def test_null_markers_become_nulls_not_strings(counts):
    assert counts["ad_feature"]["nulls"]["brand"] == 1
    assert counts["user_profile"]["nulls"]["pvalue_level"] == 1
    assert counts["user_profile"]["nulls"]["new_user_class_level"] == 1  # header had a trailing space


def test_overlap_between_tables(counts):
    ov = counts["overlap"]
    assert ov["share_with_ad_features"] == pytest.approx(3 / 4)  # ad 99 missing
    assert ov["share_with_user_profile"] == pytest.approx(3 / 4)  # user 3 missing
    assert ov["share_with_any_behaviour"] == pytest.approx(3 / 4)  # user 3 has no history


def test_actions_per_user(counts):
    apu = counts["behavior_log"]["actions_per_user"]
    assert apu["n"] == 2
    assert apu["max"] == 3


def test_header_mismatch_stops_the_run(tmp_path: Path):
    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "ad_feature.csv").write_text("adgroup_id,cate_id,brand\n1,2,3\n")
    with pytest.raises(SystemExit, match="header differs"):
        to_parquet(raw, tmp_path / "pq", "ad_feature")


def test_prior_impressions_count_only_strictly_earlier(counts):
    # User 1: two impressions in the same second -> neither is "before" the other -> 0 and 0.
    # User 2: one impression -> 0. User 3: one impression -> 0.
    prior = counts["raw_sample"]["prior_impressions"]
    assert prior["max"] == 0
    assert prior["share_with_0"] == 1.0


def test_runs_without_behaviour_log(tmp_path: Path):
    raw = tmp_path / "raw"
    raw.mkdir()
    for name, lines in FILES.items():
        if name != "behavior_log.csv":
            (raw / name).write_text("\n".join(lines) + "\n")
    tables = {stem: pl.scan_parquet(to_parquet(raw, tmp_path / "pq", stem))
              for stem in ("raw_sample", "ad_feature", "user_profile")}
    c = count_all(tables)
    assert c["behavior_log"] is None
    assert c["overlap"]["share_with_any_behaviour"] is None
    assert c["raw_sample"]["rows"] == 4


def test_prior_impressions_with_real_history():
    # User 1 sees ads at t=10 (two at once), 20, 30; user 2 at t=5 and 15.
    # Strictly-earlier counts: user 1 -> 0, 0, 2, 3; user 2 -> 0, 1.
    from rewind.data.count import count_prior_impressions

    lf = pl.LazyFrame({"user": [1, 1, 1, 1, 2, 2], "time_stamp": [10, 10, 20, 30, 15, 5]})
    prior = count_prior_impressions(lf)
    assert prior["max"] == 3
    assert prior["share_with_0"] == pytest.approx(3 / 6)
    assert prior["mean"] == pytest.approx((0 + 0 + 2 + 3 + 0 + 1) / 6)
