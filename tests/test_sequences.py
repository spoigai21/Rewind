"""History building and the day split, on a tiny log where every right answer is known."""

import numpy as np
import polars as pl
import pytest

from rewind.data.sequences import (
    MISSING, SPLITS, UNSEEN, build, check_invariants, history_window,
)

BASE = 1494000000  # 2017-05-06 00:00:00 Asia/Shanghai
DAY = 86400

# User 1: a page load of two ads, one more ad, then a val-day and a test-day impression.
# The test-day one is at 00:30 Shanghai on May 13, which is still May 12 in UTC: a UTC day cut
# would wrongly put it in validation.
# User 2: a page load of two ads, then one more five seconds later. No profile row.
# User 3: a single test-day impression.
RAW = pl.DataFrame(
    {
        "user":       [1,         1,         1,          1,                1,                  2,          2,          2,              3],
        "time_stamp": [BASE+3600, BASE+3600, BASE+7200,  BASE+6*DAY+100,   BASE+7*DAY+1800,    BASE+DAY,   BASE+DAY,   BASE+DAY+5,     BASE+7*DAY+60],
        "adgroup_id": [11,        10,        12,         10,               99,                 12,         11,         13,             13],
        "pid":        ["a",       "a",       "b",        "a",              "a",                "b",        "b",        "a",            "a"],
        "nonclk":     [0,         1,         1,          0,                1,                  1,          1,          0,              1],
        "clk":        [1,         0,         0,          1,                0,                  0,          0,          1,              0],
    }
).lazy()
ADS = pl.DataFrame(
    {
        "adgroup_id": [10, 11, 12, 13, 99],
        "cate_id": [1, 1, 2, 3, 4],
        "campaign_id": [7, 7, 8, 9, 9],
        "customer": [70, 70, 80, 90, 90],
        "brand": ["100", "101", "100", None, "102"],  # Parquet stores brand as text
        "price": [1.0, 2.0, 3.0, 4.0, 5.0],
    }
).lazy()
PROFILES = pl.DataFrame(
    {
        "userid": [1, 3],
        "cms_segid": [0, 5], "cms_group_id": [1, 2], "final_gender_code": [1, 2], "age_level": [3, 4],
        "pvalue_level": [None, 2.0], "shopping_level": [3, 1], "occupation": [0, 1],
        "new_user_class_level": [2.0, None],
    }
).lazy()


@pytest.fixture
def cols():
    cols, _ = build(RAW, ADS, PROFILES)
    return cols


def test_every_impression_appears_exactly_once(cols):
    got = sorted(zip(cols["user"], cols["time_stamp"], cols["clk"]))
    want = sorted(zip(*RAW.select("user", "time_stamp", "clk").collect().to_numpy().T))
    assert got == want


def test_history_is_strictly_earlier_seconds(cols):
    prior = (cols["hist_end"] - cols["user_start"]).tolist()
    # Sorted rows: user 1 -> (t0, ad 10), (t0, ad 11), t1, val, test; user 2 -> two at t, one at t+5; user 3.
    assert prior == [0, 0, 2, 3, 4, 0, 0, 2, 0]


def test_page_load_neighbours_are_not_history(cols):
    # Rows 0 and 1 are user 1's two ads from the same second: neither sees the other, so the
    # model cannot learn from the neighbour's click (row for ad 11 was clicked).
    pos, real = history_window(cols, np.array([0, 1]), length=4)
    assert not real.any()


def test_days_are_cut_in_shanghai_time(cols):
    assert cols["split"][:5].tolist() == [SPLITS["train"]] * 3 + [SPLITS["val"], SPLITS["test"]]
    assert cols["split"][8] == SPLITS["test"]


def test_vocabularies_come_from_training_days_only(cols):
    ads = cols["adgroup_id"].tolist()
    assert ads[4] == UNSEEN  # ad 99 appears only on the test day
    assert ads[3] not in (UNSEEN, MISSING)  # ad 10 on the val day was seen in training
    assert ads[8] == ads[7]  # ad 13 seen in training (user 2), same code on the test day
    assert cols["brand"][7] == MISSING  # brand "NULL"
    assert cols["age_level"][5] == MISSING  # user 2 has no profile row
    assert cols["pvalue_level"][0] == MISSING  # null field inside an existing profile


def test_window_is_left_padded_oldest_to_newest(cols):
    pos, real = history_window(cols, np.array([4]), length=6)  # user 1's test-day row: 4 prior
    assert real.tolist() == [[False, False, True, True, True, True]]
    assert pos[0, 2:].tolist() == [0, 1, 2, 3]
    pos, real = history_window(cols, np.array([4]), length=2)  # keeps only the newest two
    assert pos[0].tolist() == [2, 3] and real.all()


def test_window_never_reaches_into_another_user(cols):
    # User 2's first rows have no history; a long window must be all padding, not user 1's rows.
    pos, real = history_window(cols, np.array([5, 6, 7]), length=8)
    users = cols["user"][pos]
    assert (users[real] == 2).all()
    assert real.sum() == 2  # only row 7 has history: the two ads at t


def test_invariant_check_passes_on_correct_data(cols):
    check_invariants(cols)


def test_invariant_check_catches_a_same_second_leak(cols):
    # Simulate the classic bug: history ends at the row itself instead of at its page load.
    broken = dict(cols, hist_end=np.arange(len(cols["user"])))
    with pytest.raises(AssertionError, match="same second"):
        check_invariants(broken)
