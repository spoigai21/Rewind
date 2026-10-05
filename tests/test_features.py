"""The fast vectorized features must equal a slow, obviously-correct loop over every pair of rows."""

import numpy as np
import polars as pl
import pytest

from rewind.data.sequences import build
from rewind.features.tabular import WINDOWS, raw_features, transform

BASE = 1494000000  # 2017-05-06 00:00 Asia/Shanghai


def random_log(seed: int, n: int = 300):
    rng = np.random.default_rng(seed)
    # Few users, ads and distinct seconds, so same-second collisions and repeats are common.
    # Times span 8 days with clusters, so the 1h and 24h windows both cut through real data.
    t = BASE + rng.choice(np.concatenate([rng.integers(0, 8 * 86400, 40), rng.integers(0, 7200, 20)]), n)
    raw = pl.DataFrame({
        "user": rng.integers(1, 6, n), "time_stamp": t, "adgroup_id": rng.integers(1, 9, n),
        "pid": rng.choice(["a", "b"], n), "nonclk": 0, "clk": rng.integers(0, 2, n),
    }).with_columns((1 - pl.col("clk")).alias("nonclk"))
    ads = pl.DataFrame({
        "adgroup_id": np.arange(1, 9), "cate_id": [1, 1, 2, 2, 3, 3, 4, 4],
        "campaign_id": np.arange(1, 9), "customer": [1, 2, 1, 2, 1, 2, 1, 2],
        "brand": ["5", "5", "6", None, "7", "7", None, "8"], "price": np.linspace(1, 8, 8),
    })
    profiles = pl.DataFrame({"userid": [1, 2, 3], **{c: [1, 2, 3] for c in [
        "cms_segid", "cms_group_id", "final_gender_code", "age_level", "pvalue_level",
        "shopping_level", "occupation", "new_user_class_level"]}})
    cols, _ = build(raw.lazy(), ads.lazy(), profiles.lazy())
    return cols


def brute_force(cols) -> dict[str, np.ndarray]:
    t, clk, user = cols["time_stamp"], cols["clk"], cols["user"]
    n = len(t)
    f = {k: np.zeros(n, dtype=np.int64) for k in [
        "hist_user_n", "hist_user_clk", "hist_secs_since_last", "hist_secs_since_last_click",
        *[f"hist_user_{x}_{w}" for x in ("n", "clk") for w in WINDOWS],
        *[f"hist_user_{k}_{x}" for k in ("cate", "brand", "ad", "advertiser") for x in ("n", "clk")],
        *[f"ctx_{k}_{x}" for k in ("ad", "cate", "advertiser") for x in ("n", "clk")],
    ]}
    keys = {"cate": cols["raw_cate_id"], "brand": cols["raw_brand"], "ad": cols["raw_adgroup_id"],
            "advertiser": cols["raw_customer"]}
    for i in range(n):
        before = t < t[i]  # strictly earlier SECOND
        mine = before & (user == user[i])
        f["hist_user_n"][i] = mine.sum()
        f["hist_user_clk"][i] = clk[mine].sum()
        for w, secs in WINDOWS.items():
            m = mine & (t >= t[i] - secs)
            f[f"hist_user_n_{w}"][i] = m.sum()
            f[f"hist_user_clk_{w}"][i] = clk[m].sum()
        f["hist_secs_since_last"][i] = t[i] - t[mine].max() if mine.any() else -1
        clicked = mine & (clk == 1)
        f["hist_secs_since_last_click"][i] = t[i] - t[clicked].max() if clicked.any() else -1
        for k, v in keys.items():
            m = mine & (v == v[i])
            f[f"hist_user_{k}_n"][i], f[f"hist_user_{k}_clk"][i] = m.sum(), clk[m].sum()
        for k in ("ad", "cate", "advertiser"):
            m = before & (keys[k] == keys[k][i])
            f[f"ctx_{k}_n"][i], f[f"ctx_{k}_clk"][i] = m.sum(), clk[m].sum()
    return f


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_fast_features_equal_brute_force(seed):
    cols = random_log(seed)
    fast, slow = raw_features(cols), brute_force(cols)
    for name, expected in slow.items():
        np.testing.assert_array_equal(fast[name], expected, err_msg=name)


def test_same_second_neighbours_are_excluded():
    # The planted log above has many same-second rows; check directly that at least one row has a
    # same-second neighbour for the same user, so the brute-force test really exercised the rule.
    cols = random_log(0)
    pairs = pl.DataFrame({"u": cols["user"], "t": cols["time_stamp"]}).group_by("u", "t").len()
    assert (pairs["len"] > 1).any()


def test_transform_produces_finite_named_columns():
    cols = random_log(0)
    num, names = transform(raw_features(cols), base_rate=0.05)
    assert num.shape == (len(cols["user"]), len(names))
    assert np.isfinite(num).all()
    assert "hist_user_ctr" in names and "log_hist_secs_since_last_click" in names
    assert "hist_secs_since_last_never" in names


def test_smoothed_rate_with_no_history_equals_base_rate():
    cols = random_log(0)
    num, names = transform(raw_features(cols), base_rate=0.05)
    no_hist = cols["hist_end"] == cols["user_start"]
    assert no_hist.any()
    np.testing.assert_allclose(num[no_hist, names.index("hist_user_ctr")], 0.05, rtol=1e-6)
