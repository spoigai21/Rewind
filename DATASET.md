# Dataset card — Taobao display-ad impressions (Kaggle mirror, no behaviour log)

Every number below was **counted from the files**, not copied from documentation
(`results/phase1/counts.json`, produced by `uv run python -m rewind.data.count`). Where the
documentation makes a claim, it is listed next to the count.

---

## Source

- **Original:** "Ad Display/Click Data on Taobao.com", released by Alibaba on its Tianchi platform
  (dataset 56). 1.14M randomly sampled Taobao users, 8 days of display-ad impressions, May 2017.
- **Copy used here:** a third-party mirror on Kaggle,
  `pavansanagapati/ad-displayclick-data-on-taobaocom`, uploaded 2020-04-29. Kaggle lists its
  license as "Unknown". The official copy requires an account with payment details, so it was not
  used.
- **What the mirror is missing:** the original's 22-day **behaviour log** (browse, cart, favourite,
  purchase). Only three of the four tables are present. This is the single biggest constraint on
  the project: user history can only be built from earlier ad impressions.
- **Not this platform's internal data, and not any company's ranking system.** A public sample.

### Files and fingerprints

Place them in `data/raw/` (gitignored). A different or altered copy will not match these SHA-256
checksums.

| File | Bytes | Rows (excl. header) | SHA-256 |
|---|---:|---:|---|
| `raw_sample.csv` | 1,114,618,926 | 26,557,961 | `3c93325012be6fb83105102c6c269805ac25a5366881ea78f9a2230c3998be7b` |
| `ad_feature.csv` | 32,133,243 | 846,811 | `da813217693df73dc711d25fd936d83ff6fc914be80cb37e6eef086512bb5c6a` |
| `user_profile.csv` | 25,118,357 | 1,061,768 | `affea398d503076af5982f0d7e5c5dd6dbf405b4a23be2c978eeaff5dd1759f4` |

Row counts were confirmed two ways (Polars and `wc -l`). Column headers match the documented
headers exactly, including a trailing space in `user_profile`'s last column name.

### Documentation vs. counts

| Claim | Documented | Counted |
|---|---|---|
| Impressions | ~26 million | 26,557,961 |
| Users sampled | 1,140,000 | 1,141,729 in impressions |
| Users with a profile | 1,060,000 | 1,061,768 |
| Ads | 846,811 | 846,811 |
| Impression period | 8 days | 2017-05-06 00:00:00 to 2017-05-13 23:59:46, Asia/Shanghai |

On every point checked, this copy agrees with the documentation.

---

## Tables

**`raw_sample` — one row per ad impression.** The prediction target lives here.

| Column | Meaning | Counted notes |
|---|---|---|
| `user` | user ID | 1,141,729 distinct; no nulls |
| `time_stamp` | Unix seconds | 662,061 distinct values |
| `adgroup_id` | ad ID | 846,811 distinct; every one has a row in `ad_feature` |
| `pid` | ad placement (two values) | `430548_1007`: 16,472,898 rows, 5.01% clicked; `430539_1007`: 10,085,063 rows, 5.36% clicked |
| `clk` | 1 if clicked | **label**; 5.144% overall |
| `nonclk` | 1 if not clicked | always `1 - clk` (0 contradictions) |

0 exact duplicate rows.

**`ad_feature` — one row per ad.** `adgroup_id` is a unique key.

| Column | Distinct | Nulls |
|---|---:|---:|
| `cate_id` (category) | 6,769 | 0 |
| `campaign_id` | 423,436 | 0 |
| `customer` (advertiser) | 255,875 | 0 |
| `brand` | 99,815 | **246,330 (29.1%)**, stored as the text `NULL` |
| `price` | 14,861 | 0 |

**`user_profile` — one row per user.** `userid` is a unique key. Covers 94.2% of impressions;
the other 5.8% come from users with no profile row.

| Column | Distinct | Nulls |
|---|---:|---:|
| `cms_segid`, `cms_group_id` | 97, 13 | 0 |
| `final_gender_code` | 2 | 0 |
| `age_level` | 7 | 0 |
| `pvalue_level` (spending level) | 4 | **575,917 (54.2%)** |
| `shopping_level` | 3 | 0 |
| `occupation` (college student or not) | 2 | 0 |
| `new_user_class_level` | 5 | **344,920 (32.5%)** |

---

## Time, and the split

Calendar days are cut at midnight **Asia/Shanghai** (UTC+8), the platform's local time. Cutting at
UTC midnight would move 8 hours of every day across the train/validation/test boundaries.

| Split | Days | Impressions | Users | Ads | Click rate |
|---|---|---:|---:|---:|---:|
| Train | 2017-05-06 to 05-11 | 20,015,245 | 1,006,337 | 755,976 | 5.197% |
| Validation | 2017-05-12 | 3,234,051 | 385,925 | 338,688 | 4.930% |
| Test | 2017-05-13 | 3,308,665 | 391,741 | 331,994 | 5.032% |

Per day, impressions range from 3.23M to 3.43M and the click rate from 4.93% to 5.26%. The
validation day has the lowest click rate of the eight, so calibration measured on it will show
drift compared with training.

Of test-day impressions, 88.3% come from users who also appear on a training day, and 91.2% are
for ads that also appear on a training day. The rest are new users or new ads at test time.

---

## Structure that matters for leakage

**Ads arrive in page loads.** 98.7% of impressions share their exact second with at least one
other impression for the same user. Those same-second groups are 6,572,777 in total, mostly of
size 3 (4,644,411 groups) or 10 (925,938 groups), never mix placements, and never repeat an ad
within a group. 1,159,250 groups contain at least one click; 174,709 contain more than one.

Consequences:
- An impression's history is **strictly earlier seconds only**. Ads in the same page load are not
  each other's history.
- No feature may reveal whether another ad in the same page load was clicked; that is
  information from the same moment as the label.

---

## How much history there is

History here means the user's earlier ad impressions (which ad, and whether it was clicked),
counted at each impression from strictly earlier seconds, across all 8 days.

| | Earlier impressions |
|---|---:|
| Mean | 100.6 |
| 10th / 25th percentile | 0 / 6 |
| Median | 26 |
| 75th / 90th percentile | 97 / 298 |
| 99th percentile | 941 |
| Max | 3,749 |

| At least ... earlier impressions | Share of impressions |
|---:|---:|
| 1 | 86.0% |
| 16 | 59.5% |
| 32 | 45.5% |
| 64 | 32.0% |
| 128 | 21.1% |
| 256 | 11.9% |
| 512 | 4.4% |
| 1024 | 0.8% |

Impressions per user over the 8 days: median 7, mean 23.3, 90th percentile 48, max 3,756. 44,260
users have exactly one impression.

### History differs by split

Later days have more history behind them, simply because a user's past keeps growing over the
week (`results/phase1/sequences.json`):

| Split | Median earlier impressions | 90th pct | None | At least 64 | At least 256 | At least 512 |
|---|---:|---:|---:|---:|---:|---:|
| Train (May 6-11) | 20 | 236 | 16.5% | 27.9% | 9.1% | 2.7% |
| Validation (May 12) | 47 | 471 | 6.7% | 42.9% | 19.4% | 8.8% |
| Test (May 13) | 52 | 536 | 6.4% | 45.5% | 21.4% | 10.7% |

So models train mostly on short histories and are evaluated on longer ones. Effects of history
length should show more strongly on validation and test than training numbers suggest. Any
statement about "how many impressions have at least L earlier impressions" must say which split
it means.

---

## Sparsity

How often each ad and user appears on training days (`results/phase1/sequences.json`):

| | Distinct | Median training impressions | Seen once | Seen under 10 times | Share of training impressions from those seen under 10 times |
|---|---:|---:|---:|---:|---:|
| Ads | 755,976 | 3 | 31.1% | 69.2% | 7.1% |
| Users | 1,006,337 | 6 | 4.0% | 61.8% | 13.6% |

Most ads are rare, but rare ads make up a small share of traffic: the 69% of ads seen fewer than
10 times account for 7% of training impressions. An embedding learned for an ad seen once or twice
is mostly noise, so ad category, brand and advertiser carry much of the signal for the long tail.

---

## Encoding used for modelling

`rewind/data/sequences.py` re-codes every ID column to small integers using vocabularies built
from **training days only**: 0 = padding, 1 = missing, 2 = not seen in training, 3+ = seen values.
On the test day, 8.8% of impressions are for an ad never seen in training (6.7% on validation) and
get code 2, as a new ad would in production. Profile fields are a static per-user table and are
coded from the full table; 6.3% of test impressions have no profile row.

Rebuild with `uv run python -m rewind.data.sequences` (about 45 s, 2.4 GB of memory, 2.2 GB on disk
in `data/sequences/`). The build refuses to finish unless, on the full data, every history consists
only of the same user's impressions from strictly earlier seconds.

---

## Known limitations

1. **No behaviour log.** History is ad impressions only, over at most 8 days. Results say how much
   *ad history* is worth, not how much general shopping history is worth.
2. **Click is the only label.** There are no conversion, cart or purchase labels, which limits any
   multi-task experiment.
3. **Heavy missingness in two profile fields** (`pvalue_level` 54%, `new_user_class_level` 32%) and in
   ad `brand` (29%). Missing is kept as its own value, never filled with a guess.
4. **Third-party copy with an unknown license.** It matches the documentation on every count
   checked, and its checksums are recorded above.
5. **2017 data, one platform, one week.** Patterns may not carry over to other platforms or periods.
6. **The profile table is undated.** It is a single snapshot with no timestamp, so it cannot be
   ruled out that some fields (e.g. `shopping_level`) were computed over a period overlapping the
   impressions. It is used as given, and any result that depends heavily on profile fields should
   be read with that in mind.
