# Design

What the models are, where they come from, and the decisions that keep the comparison fair. The
models are standard published designs; the contribution of this repo is the measurement.

---

## The question and the one variable

Every model gets the same information about the impression being scored:

- **Candidate ad:** ad, category, brand, campaign, advertiser, placement (`pid`), price.
- **User profile:** the eight `user_profile` fields.
- **Context:** how often this ad, its category and its advertiser were shown and clicked across
  all users, in strictly earlier seconds.

What differs between model families is **only how the user's own past ad impressions enter**:

- **Tabular (Phase 2):** compressed into counts, click rates and recency: overall, in the last
  hour and day, and per category, brand, ad and advertiser.
- **Sequence (Phase 3):** the ordered list of past impressions, read by a transformer.

User IDs are not a model input for either family.

## Leakage rules

Every feature and every history uses only impressions from **strictly earlier seconds** than the
impression being scored. Same-second impressions are one page load and are excluded, including
other users' impressions in that second for the global counts. Vocabularies for ID embeddings
come from training days only. Normalization statistics come from training rows only. The test
day (2017-05-13) is not read by any tuning or training code.

How each rule is enforced and tested: `DATASET.md`, `rewind/data/sequences.py`
(`check_invariants`, run on the full data at every build), and `tests/`.

## Tabular baselines (Phase 2)

| Model | Source | Implementation |
|---|---|---|
| Logistic regression | — | one weight per categorical value and per numeric feature |
| DeepFM | Guo et al., 2017, *DeepFM: A Factorization-Machine based Neural Network for CTR Prediction* | first-order terms + factorization machine over all field embeddings + MLP (256, 128) |
| DCN | Wang et al., 2021, *DCN V2: Improved Deep & Cross Network and Practical Lessons for Web-scale Learning to Rank Systems* | 3 full-rank cross layers in parallel with an MLP (256, 128), combined linearly |

Numeric features enter the factorization machine and cross layers as their own fields: a learned
vector per feature, scaled by the feature's value.

## Training and tuning protocol

Fixed before any run, identical for every model family including Phase 3:

- One epoch over the 20.0M training impressions, Adam, batch 4,096.
- Up to 4 configurations per family, tried with seed 0.
- Selection on **validation log loss** (NE is log loss rescaled; the project cares about
  probability quality, not ranking alone).
- The selected configuration is repeated with seeds 1 and 2; results are reported as the mean
  and min-max over 3 seeds.

## Metrics

AUC, GAUC (per-user AUC weighted by impressions; users with only clicks or only non-clicks are
skipped), normalized entropy, ECE over 20 equal-count bins, and the calibration ratio (mean
prediction / observed click rate). Definitions and tests: `rewind/metrics.py`,
`tests/test_metrics.py`. Every result reports all five, on all validation impressions and on two
slices: impressions with no earlier impression, and with 256 or more.

## Sequence model (Phase 3)

Behavior Sequence Transformer: Chen et al., 2019, *Behavior Sequence Transformer for E-commerce
Recommendation in Alibaba*. `rewind/sequence/model.py`, built on the tested attention block and
mask in `rewind/model.py`.

- **Input:** the user's last 64 earlier impressions (strictly earlier seconds), oldest to newest,
  left-padded, then the candidate ad.
- **One history token** = a projection of the 16-wide ID embeddings (ad, category, brand,
  campaign, advertiser, placement) + an action embedding (shown-not-clicked / clicked) + a
  time-gap embedding (13 buckets, under a minute to 4+ days) + a position embedding counted
  backwards from the candidate. The candidate token has its own action value and no time gap.
- **Transformer:** pre-norm blocks, causal attention that also ignores padding.
- **Output:** the candidate's output vector, concatenated with the profile embeddings and the
  context features (the same `ctx_*` features the tabular models get), through an MLP to one logit.
- **Not an input:** the tabular history totals (`hist_*`). The ordered timeline replaces them.

**Matched to DCN:** the ID embedding tables are the same vocabularies at the same width, so the
two families have identical tables (23,849,952 parameters). The prediction head is sized so the
dense parameters match DCN's 1,364,049 to within 0.1%. The extra action, time-gap and position
tables (5-11K parameters) are reported separately.

**Batches are built on the GPU** (`rewind/sequence/data.py`): one compact per-impression table plus
the Phase 1 bookmarks live in device memory, and each batch is a few indexing operations. Leading
columns that are padding in every row are dropped; because positions count backwards from the
candidate, this changes no prediction (tested).

**Tuning:** the same protocol and budget as the baselines (`rewind/sequence/run.py`): 4
configurations (learning rate {1e-3, 3e-4} x {width 64 / 2 layers, width 128 / 3 layers}), seed 0,
selected on validation log loss, then 2 more seeds.

## The test-day comparison

`rewind/compare/final.py` scores 3 sequence and 3 DCN checkpoints on the same rows. Per metric:
**win** if the sequence model's worst seed beats DCN's best seed, **loss** if the reverse, **tie** if
the ranges overlap. Opening the test day is recorded in `results/phase3/test_day_opened.json`, and a
second opening is refused unless a reason is given and logged. Run procedure: `GPU.md`.
