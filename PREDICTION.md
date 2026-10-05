# PREDICTION — written before any GPU is rented

**Date written:** 2026-10-04. **No real-data model has been trained yet.** Everything below is
either a laptop measurement on synthetic inputs, or a guess made in advance so it can be checked.
Being wrong here is expected and will be reported, not edited away.

**Amended 2026-10-04, same day, before any model touched real data:** the dataset turned out to
lack its behaviour log, which changes the sweep and several predictions. Sections 1-4 are left
exactly as first written; **section 5 records what changed and which predictions it replaces.**

---

## 1. What was measured

**Hardware lane: Apple M5, 16 GB unified memory, PyTorch 2.14.1, MPS backend.** These numbers are
never combined with rented-GPU numbers.

Model: the Phase 3 sequence model at its starting size (2 layers, width 64, 2 heads, 262,144 hashed
item buckets). 16.9M parameters, of which **16.8M (99.4%) are the item embedding table** and 0.1M
are the transformer itself. Inputs are synthetic and every history is full length, so these are
worst-case costs per length. Compute cost depends on tensor shapes, not values, so synthetic inputs
cost what real ones do. They say nothing about quality.

Training cost per example, fastest step to median step over 20 timed steps
(`results/phase0/cost_m5-mps_b32.json`, `results/phase0/cost_m5-mps_b8.json`):

| History length | Batch | ms per example | Cost vs. previous length |
|---:|---:|---:|---:|
| 16 | 32 | 0.46 - 0.48 | — |
| 32 | 32 | 0.47 - 0.50 | x1.0 |
| 64 | 32 | 0.48 - 0.49 | x1.0 |
| 128 | 32 | 0.58 - 0.64 | x1.2 |
| 256 | 32 | 0.87 - 0.93 | x1.5 |
| 512 | 32 | 1.93 - 2.15 | x2.2 |
| 1024 | 32 | 6.11 - 7.19 | x3.2 |
| 2048 | 8 | 22.7 - 26.2 | x3.7 (smaller batch; see note) |

L = 2048 did not fit in memory at batch 32, so it was measured at batch 8. Smaller batches use
the GPU less efficiently, so the x3.7 step slightly overstates the true growth.

**What the shape says:**

- **Up to ~128, history is nearly free.** The fixed cost of every step (looking up and updating the
  16.8M-entry embedding table) dominates. Cost per doubling is x1.0-1.2.
- **Past ~256, attention takes over.** Cost per doubling climbs x1.5, x2.2, x3.2, x3.7, heading
  for the x4 that pure attention (every token compared with every other) would give.
- **The laptop cannot train past ~1024 at batch 32 or ~2048 at batch 8.** Above that, memory runs out
  and the OS swaps. This is the memory wall that a memory-efficient attention implementation
  (Phase 7) would move.

No single "crossover length" is claimed. A quadratic fitted to these timings put the point where
attention overtakes the rest at L = 220 on one run and L = 59 on the next, so the fit is too
sensitive to timing noise to support that number; the per-doubling ratios above are the finding.
An earlier, noisier run of the same code (`BUGLOG.md` #2-#5) gave ranges 2-3x wider; these figures
come from a rerun at commit `e5ec757` with a clean working tree.

---

## 2. Predicted cost of the Phase 4 sweep

**Plan:** lengths 16, 32, ..., 2048 (8 lengths) x 3 seeds x ~19.5M training examples x 1 epoch,
plus 50% for validation passes, failed runs and restarts.

**Predicted: 55 - 188 A100 hours, $55 - $470, mid estimate ~94 hours and ~$164.**
(`results/phase0/projection.json`)

| Length | A100 hours |
|---:|---:|
| 16 - 512 (six lengths together) | 8 - 25 |
| 1024 | 10 - 35 |
| 2048 | 37 - 128 |

**L = 2048 alone is about 68% of the bill.**

**Assumptions, weakest first:**

1. **A100 is 5-15x faster than the laptop for this model.** Not measured. Rationale: an A100 80GB
   has ~13x the M5's memory bandwidth, and a model this narrow is limited more by moving data than by
   arithmetic. The low end allows for poor GPU utilisation at width 64.
2. **~19.5M training examples.** From the dataset's published size (~26M ad impressions over 8 days,
   6 days used for training). **Not yet counted**; Phase 1 counts it and this projection is rerun.
3. **A100 80GB at $1.00-2.50 per hour.** Confirm on the day of rental.
4. **No length bucketing.** Every example is padded to the full length, so real data costs the same
   as these worst-case figures. Grouping examples by history length would cut cost if most histories
   are short.

**Rules decided now, before seeing any rented-GPU number:**

- **First rented hour:** re-measure L = 128 and L = 1024 on the A100. If the speedup is below 5x, or
  the re-projected total exceeds $500, stop and re-plan before launching the sweep.
- **L = 2048 runs only if going from 512 to 1024 improved validation NE by more than the spread
  across seeds.** If 1024 bought nothing, 2048 is not worth two thirds of the budget, and saying so is
  itself the Phase 4 result.

---

## 3. Predicted results

Made before any real-data training. Dataset: the public Taobao display-ad click log (ad
impressions with click labels, plus each user's earlier browsing, cart, favourite and purchase
actions). Comparisons are only against this repo's own baselines on the same split.

**P1 — Does the sequence model beat the best tabular model at matched parameters?**
**Predicted: a small win.** AUC +0.002 to +0.008 over the strongest tabular baseline (expected to
be DCN), best guess +0.004. Normalized entropy 0.3-1.0% lower. Confidence: win 55%, tie (within
seed spread) 35%, loss 10%.
*Why so small:* the behaviour log records categories and brands, not individual items, so the
sequence is coarse; and a tabular model with good aggregates (user-by-category counts, recency)
already captures much of what a short history says. Ad clicks are also inherently noisy: I expect
every model's absolute AUC to land in roughly 0.62-0.66.

**P2 — Calibration.**
**Predicted: no meaningful difference between the two families.** Both are trained on log loss at
the natural click rate, so both should be reasonably calibrated on validation (ECE below 0.006).
Both will drift by a similar amount on the test day, because the click rate moves from day to day.
Difference in ECE between best sequence and best tabular model: under 0.002.

**P3 — Quality vs. history length (the main question).**
**Predicted: most of the gain arrives by length 64, and the curve is flat by 128-256.** Past 256,
each doubling adds less than 0.001 AUC (inside seed noise) while costing x2-x3.5 more to train.
**Predicted recommended operating point: 128.**
*Why:* what someone browsed in the last session or two says far more about the next click than
what they browsed two weeks ago, and the coarse category/brand tokens saturate quickly.

**P4 — Per-user ranking (GAUC).**
**Predicted: the sequence model's GAUC gain is larger than its global AUC gain**, by roughly 1.5x,
because history mainly helps rank ads *within* one user rather than across users.

**P5 — The data itself.**
**Predicted: median history before an impression ~200 actions, with at least 25% of impressions
having more than 1024.** If this is badly wrong (most histories short), the long end of the sweep is
mostly padding and the length question partly answers itself.

**P6 — Serving cost.**
**Predicted: at length 128, the sequence model costs 10-50x more per million predictions than the
tabular model on the same GPU.** If P1 holds (a small win), that ratio is the honest headline.

---

## 4. Known risks, found while writing this

- **What does "matched parameter count" mean?** The embedding table is 99.4% of this model's
  parameters. Matching *total* parameters would mostly match embedding tables and let the dense
  parts differ widely. Phase 3 must report embedding and dense parameters separately and match
  both, or state which one it matched and why.
- **Tokens are categories and brands, not items.** The behaviour log in this dataset has no item
  IDs. The model's input layer needs adapting in Phase 1; the cost figures above are unaffected,
  because shapes don't change.
- **Phase 5 needs more than one target.** The impression log labels clicks only. Multi-task
  targets would have to come from the behaviour log (e.g. a later cart or purchase in the ad's
  category), and that must be defined without leaking the future.
- **Synthetic-only so far.** Phase 0 also asks for one small real-data run to confirm that real
  inputs cost what synthetic ones do. **That run is still outstanding and must happen before any GPU
  is rented**; it needs the dataset downloaded.

---

## 5. Amendment — 2026-10-04, after counting the data, before any training

### What changed

The only copy of the Taobao display-ad log obtainable without entering payment details is a
third-party mirror on Kaggle. It contains the impressions, ad features and user profiles, but
**not the 22-day behaviour log** (browse / cart / favourite / buy). Counted in
`results/phase1/counts.json`; described in `DATASET.md`.

So a user's history can only be **the ads they were shown earlier, and whether they clicked
them**, over at most 8 days. Counted at the moment of each impression (strictly earlier seconds
only):

| Earlier impressions | Value |
|---|---:|
| Median | 26 |
| 90th percentile | 298 |
| 99th percentile | 941 |
| Share of impressions with none | 14.0% |
| Share with at least 16 / 32 / 64 | 59.5% / 45.5% / 32.0% |
| Share with at least 128 / 256 / 512 | 21.1% / 11.9% / 4.4% |
| Share with at least 1024 | 0.8% |

### New sweep and cost

**Plan:** lengths 16, 32, ..., 512 (6 lengths) x 3 seeds x **20,015,245** training impressions
(counted: 2017-05-06 to 05-11) x 1 epoch, plus 50% overhead. Lengths past 512 are dropped: a length-1024 run could differ from
the 512 run only on the 4.4% of impressions with more than 512 earlier impressions, and would pay
for padding on the other 95.6%.

**Predicted: 8 - 26 A100 hours, $8 - $65, mid estimate ~13 hours and ~$23.**
(`results/phase0/projection.json`, same laptop measurements as section 1, same assumptions as
section 2.) On the laptop alone the sweep would take roughly 120-130 hours.

The section 2 rule about L = 2048 no longer applies. The first-rented-hour rule stands, with the
stop threshold lowered from $500 to **$100**, in proportion to the smaller plan.

Assumption 4 (no length bucketing) now matters more: with a median of 26 earlier impressions,
padding every example to 512 wastes most of the compute at the long end. Bucketing is a cost
optimisation that does not change results, so it may be added before renting without amending
these predictions.

### Predictions: which stand, which are replaced

**P1 — replaced.** The history is now the same kind of information the tabular baseline's
aggregates are built from (past ads and clicks), and it is short. **Predicted: a tie or a very
small win.** AUC +0.001 to +0.006 over the strongest tabular baseline, best guess +0.003; NE
0.1-0.6% lower. Confidence: win 45%, tie 45%, loss 10%. The expected absolute AUC range
(0.62-0.66) stands.

**P2 — stands.** Additional detail from the counts: the click rate on the validation day is 4.93%
and on the test day 5.03%, against 5.14% overall, so both families should show a small, similar
calibration drift between days.

**P3 — replaced.** **Predicted: most of the gain arrives by length 32-64, and the curve is flat by
128** when measured over all impressions. **Recommended operating point: 64.** Going past 128 can
only change predictions for the 21% of impressions with more than 128 earlier impressions, so the
overall curve flattens partly by construction. To keep that from hiding a real effect, quality will also be reported for
impressions with at least 256 earlier impressions; **predicted: within that group, 64 -> 512 still
gains about +0.003 AUC.**

**P4 — stands.**

**P5 — cannot be scored as written, recorded as a miss in spirit.** It predicted a median of ~200
earlier *behaviour-log actions*, which this data does not have. The nearest available quantity,
earlier *ad impressions*, has a median of 26, about 8x lower than the predicted history size. It is
not edited away.

**P6 — stands,** evaluated at the recommended length (now 64) instead of 128.

**New — P7: impressions with no history.** 14% of impressions have no earlier impression. There the
sequence model has only the candidate ad to go on. **Predicted: on that slice the sequence model is
no better than the tabular model, and possibly slightly worse** (the tabular model still has user
profile features; the sequence model as planned does not). This will be reported as its own row.

### Risks: updated

- **Resolved:** "tokens are categories and brands, not items." Past impressions carry the real ad
  ID, plus its category, brand, campaign and advertiser from `ad_feature`.
- **New: ads arrive in page loads.** 98.7% of impressions share their exact second with other
  impressions for the same user (groups of mostly 3 or 10 ads). Ads from the same page load must
  never be each other's history, and no feature may reveal whether a neighbouring ad on the same
  page was clicked.
- **New: Phase 5 has no second label.** The impression log labels clicks only, and without the
  behaviour log there are no cart or purchase targets. Phase 5 needs rethinking (for example the two
  ad placements, `pid`, as two tasks); to be decided with the project owner before Phase 5 starts.
- **Stands:** what "matched parameter count" means (embedding vs dense parameters).
- **Stands:** the real-data cost check is still outstanding and must happen before any GPU is
  rented.

---

## 6. Real-data cost check — 2026-10-04, closes Phase 0

The outstanding check from sections 4 and 5. Real training batches (built from the Phase 1
sequences, training days only) and synthetic batches of the same shape were timed alternately in
one process, same model and config as section 1. **Pass rule, fixed in the script before the
first run: real/synthetic ratio of median step time within 0.8-1.25 at every length.**
(`results/phase0/real_check_m5-mps_b32.json`, `rewind/phase0/real_check.py`)

| Length | Synthetic ms/step | Real ms/step | Ratio | Batch assembly ms | Padding share |
|---:|---:|---:|---:|---:|---:|
| 16 | 15.9 | 16.1 | 1.01 | 8.0 | 30.2% |
| 32 | 17.4 | 17.0 | 0.98 | 6.6 | 43.8% |
| 64 | 17.1 | 17.0 | 0.99 | 6.0 | 54.9% |
| 128 | 19.2 | 19.3 | 1.01 | 5.1 | 63.9% |
| 256 | 29.2 | 29.5 | 1.01 | 5.0 | 77.6% |
| 512 | 67.9 | 66.0 | 0.97 | 4.8 | 85.0% |

**Result: PASS.** Real inputs cost what synthetic ones do, to within 3%, so the cost projection in
section 5 stands unchanged. The training loss on real batches was finite at every length.

**Two findings for the rented-GPU runs (no prediction changes):**

- **Data loading could become the bottleneck.** Building a batch of 32 histories on the CPU takes
  5-8 ms. On the laptop that is well under the 16-68 ms training step, but at the assumed 5-15x
  speedup an A100 step at short lengths would take roughly 1-4 ms, faster than one CPU thread can
  feed it. The Phase 4 training loop needs parallel batch assembly (several worker processes) or
  larger batches, and the first-rented-hour check must time the full loop, not the model alone.
- **Most history slots are padding** (training days): 55% at length 64, 85% at 512. Grouping
  examples by history length would cut the long-length runs substantially. As noted in section 5,
  that is a cost optimisation that leaves results unchanged.

**Phase 0 status: complete.** Every item in its definition of done is in this file: measured rates,
the extrapolated sweep cost, the predicted answers, and a real-data confirmation, all recorded
before any GPU was rented or any model was evaluated on real data.

