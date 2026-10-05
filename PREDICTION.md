# PREDICTION — written before any GPU is rented

**Date written:** 2026-10-04. **No real-data model has been trained yet.** Everything below is
either a laptop measurement on synthetic inputs, or a guess made in advance so it can be checked.
Being wrong here is expected and will be reported, not edited away.

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
