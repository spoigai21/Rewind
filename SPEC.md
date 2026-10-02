# Rewind — does replaying a user's recent actions beat a feature-crossing model for ad ranking?

*(named 2026-10-01: you rewind the user's recent actions and replay them to predict the next one.
Earlier working name was AdSeq, dropped because AdFill / AdSeq / AdServe all read alike at a glance.)*

**One line:** predict the next ad action from a user's sequence of past actions with a
transformer, measure it honestly against the tabular feature-crossing models that ads ranking has
used for a decade, and find out what sequence length is actually worth in quality and in dollars.

---

## Why it exists

Ad ranking has been dominated for a decade by **feature-crossing models** — logistic regression,
FM, DeepFM, DCN. They take a flat row of features about one impression and predict a click. They
are fast, well understood, and they throw away order: everything the user did before this moment
is compressed into a handful of aggregate counts.

The field is now moving to **sequence models** — feed the model the user's actual ordered history
and let attention decide what matters. Large production ad systems have been moving this way
publicly.

**Nobody has shown you, on data you can check, whether this is worth it.** Papers report wins on
private data at a scale you cannot reproduce. This project asks the smaller and more answerable
question: *on a real public log, with a fair baseline and an honest split, does sequence modelling
beat feature crossing — and how much sequence do you actually need before it stops paying?*

That last clause is the project. It is also why it needs a GPU: attention cost grows fast with
sequence length, so "how much history is worth keeping" is a question you buy answers to.

**Where it sits next to the others.** ClickRank benchmarks the feature-crossing family and adds
retrieval; this deliberately treats that family as the **baseline to beat** rather than the
subject. AdFill consumes predictions and decides allocation; this produces them. Fusion Bench
wrote the online-softmax kernel that makes long-sequence attention tractable — that connection is
real and worth stating, but the kernel is not re-litigated here.

---

## Honest limits, written before any results

1. **This is a public e-commerce log, not an ads platform's internal data.** No result here is a
   claim about any company's ranking system.
2. **Absolute numbers are not comparable to published papers.** Different data, different scale,
   bounded hash space. The only comparisons that mean anything are **against the baselines in this
   repo, on the same split, with the same budget.**
3. **Every figure carries its hardware.** Laptop and rented-GPU numbers are two lanes and are
   never combined.
4. **Parameter count is controlled** in any quality comparison, or the comparison is confounded
   and says nothing. A bigger model winning is not a finding.

---

## Phase 0 — predict before renting

Same discipline as Fusion Bench and SceneSafe.

Train the smallest sequence model on a slice, on the laptop or a cheap GPU. Record throughput,
memory at a short sequence length, and the measured scaling as length doubles. **Write down the
predicted cost of the full sequence-length sweep and commit it** before renting anything.

**Done when:** a committed `PREDICTION.md` with measured rates, the extrapolation, and the
predicted answer to "does sequence beat tabular". Being wrong in public is the point.

---

## Phase 1 — data and sequences

**Question:** can a leak-free sequence dataset be built at all?

**Build:** user action sequences with timestamps from a real log. **Criteo cannot be used here** —
it has no user identifiers, which is exactly why ClickRank runs retrieval on Taobao instead. A
behaviour log with user IDs, item IDs, action types and timestamps is required.

**The discipline that matters:** sequences make leakage trivially easy. Any aggregate computed
over the full history, any item embedding trained on the test period, any sequence that includes
actions after the label, and the result is meaningless.

- strict **time-forward split**: train on days 1..n, validate on n+1, test on the final day
- every model choice made on validation **without looking at the test day**
- the label's own action never appears in its input sequence

**Measure:** sequence length distribution, action-type mix, sparsity, and how many users have
enough history to model at all.

**Done when:** a dataset card with counted values — not values copied from the source's README.
Criteo's documentation was wrong by roughly 10x during AdFill; assume nothing until counted.

---

## Phase 2 — the baseline that must be beaten

**Question:** how good is the decade-old approach on this exact data?

**Build:** logistic regression, DeepFM and DCN over the same examples, with history compressed the
way it conventionally is — counts, recency, aggregates.

This phase exists to make the project honest. A sequence model beating nothing is not a result.

**Measure:** AUC, **normalized entropy**, **calibration (ECE)**, and GAUC grouped by user.

NE and calibration are not decoration. Ads ranking cares about calibration because the prediction
is multiplied by a bid to price an auction — AdFill's central finding is that a model over-
predicting by 30% moved revenue 0.1% and broke 24 more advertiser commitments. Ranking quality and
calibration are different things and this project reports both throughout.

**Done when:** a baseline table with every metric, reproducible from one command.

---

## Phase 3 — the sequence model

**Question:** does order carry information the aggregates lose?

**Build:** a transformer over the user's action history — item and action-type embeddings, learned
or rotary positional encoding, causal masking so nothing sees its own future, a prediction head
per target action.

Keep the architecture ordinary and credited. The contribution here is the honest comparison, not
a novel block. Standard designs, cited in `DESIGN.md`, exactly as ClickRank credits its models.

**Measure:** the Phase 2 metric set, at matched parameter count against the strongest baseline.

**Done when:** a like-for-like table, and a plain statement of whether sequence modelling won,
lost, or tied — with the tie or loss reported just as prominently as a win would be.

---

## Phase 4 — how much history is worth keeping

**This is the core of the project and the reason a GPU is rented.**

**Question:** quality against sequence length, and what each extra token costs.

**Build:** the same model trained at a ladder of maximum sequence lengths, everything else fixed.

**Measure, at each length:**
- quality (NE, AUC, calibration)
- training time and peak memory
- inference latency and throughput
- **the marginal return: quality gained per doubling of length, against cost per doubling**

**The result to look for:** the length where the curve flattens. Attention cost grows quadratically
with length while information gain saturates, so there is a point past which more history is
expensive and useless. Finding that point on real data, with the cost attached, is the finding.

**Done when:** a quality-against-length curve with a cost curve beside it, and a recommended
operating point stated with its justification.

---

## Phase 5 — multiple objectives and calibration

**Question:** does one model predicting several actions beat separate models?

**Build:** a multi-task head predicting several action types at once, against independent
single-task models at matched total parameters.

Report in terms of **estimated action rates**, **normalized entropy** and **calibration** — the
quantities that matter when a prediction is multiplied by a bid.

**Measure:** per-task NE and calibration, total parameters and training cost, and whether any task
is harmed by sharing.

**Done when:** a multi-task vs single-task table, including any task that got worse.

---

## Phase 6 — what it costs to serve

**Question:** could this run in a latency budget, and at what price?

**Build:** export and sweep serving paths — PyTorch eager, ONNX Runtime, TensorRT at fp32 / fp16 /
int8 with calibration. Cache what can be cached across requests for the same user.

**Measure:** p50 and p99 latency and throughput per backend and precision at several batch sizes
and sequence lengths; **quality of each precision against exact fp32 on the same rows**; and
**dollars per million predictions** at real rented-GPU pricing.

**Baseline:** the Phase 2 tabular model's serving cost. If the sequence model is 2% better and 20x
more expensive to serve, that is the honest headline and it is a genuinely useful one.

**Done when:** a cost-per-million-predictions figure for both approaches, with hardware named.

---

## Phase 7 — optional: the kernel connection

Only after 1-6 are measured.

Fusion Bench implemented **online softmax, the FlashAttention primitive** — the technique that
makes long-sequence attention fit in memory by never materialising the full attention matrix.
Phase 4 is precisely where that matters.

Swap the attention implementation between a naive materialising version and a memory-efficient
one, and measure the sequence length reachable within the same memory budget.

This links the two GPU projects with a measurement rather than an assertion. It is optional
because Phases 1-6 stand alone.

---

## Data

- **Required shape:** user ID, item ID, action type, timestamp, over enough days for a
  time-forward split. A public e-commerce behaviour log is the realistic option; Taobao's
  behaviour data is the usual choice and is what ClickRank falls back to for user-level work.
- **Not usable:** Criteo — no user identifiers, so no sequences. State this explicitly; it is the
  reason the dataset choice differs from the tabular literature.
- **Count everything yourself.** See the Criteo README failure during AdFill.

---

## Measurement discipline

- `NUMBERS.md` ledger: every figure traces to the results file and seed that produced it.
- Every row carries hardware; laptop and GPU lanes never blended.
- Repeats with ranges, not single runs, for anything timed.
- Seeds fixed and stated; multiple seeds wherever a difference is small enough to be noise.
- A bug log with what caught each one. AdFill logged 11 and it is one of the most convincing
  things in that repo.
- Re-runnable from a fresh clone.

---

## Stack

Python · PyTorch · CUDA · ONNX · TensorRT · mixed precision · optionally a memory-efficient
attention path in Phase 7.

**Hardware:** develop small and local; rent an A100 for the Phase 4 sweep and the Phase 6 serving
grid. Checkpoint every epoch so a preempted spot instance costs minutes. Sequence datasets are
small on disk compared to video — storage is not the constraint here, GPU hours are.
