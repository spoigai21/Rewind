# Running on a rented GPU

Everything here was built and tested on a laptop first; the rented machine runs the same code with
`--device cuda`. Steps are in order. **Stop at every "CHECK" and confirm before continuing.**

What runs on the GPU (Phase 3):
- 6 sequence-model training runs: 4 tuning configurations + 2 extra seeds of the selected one
- 3 DCN retrains with their weights saved (Phase 2 kept only DCN's validation predictions)
- the final comparison, scored once on the test day — **only after the validation results have
  been reviewed**

---

## 1. Rent

- One NVIDIA A100 80GB (an H100 also works), Linux with CUDA drivers installed, about 100 GB disk.
- Note the hourly price: step 5 needs it.

## 2. Set up (about 10 minutes)

```bash
# on the rented machine
curl -LsSf https://astral.sh/uv/install.sh | sh && source ~/.local/bin/env
git clone https://github.com/spoigai21/Rewind.git rewind && cd rewind
uv sync                                    # exact library versions from uv.lock
uv run python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

**CHECK:** prints `True` and the GPU name.

## 3. Copy the raw data up (1.17 GB)

```bash
# on the laptop
scp ~/Desktop/rewind/data/raw/*.csv <user>@<host>:~/rewind/data/raw/
```

```bash
# on the rented machine
mkdir -p data/raw   # (before the scp, if needed)
sha256sum data/raw/*.csv
```

**CHECK:** the three checksums match `DATASET.md` exactly.

## 4. Rebuild everything from the raw files (about 10 minutes)

Rebuilding instead of copying the processed files doubles as a check that the repo reproduces from a
fresh clone.

```bash
uv run python -m rewind.data.count
uv run python -m rewind.data.sequences
uv run python -m rewind.features.tabular
git diff --stat results/phase1/
```

**CHECK:** `git diff` shows only the `commit` / `dirty` / `created_utc` lines changed in
`results/phase1/*.json`. Any changed count means the rebuild differs from the laptop: stop.

## 5. First-hour speed check (PREDICTION.md, section 5)

```bash
uv run python -m rewind.sequence.run --device cuda --speedcheck --price <USD per hour>
uv run python -m rewind.baselines.train --model dcn --lr 1e-3 --dim 16 --device cuda \
    --runs-dir /tmp/dcn_speed --preds-dir /tmp/dcn_speed   # note its examples/s, then Ctrl-C is fine
```

**CHECK:** the projection says `WITHIN BUDGET` (under $100). If not, stop and re-plan.

## 6. Train the sequence model (the long part)

```bash
nohup uv run python -u -m rewind.sequence.run --device cuda > phase3_sequence.log 2>&1 &
tail -f phase3_sequence.log
```

Runs that finish are saved and skipped on restart, so an interrupted run loses at most one run.
Output: `results/phase3/runs/`, `results/phase3/sequence.json`, `results/phase3/VALIDATION.md`,
checkpoints in `data/checkpoints/phase3/`.

## 7. Retrain DCN with its weights saved

The configuration selected in Phase 2 (lr 1e-3, embedding width 16), the same 3 seeds:

```bash
for s in 0 1 2; do
  uv run python -m rewind.baselines.train --model dcn --lr 1e-3 --dim 16 --seed $s --device cuda \
      --checkpoint-dir data/checkpoints/phase3_dcn \
      --runs-dir results/phase3/dcn_retrain --preds-dir data/preds/phase3_dcn
done
```

These numbers will not match Phase 2 to the last digit (different hardware), so the comparison
uses these retrained models and reports how close they came to the Phase 2 table.

## 8. Rehearse the comparison on the validation day

```bash
uv run python -m rewind.compare.final --split val --device cuda
```

Safe to repeat; it never touches the test day. Output: `results/phase3/final_val.md`.

**CHECK:** review `VALIDATION.md` and `final_val.md` together **before step 9**. Nothing about either
model may change after the test day is opened.

## 9. The test day — once

```bash
uv run python -m rewind.compare.final --split test --open-test-day --device cuda
```

Writes `results/phase3/final_test.md` and `results/phase3/test_day_opened.json`. A second run is
refused unless given `--rerun-reason "..."`, and every reason is kept in that file.

## 10. Bring the results home, then shut the machine down

```bash
# on the laptop
scp -r <user>@<host>:~/rewind/results/phase3 ~/Desktop/rewind/results/
scp -r <user>@<host>:~/rewind/data/checkpoints ~/Desktop/rewind/data/   # optional, ~1.2 GB
```

Then **stop and delete the instance** so billing ends. Commit the results from the laptop.
