"""Phase 0, last step: does training on REAL sequences cost what the synthetic benchmark said?

For each history length, real batches (built from data/sequences/ with history_window) and
synthetic batches of the same shape are timed back to back in the same process, so background
load on the machine affects both equally. Same model and config as rewind.phase0.cost.

Pass rule, fixed before the first run: at every length, the real/synthetic ratio of median
training-step time lies within 0.8 - 1.25. The model pads every history to full length, so the
two should cost the same; a failure would mean the projection in PREDICTION.md is off.

Also measured, because they matter for the rented GPU:
- batch assembly time on the CPU (gathering histories from disk), against the training step
- the share of history slots that are padding, i.e. compute that length bucketing could save

Model inputs from the real data: the ad code is folded into the model's 2^18 item buckets, and
each history item's action is 1 = shown, not clicked / 2 = clicked. Only training-day rows are
used; nothing from validation or test is read.

    uv run python -m rewind.phase0.real_check --device mps
"""

import argparse
import gc
import json
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from rewind.data.sequences import SPLITS, history_window, load
from rewind.model import SeqConfig, SequenceCTR
from rewind.phase0.cost import empty_cache, git_info, hardware_info, summarize, sync

PASS_RANGE = (0.8, 1.25)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--device", default="mps", choices=["mps", "cpu", "cuda"])
    ap.add_argument("--lengths", type=int, nargs="+", default=[16, 32, 64, 128, 256, 512])
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--data-dir", type=Path, default=Path("data/sequences"))
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    device = torch.device(args.device)
    hw = hardware_info(device)
    git = git_info()
    out = args.out or Path("results/phase0") / f"real_check_{hw['lane']}_b{args.batch_size}.json"

    cols = load(args.data_dir)
    train_rows = np.flatnonzero(np.asarray(cols["split"]) == SPLITS["train"])
    rng = np.random.default_rng(args.seed)
    cfg_base = SeqConfig()

    print(f"lane: {hw['lane']}  ({hw['chip']}, torch {hw['torch']})")
    print(f"{len(train_rows):,} training rows; batch {args.batch_size}, "
          f"{args.steps} timed steps after {args.warmup} warmup, seed {args.seed}\n")

    # Absorb one-off GPU start-up cost before anything is measured (BUGLOG #4).
    measure(min(args.lengths), args, device, cols, train_rows, rng, cfg_base)

    print(f"{'len':>5} {'synthetic ms':>13} {'real ms':>9} {'ratio':>6} {'assemble ms':>12} {'padding':>8}")
    rows = []
    for L in args.lengths:
        r = measure(L, args, device, cols, train_rows, rng, cfg_base)
        rows.append(r)
        print(f"{L:>5} {r['synthetic']['ms_median']:>13.1f} {r['real']['ms_median']:>9.1f} "
              f"{r['ratio_median']:>6.2f} {r['assemble_ms_median']:>12.2f} {r['padding_share']:>7.1%}"
              + ("" if r["within_pass_range"] else "  OUTSIDE PASS RANGE"))

    passed = all(r["within_pass_range"] for r in rows)
    print(f"\nPass rule {PASS_RANGE[0]}-{PASS_RANGE[1]} at every length: {'PASS' if passed else 'FAIL'}")

    result = {
        "experiment": "phase0_real_vs_synthetic_cost",
        "pass_range": PASS_RANGE,
        "passed": passed,
        "hardware": hw,
        "git": git,
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "config": {"batch_size": args.batch_size, "warmup_steps": args.warmup, "timed_steps": args.steps,
                   "seed": args.seed, "model": {k: v for k, v in vars(cfg_base).items() if k != "max_len"}},
        "rows": rows,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(f"wrote {out}")


def measure(L, args, device, cols, train_rows, rng, cfg_base) -> dict:
    torch.manual_seed(args.seed)
    cfg = SeqConfig(**{**vars(cfg_base), "max_len": L})
    B, n = args.batch_size, args.warmup + args.steps

    # Real batches: assembled on the CPU (timed), then moved to the device before training is timed.
    assemble_ms, real, padding = [], [], []
    for _ in range(n):
        t0 = time.perf_counter()
        batch, pad_share = real_batch(cols, rng.choice(train_rows, B, replace=False), L, cfg.n_items)
        assemble_ms.append((time.perf_counter() - t0) * 1000)
        real.append([t.to(device) for t in batch])
        padding.append(pad_share)

    synthetic = [
        [torch.randint(1, cfg.n_items, (B, L)), torch.randint(1, 3, (B, L)),
         torch.randint(1, cfg.n_items, (B,)), torch.randint(0, 2, (B,)).float()]
        for _ in range(n)
    ]
    synthetic = [[t.to(device) for t in b] for b in synthetic]

    model = SequenceCTR(cfg).to(device).train()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    losses = []

    def step(batch):
        items, actions, target, labels = batch
        loss = F.binary_cross_entropy_with_logits(model(items, actions, target), labels)
        loss.backward()
        opt.step()
        opt.zero_grad(set_to_none=True)
        return loss

    # Alternate synthetic and real steps so both see the same machine conditions.
    times = {"synthetic": [], "real": []}
    for i in range(n):
        for kind, batches in (("synthetic", synthetic), ("real", real)):
            sync(device)
            t0 = time.perf_counter()
            loss = step(batches[i])
            sync(device)
            if i >= args.warmup:
                times[kind].append((time.perf_counter() - t0) * 1000)
                if kind == "real":
                    losses.append(loss.item())

    if not all(np.isfinite(losses)):
        raise SystemExit(f"L={L}: non-finite loss on real data: {losses}")

    s, r = summarize(times["synthetic"], B), summarize(times["real"], B)
    ratio = r["ms_median"] / s["ms_median"]
    del model, opt, real, synthetic
    gc.collect()
    empty_cache(device)
    return {
        "seq_len": L,
        "synthetic": s,
        "real": r,
        "ratio_median": round(ratio, 3),
        "within_pass_range": PASS_RANGE[0] <= ratio <= PASS_RANGE[1],
        "assemble_ms_median": round(statistics.median(assemble_ms[args.warmup:]), 3),
        "padding_share": round(float(np.mean(padding)), 4),
    }


def real_batch(cols, rows: np.ndarray, L: int, n_items: int):
    """Model inputs for training rows: history (ad + clicked?) and the candidate ad, plus labels."""
    rows = np.sort(rows)  # sorted indices read the memory-mapped arrays more sequentially
    pos, real = history_window(cols, rows, L)
    ad, clk = cols["adgroup_id"], cols["clk"]
    hist_items = np.where(real, fold(ad[pos], n_items), 0)
    hist_actions = np.where(real, clk[pos].astype(np.int64) + 1, 0)
    target = fold(ad[rows], n_items)
    labels = clk[rows].astype(np.float32)
    batch = [torch.from_numpy(np.ascontiguousarray(a)) for a in (hist_items, hist_actions, target, labels)]
    return batch, 1.0 - real.mean()


def fold(codes: np.ndarray, n_items: int) -> np.ndarray:
    """Fold ad codes (up to ~756K) into the model's buckets, keeping 0 for padding."""
    codes = codes.astype(np.int64)
    return np.where(codes > 0, (codes - 1) % (n_items - 1) + 1, 0)


if __name__ == "__main__":
    main()
