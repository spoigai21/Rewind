"""Phase 3 end to end on the rented GPU: tune the sequence model on the validation day with the SAME
protocol and budget as the Phase 2 baselines (DESIGN.md), then repeat the selected configuration
over seeds. Everything is chosen on validation; the test day is never read.

Protocol, fixed before any run:
  1. Four configurations, seed 0, one epoch each, history length 64 (PREDICTION.md, P3 amended):
     learning rate {1e-3, 3e-4} x transformer {width 64 / 2 layers, width 128 / 3 layers}.
     Every configuration has DCN's embedding tables and its dense size matched to DCN's
     (within 0.1%) by sizing the prediction head.
  2. Select the lowest validation log loss.
  3. Repeat the selected configuration with seeds 1 and 2; report mean and min-max of 3 seeds.

Runs already saved in results/phase3/runs/ are reused, so an interrupted run picks up where it
stopped.

    uv run python -m rewind.sequence.run --device cuda --speedcheck --price 1.75   # first rented hour
    uv run python -m rewind.sequence.run --device cuda                             # the full protocol
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from rewind.baselines.run import METRICS, aggregate
from rewind.phase0.cost import git_info
from rewind.sequence.data import SequenceData
from rewind.sequence.train import RUNS_DIR, run_name, save_run, train_one

LENGTH = 64
GRID = [{"lr": lr, "d_model": d, "n_layers": n, "n_heads": h}
        for lr in (1e-3, 3e-4) for d, n, h in ((64, 2, 2), (128, 3, 4))]
SEEDS = [0, 1, 2]
STOP_ABOVE_USD = 100  # PREDICTION.md section 5: stop and re-plan above this projected total


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--speedcheck", action="store_true", help="time 200 batches and project the bill")
    ap.add_argument("--price", type=float, default=None, help="GPU price, USD per hour (speedcheck)")
    args = ap.parse_args()

    data = SequenceData(torch.device(args.device))
    if args.speedcheck:
        speedcheck(data, args.price)
        return

    git = git_info()
    tuning = [run(data, cfg, 0, git) for cfg in GRID]
    best = min(tuning, key=lambda r: r["metrics"]["val"]["log_loss"])
    cfg = {k: best[k] for k in ("lr", "d_model", "n_layers", "n_heads")}
    print(f"=== selected {cfg}; repeating with seeds {SEEDS[1:]} ===")
    seeds = [best] + [run(data, cfg, s, git) for s in SEEDS[1:]]

    dcn = json.loads(Path("results/phase2/baselines.json").read_text())["models"]["dcn"]
    summary = {
        "protocol": __doc__.split("Protocol, fixed before any run:")[1].split("Runs already")[0].strip(),
        "grid": GRID, "seeds": SEEDS, "seq_len": LENGTH, "selected": cfg, "params": best["params"],
        "tuning": [{**{k: r[k] for k in ("lr", "d_model", "n_layers")},
                    "val_log_loss": r["metrics"]["val"]["log_loss"], "val_auc": r["metrics"]["val"]["auc"]}
                   for r in tuning],
        "results": {split: aggregate([r["metrics"][split] for r in seeds]) for split in best["metrics"]},
        "dcn_phase2_validation": dcn["results"],
        "runs": [run_name(cfg["lr"], cfg["d_model"], cfg["n_layers"], LENGTH, s) for s in SEEDS],
        "hardware": best["hardware"], "git": git,
    }
    Path("results/phase3/sequence.json").write_text(json.dumps(summary, indent=2) + "\n")
    Path("results/phase3/VALIDATION.md").write_text(table(summary))
    print(table(summary))


def run(data: SequenceData, cfg: dict, seed: int, git: dict) -> dict:
    path = RUNS_DIR / f"{run_name(cfg['lr'], cfg['d_model'], cfg['n_layers'], LENGTH, seed)}.json"
    if path.exists():
        print(f"  reuse {path.name}")
        return json.loads(path.read_text())
    print(f"  train {path.name}", flush=True)
    result, preds, model = train_one(data, cfg["lr"], cfg["d_model"], cfg["n_layers"], cfg["n_heads"], seed, LENGTH)
    save_run(result, preds, model, git)
    v = result["metrics"]["val"]
    print(f"    val log loss {v['log_loss']:.5f}  AUC {v['auc']:.4f}  NE {v['ne']:.4f}  "
          f"({result['train_seconds'] / 60:.1f} min)", flush=True)
    return json.loads(path.read_text())


def speedcheck(data: SequenceData, price: float | None) -> None:
    """The first-rented-hour rule: measure real training speed on the rented GPU and project the
    bill for the Phase 3 sequence runs (4 tuning runs + 2 seeds). The 3 DCN retrains are timed
    separately (GPU.md), not assumed."""
    rates = {}
    for d, n, h in ((64, 2, 2), (128, 3, 4)):
        r, _, _ = train_one(data, 1e-3, d, n, h, seed=0, length=LENGTH, max_batches=200, log_every=0)
        rates[(d, n)] = r["train_examples_per_s"]
        print(f"width {d} x {n} layers: {rates[(d, n)]:,} examples/s")
    train_n = len(data.rows("train"))
    hours = sum(train_n / rates[(c["d_model"], c["n_layers"])] for c in GRID) / 3600
    hours += 2 * train_n / min(rates.values()) / 3600  # two extra seeds, slower architecture assumed
    print(f"\nPhase 3 sequence runs (4 configs + 2 seeds, 1 epoch each): ~{hours:.1f} GPU hours "
          f"of training, plus validation scoring")
    if price:
        usd = hours * price
        print(f"at ${price}/h: ~${usd:.0f}")
        print("WITHIN BUDGET" if usd <= STOP_ABOVE_USD else
              f"OVER ${STOP_ABOVE_USD}: stop and re-plan before launching (PREDICTION.md section 5)")


def table(summary: dict) -> str:
    def cell(r: dict, k: str) -> str:
        return f"{r[k]['mean']:.4f} ({r[k]['min']:.4f}-{r[k]['max']:.4f})"

    p = summary["params"]
    lines = [
        "# Phase 3 — sequence model vs DCN, validation day (2017-05-12)", "",
        "Mean over 3 seeds, min-max in brackets. **Validation numbers**: both models were selected on",
        "this day. The test-day comparison is `rewind/compare/final.py`, run once.", "",
        f"Sequence model: history length {summary['seq_len']}, width {summary['selected']['d_model']} x "
        f"{summary['selected']['n_layers']} layers, lr {summary['selected']['lr']:g}. "
        f"Parameters: {p['id_embedding']:,} ID-embedding (identical tables to DCN) + "
        f"{p['aux_embedding']:,} action/time/position embedding + {p['dense']:,} dense "
        f"(DCN: 1,364,049 dense).", "",
    ]
    for split, title in (("val", "All validation impressions"),
                         ("val_no_history", "No earlier impressions (P7)"),
                         ("val_history_256_plus", "256+ earlier impressions (P3)")):
        lines += [f"## {title}", "", "| Model | " + " | ".join(m.upper() if m != "calibration" else "Calibration"
                                                              for m in METRICS) + " |",
                  "|---|" + "---|" * len(METRICS)]
        for name, r in (("DCN (Phase 2)", summary["dcn_phase2_validation"][split]),
                        ("Sequence", summary["results"][split])):
            lines.append(f"| {name} | " + " | ".join(cell(r, k) for k in METRICS) + " |")
        lines.append("")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
