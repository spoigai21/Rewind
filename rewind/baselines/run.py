"""Phase 2 end to end: tune each baseline on the validation day, repeat the best over seeds, write
the baseline table. Everything is chosen on validation; the test day is never read.

Protocol, fixed before any run:
  1. Each model family tries the configurations in GRID with seed 0, one epoch each.
  2. The configuration with the lowest validation log loss is selected (log loss, not AUC: the
     project cares about probability quality, and NE is log loss rescaled).
  3. The selected configuration is retrained with seeds 1 and 2; results are reported as the mean
     and the min-max range over the 3 seeds.
  The same tuning budget (up to 4 configurations, then 3 seeds) applies to the Phase 3 sequence
  model, so neither family is tuned harder than the other.

Runs already saved in results/phase2/runs/ are reused, so an interrupted run picks up where it stopped.

    uv run python -m rewind.baselines.run            # builds features first if missing
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

from rewind.baselines.train import Data, save_run, train_one
from rewind.phase0.cost import git_info

GRID = {
    "lr": [{"lr": 1e-3, "dim": 1}, {"lr": 3e-3, "dim": 1}, {"lr": 1e-2, "dim": 1}],
    "deepfm": [{"lr": lr, "dim": d} for lr in (1e-3, 3e-3) for d in (8, 16)],
    "dcn": [{"lr": lr, "dim": d} for lr in (1e-3, 3e-3) for d in (8, 16)],
}
SEEDS = [0, 1, 2]
METRICS = ["auc", "gauc", "ne", "ece", "calibration"]
RUNS_DIR = Path("results/phase2/runs")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--models", nargs="+", default=list(GRID))
    args = ap.parse_args()

    ensure_inputs()
    device = torch.device(args.device)
    git = git_info()
    data = Data()

    summary = {"protocol": __doc__.split("Protocol, fixed before any run:")[1].split("Runs already")[0].strip(),
               "grid": GRID, "seeds": SEEDS, "models": {}}
    for model in args.models:
        print(f"\n=== {model}: tuning {len(GRID[model])} configurations (seed 0) ===")
        tuning = [run(data, model, cfg, 0, device, git) for cfg in GRID[model]]
        best = min(tuning, key=lambda r: r["metrics"]["val"]["log_loss"])
        cfg = {"lr": best["lr"], "dim": best["dim"]}
        print(f"=== {model}: selected {cfg}; repeating with seeds {SEEDS[1:]} ===")
        seeds = [best] + [run(data, model, cfg, s, device, git) for s in SEEDS[1:]]
        summary["models"][model] = {
            "selected": cfg,
            "tuning": [{"lr": r["lr"], "dim": r["dim"], "val_log_loss": r["metrics"]["val"]["log_loss"],
                        "val_auc": r["metrics"]["val"]["auc"]} for r in tuning],
            "params": best["params"],
            "train_seconds_mean": float(np.mean([r["train_seconds"] for r in seeds])),
            "hardware": best["hardware"],
            "results": {split: aggregate([r["metrics"][split] for r in seeds])
                        for split in best["metrics"]},
            "runs": [run_name(model, cfg, s) for s in SEEDS],
        }

    out = Path("results/phase2/baselines.json")
    out.write_text(json.dumps({**summary, "git": git}, indent=2) + "\n")
    Path("results/phase2/BASELINES.md").write_text(table(summary))
    print(f"\nwrote {out} and results/phase2/BASELINES.md\n")
    print(table(summary))


def ensure_inputs() -> None:
    """Build the Phase 1 sequences and the features if they are not on disk yet."""
    steps = [(Path("data/parquet/raw_sample.parquet"), "rewind.data.count"),
             (Path("data/sequences/hist_end.npy"), "rewind.data.sequences"),
             (Path("data/features/num.npy"), "rewind.features.tabular")]
    for path, module in steps:
        if not path.exists():
            print(f"{path} missing: running {module}")
            subprocess.run([sys.executable, "-m", module], check=True)


def run_name(model: str, cfg: dict, seed: int) -> str:
    return f"{model}_lr{cfg['lr']:g}_d{cfg['dim']}_s{seed}"


def run(data: Data, model: str, cfg: dict, seed: int, device, git: dict) -> dict:
    path = RUNS_DIR / f"{run_name(model, cfg, seed)}.json"
    if path.exists():
        print(f"  reuse {path.name}")
        return json.loads(path.read_text())
    print(f"  train {path.name}", flush=True)
    result, preds = train_one(data, model, cfg["lr"], cfg["dim"], seed, device, log_every=1000)
    save_run(result, preds, device, git)
    v = result["metrics"]["val"]
    print(f"    val log loss {v['log_loss']:.5f}  AUC {v['auc']:.4f}  NE {v['ne']:.4f}  "
          f"({result['train_seconds']:.0f}s)", flush=True)
    return json.loads(path.read_text())


def aggregate(results: list[dict]) -> dict:
    out = {}
    for k in METRICS:
        vals = [r[k] for r in results]
        out[k] = {"mean": float(np.mean(vals)), "min": float(np.min(vals)), "max": float(np.max(vals))}
    out["n"] = results[0]["n"]
    out["click_rate"] = results[0]["click_rate"]
    return out


def table(summary: dict) -> str:
    def cell(m: dict, k: str) -> str:
        r = m[k]
        return f"{r['mean']:.4f} ({r['min']:.4f}-{r['max']:.4f})"

    lines = [
        "# Phase 2 baselines — validation day (2017-05-12)",
        "",
        "Code: the runs were made from uncommitted source; `results/phase2/source_fingerprint.json`",
        "records the exact files.",
        "",
        "Mean over 3 seeds, min-max in brackets. Selected on validation log loss; **these are",
        "validation numbers, not test numbers** (the test day stays locked until the final",
        "comparison). Generated by `uv run python -m rewind.baselines.run`.",
        "",
    ]
    for split, title in (("val", "All validation impressions"),
                         ("val_no_history", "No earlier impressions (P7)"),
                         ("val_history_256_plus", "256+ earlier impressions (P3)")):
        first = next(iter(summary["models"].values()))["results"][split]
        lines += [f"## {title}", "",
                  f"{first['n']:,} impressions, click rate {first['click_rate']:.2%}.", "",
                  "| Model | AUC | GAUC | NE | ECE | Calibration |", "|---|---|---|---|---|---|"]
        for name, m in summary["models"].items():
            r = m["results"][split]
            lines.append(f"| {name} | " + " | ".join(cell(r, k) for k in METRICS) + " |")
        lines.append("")
    lines += ["## Models", "",
              "Train times are wall-clock under heavy, varying memory pressure on the laptop (the OS",
              "was swapping), so they are not cost measurements; Phase 0 and Phase 6 measure cost.", "",
              "| Model | Selected config | Params (embedding / dense) | Train time per run |",
              "|---|---|---|---|"]
    for name, m in summary["models"].items():
        p = m["params"]
        lines.append(f"| {name} | lr {m['selected']['lr']:g}, dim {m['selected']['dim']} | "
                     f"{p['embedding']:,} / {p['dense']:,} | {m['train_seconds_mean'] / 60:.1f} min "
                     f"({m['hardware']['lane']}) |")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
