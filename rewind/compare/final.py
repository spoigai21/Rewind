"""Phase 3: the one-time comparison of the sequence model and DCN on the TEST day (2017-05-13).

Both families are scored from their saved checkpoints: 3 seeds each, same rows, same metrics,
same slices. For each metric the verdict is decided by the seed ranges, fixed before any run:

    win   the sequence model's WORST seed beats DCN's BEST seed
    loss  DCN's worst seed beats the sequence model's best seed
    tie   the seed ranges overlap

("Beats" means higher AUC / GAUC, lower NE / ECE, calibration ratio closer to 1.)

Rehearsal on the validation day (no lock, safe to repeat):
    uv run python -m rewind.compare.final --split val

The real thing, once, after both models are final on validation:
    uv run python -m rewind.compare.final --split test --open-test-day

Opening the test day writes results/phase3/test_day_opened.json. A second test-day run is refused
unless given --rerun-reason, and every reason is appended to that file, so a re-run is never silent.
"""

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from rewind.baselines.models import MODELS
from rewind.baselines.run import METRICS, aggregate
from rewind.baselines.train import predict as predict_tabular
from rewind.metrics import evaluate
from rewind.phase0.cost import git_info, hardware_info
from rewind.sequence.data import SequenceData
from rewind.sequence.model import SequenceModel, SequenceModelConfig
from rewind.sequence.train import predict as predict_sequence

LOCK = Path("results/phase3/test_day_opened.json")
SLICES = {"all": lambda n: np.ones_like(n, dtype=bool), "no_history": lambda n: n == 0,
          "history_256_plus": lambda n: n >= 256}
HIGHER_IS_BETTER = {"auc": True, "gauc": True, "ne": False, "ece": False}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", choices=["val", "test"], default="val")
    ap.add_argument("--open-test-day", action="store_true")
    ap.add_argument("--rerun-reason", default=None)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seq-summary", type=Path, default=Path("results/phase3/sequence.json"))
    ap.add_argument("--dcn-dir", type=Path, default=Path("data/checkpoints/phase3_dcn"))
    args = ap.parse_args()

    if args.split == "test":
        guard_test_day(args.open_test_day, args.rerun_reason)

    device = torch.device(args.device)
    git = git_info()
    data = SequenceData(device)
    rows = data.rows(args.split)
    y = data.clk[torch.from_numpy(rows).to(device)].cpu().numpy()
    users, prior = data.user[rows], data.prior[rows]

    summary = json.loads(args.seq_summary.read_text())
    seq_ckpts = [Path("data/checkpoints/phase3") / f"{name}.pt" for name in summary["runs"]]
    dcn_ckpts = sorted(args.dcn_dir.glob("dcn_*.pt"))
    if len(seq_ckpts) != 3 or len(dcn_ckpts) != 3:
        raise SystemExit(f"need 3 checkpoints per family; found {len(seq_ckpts)} sequence, {len(dcn_ckpts)} DCN")

    preds = {"sequence": [score_sequence(p, data, rows, summary["seq_len"]) for p in seq_ckpts],
             "dcn": [score_dcn(p, rows, device) for p in dcn_ckpts]}

    results, verdicts = {}, {}
    for sl, rule in SLICES.items():
        m = rule(prior)
        results[sl] = {fam: aggregate([evaluate(y[m], p[m], users[m]) for p in ps]) for fam, ps in preds.items()}
        verdicts[sl] = {k: verdict(results[sl]["sequence"][k], results[sl]["dcn"][k], k) for k in METRICS}

    out = {
        "split": args.split, "rows": int(len(rows)), "click_rate": float(y.mean()),
        "checkpoints": {"sequence": [str(p) for p in seq_ckpts], "dcn": [str(p) for p in dcn_ckpts]},
        "results": results, "verdicts": verdicts, "rule": __doc__.split("fixed before any run:")[1].split("Rehearsal")[0].strip(),
        "hardware": hardware_info(device), "git": git,
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    stem = Path(f"results/phase3/final_{args.split}")
    stem.with_suffix(".json").write_text(json.dumps(out, indent=2) + "\n")
    stem.with_suffix(".md").write_text(report(out))
    print(report(out))


def guard_test_day(opened: bool, reason: str | None) -> None:
    if not opened:
        raise SystemExit("the test day is locked: pass --open-test-day to score it (once)")
    now = datetime.now(timezone.utc).isoformat(timespec="seconds")
    if LOCK.exists():
        if not reason:
            raise SystemExit(f"the test day was already opened ({LOCK}); re-running needs --rerun-reason")
        log = json.loads(LOCK.read_text())
        log["reruns"].append({"at_utc": now, "reason": reason, "git": git_info()})
    else:
        log = {"opened_at_utc": now, "git": git_info(), "reruns": []}
    LOCK.parent.mkdir(parents=True, exist_ok=True)
    LOCK.write_text(json.dumps(log, indent=2) + "\n")


def score_sequence(path: Path, data: SequenceData, rows: np.ndarray, length: int) -> np.ndarray:
    ckpt = torch.load(path, map_location=data.device)
    model = SequenceModel(SequenceModelConfig(**ckpt["config"])).to(data.device)
    model.load_state_dict(ckpt["state_dict"])
    return predict_sequence(model, data, rows, length)


def score_dcn(path: Path, rows: np.ndarray, device: torch.device, feature_dir: Path = Path("data/features")) -> np.ndarray:
    ckpt = torch.load(path, map_location=device)
    model = MODELS[ckpt["model"]](ckpt["cardinalities"], ckpt["n_num"], dim=ckpt["dim"]).to(device)
    model.load_state_dict(ckpt["state_dict"])
    cat = np.ascontiguousarray(np.load(feature_dir / "cat.npy", mmap_mode="r")[rows])
    num = np.ascontiguousarray(np.load(feature_dir / "num.npy", mmap_mode="r")[rows])
    return predict_tabular(model, cat, num, device)


def verdict(seq: dict, dcn: dict, metric: str) -> str:
    """'win' / 'loss' / 'tie' for the sequence model, from 3-seed min-max ranges."""
    if metric == "calibration":  # closer to 1 is better; compare distances from 1
        s = sorted(abs(seq[k] - 1) for k in ("min", "max"))
        d = sorted(abs(dcn[k] - 1) for k in ("min", "max"))
        # A range straddling 1 has a best distance of 0.
        if seq["min"] <= 1 <= seq["max"]:
            s[0] = 0.0
        if dcn["min"] <= 1 <= dcn["max"]:
            d[0] = 0.0
        if s[1] < d[0]:
            return "win"
        if d[1] < s[0]:
            return "loss"
        return "tie"
    if HIGHER_IS_BETTER[metric]:
        if seq["min"] > dcn["max"]:
            return "win"
        if dcn["min"] > seq["max"]:
            return "loss"
        return "tie"
    if seq["max"] < dcn["min"]:
        return "win"
    if dcn["max"] < seq["min"]:
        return "loss"
    return "tie"


def report(out: dict) -> str:
    day = {"val": "validation day (2017-05-12) — REHEARSAL", "test": "TEST day (2017-05-13)"}[out["split"]]
    lines = [f"# Phase 3 final comparison — {day}", "",
             f"{out['rows']:,} impressions, click rate {out['click_rate']:.2%}. 3 seeds per model; "
             "mean with min-max in brackets.", "", "Verdict rule: " + out["rule"].replace("\n", " "), ""]
    for sl, title in (("all", "All impressions"), ("no_history", "No earlier impressions"),
                      ("history_256_plus", "256+ earlier impressions")):
        lines += [f"## {title}", "", "| Metric | Sequence | DCN | Verdict for sequence |", "|---|---|---|---|"]
        for k in METRICS:
            s, d = out["results"][sl]["sequence"][k], out["results"][sl]["dcn"][k]
            fmt = lambda r: f"{r['mean']:.4f} ({r['min']:.4f}-{r['max']:.4f})"
            lines.append(f"| {k} | {fmt(s)} | {fmt(d)} | **{out['verdicts'][sl][k]}** |")
        lines.append("")
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    main()
