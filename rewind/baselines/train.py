"""Train one tabular baseline for one epoch on the training days and score it on the validation day.

The test day is never read here.

    uv run python -m rewind.baselines.train --model dcn --lr 1e-3 --dim 16 --seed 0
"""

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from rewind.baselines.models import MODELS, param_counts
from rewind.data.sequences import SPLITS
from rewind.metrics import evaluate
from rewind.phase0.cost import git_info, hardware_info

# Slices reported alongside the full validation day: impressions with no earlier impression
# (PREDICTION.md P7) and with long histories (P3).
SLICES = {"no_history": lambda n: n == 0, "history_256_plus": lambda n: n >= 256}


class Data:
    """Training and validation rows held in memory (~3.5 GB), plus what evaluation needs."""

    def __init__(self, feature_dir: Path = Path("data/features"), seq_dir: Path = Path("data/sequences")):
        self.meta = json.loads((feature_dir / "meta.json").read_text())
        split = np.load(seq_dir / "split.npy")
        cat = np.load(feature_dir / "cat.npy", mmap_mode="r")
        num = np.load(feature_dir / "num.npy", mmap_mode="r")
        clk = np.load(seq_dir / "clk.npy")
        prior = np.load(seq_dir / "hist_end.npy") - np.load(seq_dir / "user_start.npy")
        user = np.load(seq_dir / "user.npy")

        tr = np.flatnonzero(split == SPLITS["train"])
        va = np.flatnonzero(split == SPLITS["val"])
        self.train = (np.ascontiguousarray(cat[tr]), np.ascontiguousarray(num[tr]), clk[tr].astype(np.float32))
        self.val = (np.ascontiguousarray(cat[va]), np.ascontiguousarray(num[va]), clk[va])
        self.val_user, self.val_prior, self.val_rows = user[va], prior[va], va


def train_one(data: Data, model_name: str, lr: float, dim: int, seed: int, device: torch.device,
              batch_size: int = 4096, log_every: int = 500, max_batches: int | None = None,
              checkpoint: Path | None = None) -> tuple[dict, np.ndarray]:
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    cards, n_num = data.meta["cat_cardinalities"], len(data.meta["num_features"])
    model = MODELS[model_name](cards, n_num, dim=dim).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    cat, num, y = data.train
    order = rng.permutation(len(y))
    n_batches = (len(order) + batch_size - 1) // batch_size
    if max_batches:  # smoke tests only; real runs always see the full epoch
        n_batches = min(n_batches, max_batches)
    model.train()
    t0 = time.perf_counter()
    running = []
    for b in range(n_batches):
        idx = np.sort(order[b * batch_size:(b + 1) * batch_size])
        c = torch.from_numpy(cat[idx]).to(device, dtype=torch.long)
        x = torch.from_numpy(num[idx]).to(device)
        t = torch.from_numpy(y[idx]).to(device)
        loss = F.binary_cross_entropy_with_logits(model(c, x), t)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        running.append(loss.item())
        if not np.isfinite(running[-1]):
            raise SystemExit(f"{model_name}: non-finite loss at batch {b}")
        if log_every and (b + 1) % log_every == 0:
            print(f"  batch {b + 1}/{n_batches}  loss {np.mean(running[-log_every:]):.4f}  "
                  f"{(b + 1) * batch_size / (time.perf_counter() - t0):,.0f} ex/s", flush=True)
    train_s = time.perf_counter() - t0

    p = predict(model, data.val[0], data.val[1], device)
    y_val = data.val[2]
    metrics = {"val": evaluate(y_val, p, data.val_user)}
    for name, rule in SLICES.items():
        m = rule(data.val_prior)
        metrics[f"val_{name}"] = evaluate(y_val[m], p[m], data.val_user[m])

    result = {
        "model": model_name, "lr": lr, "dim": dim, "seed": seed, "batch_size": batch_size, "epochs": 1,
        "optimizer": "Adam", "params": param_counts(model),
        "train_seconds": round(train_s, 1), "train_examples_per_s": round(len(y) / train_s),
        "final_train_loss_last_500": float(np.mean(running[-500:])),
        "metrics": metrics,
    }
    if checkpoint:
        # Phase 3: saved so the one-time test-day comparison can score this exact model.
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"state_dict": model.state_dict(), "model": model_name, "dim": dim,
                    "cardinalities": cards, "n_num": n_num}, checkpoint)
        result["checkpoint"] = str(checkpoint)
    return result, p


@torch.no_grad()
def predict(model, cat: np.ndarray, num: np.ndarray, device, batch_size: int = 65536) -> np.ndarray:
    model.eval()
    out = []
    for s in range(0, len(cat), batch_size):
        c = torch.from_numpy(cat[s:s + batch_size]).to(device, dtype=torch.long)
        x = torch.from_numpy(num[s:s + batch_size]).to(device)
        out.append(torch.sigmoid(model(c, x)).float().cpu().numpy())
    return np.concatenate(out)


def save_run(result: dict, preds: np.ndarray, device: torch.device, git: dict,
             runs_dir: Path = Path("results/phase2/runs"), preds_dir: Path = Path("data/preds/phase2")) -> Path:
    name = f"{result['model']}_lr{result['lr']:g}_d{result['dim']}_s{result['seed']}"
    result = {**result, "hardware": hardware_info(device), "git": git,
              "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
              "val_predictions": str(preds_dir / f"{name}.npy")}
    runs_dir.mkdir(parents=True, exist_ok=True)
    preds_dir.mkdir(parents=True, exist_ok=True)
    np.save(preds_dir / f"{name}.npy", preds.astype(np.float32))
    path = runs_dir / f"{name}.json"
    path.write_text(json.dumps(result, indent=2) + "\n")
    return path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", choices=list(MODELS), required=True)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--dim", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--batch-size", type=int, default=4096)
    ap.add_argument("--checkpoint-dir", type=Path, default=None, help="save trained weights here")
    ap.add_argument("--runs-dir", type=Path, default=Path("results/phase2/runs"))
    ap.add_argument("--preds-dir", type=Path, default=Path("data/preds/phase2"))
    args = ap.parse_args()
    device = torch.device(args.device)
    git = git_info()
    data = Data()
    name = f"{args.model}_lr{args.lr:g}_d{args.dim}_s{args.seed}"
    ckpt = args.checkpoint_dir / f"{name}.pt" if args.checkpoint_dir else None
    result, preds = train_one(data, args.model, args.lr, args.dim, args.seed, device, args.batch_size,
                              checkpoint=ckpt)
    print(json.dumps(result["metrics"]["val"], indent=2))
    print(f"wrote {save_run(result, preds, device, git, args.runs_dir, args.preds_dir)}")


if __name__ == "__main__":
    main()
