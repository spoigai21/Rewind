"""Train the Phase 3 sequence model for one epoch on the training days, score it on the validation
day, and save its weights. Same recipe as the Phase 2 baselines (DESIGN.md): Adam, batch 4,096,
one epoch, binary cross-entropy. The test day is never read here.

    uv run python -m rewind.sequence.train --device cuda --lr 1e-3 --d-model 64 --seed 0
    uv run python -m rewind.sequence.train --device mps --max-batches 50   # short trial
"""

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from rewind.metrics import evaluate
from rewind.phase0.cost import git_info, hardware_info, sync
from rewind.sequence.data import SequenceData
from rewind.sequence.model import SequenceModel, SequenceModelConfig, param_counts, size_head

DCN_DENSE_PARAMS = 1_364_049  # results/phase2/baselines.json, selected DCN
SLICES = {"no_history": lambda n: n == 0, "history_256_plus": lambda n: n >= 256}
RUNS_DIR = Path("results/phase3/runs")
CKPT_DIR = Path("data/checkpoints/phase3")
PREDS_DIR = Path("data/preds/phase3")


def make_config(data: SequenceData, d_model: int, n_layers: int, n_heads: int, length: int) -> SequenceModelConfig:
    cfg = SequenceModelConfig(cardinalities=data.cardinalities, n_context=len(data.context_names),
                              d_model=d_model, n_layers=n_layers, n_heads=n_heads, max_len=length)
    return size_head(cfg, DCN_DENSE_PARAMS)


def run_name(lr: float, d_model: int, n_layers: int, length: int, seed: int) -> str:
    return f"seq_L{length}_d{d_model}x{n_layers}_lr{lr:g}_s{seed}"


def train_one(data: SequenceData, lr: float, d_model: int, n_layers: int, n_heads: int, seed: int,
              length: int = 64, batch_size: int = 4096, max_batches: int | None = None,
              log_every: int = 500) -> tuple[dict, np.ndarray, SequenceModel]:
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    device = data.device
    cfg = make_config(data, d_model, n_layers, n_heads, length)
    model = SequenceModel(cfg).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    order = torch.from_numpy(rng.permutation(data.rows("train"))).to(device)
    n_batches = (len(order) + batch_size - 1) // batch_size
    if max_batches:  # trials only; real runs always see the full epoch
        n_batches = min(n_batches, max_batches)

    model.train()
    sync(device)
    t0 = time.perf_counter()
    running = []
    for i in range(n_batches):
        b = data.batch(order[i * batch_size:(i + 1) * batch_size], length)
        loss = F.binary_cross_entropy_with_logits(model(b), b["label"])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        running.append(loss.item())
        if not np.isfinite(running[-1]):
            raise SystemExit(f"non-finite loss at batch {i}")
        if log_every and (i + 1) % log_every == 0:
            print(f"  batch {i + 1}/{n_batches}  loss {np.mean(running[-log_every:]):.4f}  "
                  f"{(i + 1) * batch_size / (time.perf_counter() - t0):,.0f} ex/s", flush=True)
    sync(device)
    train_s = time.perf_counter() - t0
    examples = min(len(order), n_batches * batch_size)

    val_rows = data.rows("val")
    p = predict(model, data, val_rows, length)
    y = data.clk[torch.from_numpy(val_rows).to(device)].cpu().numpy()
    users, prior = data.user[val_rows], data.prior[val_rows]
    metrics = {"val": evaluate(y, p, users)}
    for name, rule in SLICES.items():
        m = rule(prior)
        metrics[f"val_{name}"] = evaluate(y[m], p[m], users[m])

    result = {
        "model": "sequence", "lr": lr, "d_model": d_model, "n_layers": n_layers, "n_heads": n_heads,
        "head_hidden": cfg.head_hidden, "seq_len": length, "seed": seed, "batch_size": batch_size,
        "epochs": 1 if not max_batches else f"trial: {n_batches} batches", "optimizer": "Adam",
        "params": param_counts(model), "dcn_dense_target": DCN_DENSE_PARAMS,
        "train_seconds": round(train_s, 1), "train_examples_per_s": round(examples / train_s),
        "final_train_loss_last_500": float(np.mean(running[-500:])), "metrics": metrics,
    }
    return result, p, model


@torch.no_grad()
def predict(model: SequenceModel, data: SequenceData, rows: np.ndarray, length: int,
            batch_size: int = 4096) -> np.ndarray:
    """Scoring uses the training batch size: attention memory grows with batch x length^2, and a
    16,384-row scoring batch needed several GB at length 64 (BUGLOG #13)."""
    model.eval()
    out = []
    rows_t = torch.from_numpy(rows).to(data.device)
    for s in range(0, len(rows_t), batch_size):
        b = data.batch(rows_t[s:s + batch_size], length)
        out.append(torch.sigmoid(model(b)).float().cpu().numpy())
    model.train()
    return np.concatenate(out)


def save_run(result: dict, preds: np.ndarray, model: SequenceModel, git: dict) -> Path:
    name = run_name(result["lr"], result["d_model"], result["n_layers"], result["seq_len"], result["seed"])
    for d in (RUNS_DIR, CKPT_DIR, PREDS_DIR):
        d.mkdir(parents=True, exist_ok=True)
    ckpt = CKPT_DIR / f"{name}.pt"
    torch.save({"state_dict": model.state_dict(), "config": vars(model.cfg)}, ckpt)
    np.save(PREDS_DIR / f"{name}.npy", preds.astype(np.float32))
    result = {**result, "hardware": hardware_info(model.norm.weight.device), "git": git,
              "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
              "checkpoint": str(ckpt), "val_predictions": str(PREDS_DIR / f"{name}.npy")}
    path = RUNS_DIR / f"{name}.json"
    path.write_text(json.dumps(result, indent=2) + "\n")
    return path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--d-model", type=int, default=64)
    ap.add_argument("--n-layers", type=int, default=2)
    ap.add_argument("--n-heads", type=int, default=2)
    ap.add_argument("--length", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-batches", type=int, default=None, help="short trial; result is NOT saved")
    ap.add_argument("--user-fraction", type=float, default=1.0,
                    help="laptop trials: load only this share of users (complete histories)")
    args = ap.parse_args()

    git = git_info()
    if args.user_fraction < 1 and not args.max_batches:
        raise SystemExit("--user-fraction is for trials only; combine it with --max-batches")
    data = SequenceData(torch.device(args.device), user_fraction=args.user_fraction)
    result, preds, model = train_one(data, args.lr, args.d_model, args.n_layers, args.n_heads, args.seed,
                                     args.length, max_batches=args.max_batches,
                                     log_every=10 if args.max_batches else 500)
    print(json.dumps({k: result[k] for k in ("params", "train_seconds", "train_examples_per_s")}, indent=2))
    print(json.dumps(result["metrics"]["val"], indent=2))
    if args.max_batches:
        print("trial run: not saved")
    else:
        print(f"wrote {save_run(result, preds, model, git)}")


if __name__ == "__main__":
    main()
