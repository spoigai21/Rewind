"""Phase 0: measure how training and inference cost grow with history length.

Runs on synthetic inputs. Compute cost depends on tensor shapes, not on the values inside them,
so random item IDs cost exactly what real ones do. What synthetic data can NOT tell us is model
quality; that needs the real log.

Every history is full length (no padding), so these are worst-case numbers for each length.

    uv run python -m rewind.phase0.cost --device mps
"""

import argparse
import gc
import hashlib
import json
import platform
import statistics
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
import torch.nn.functional as F

from rewind.model import SeqConfig, SequenceCTR, count_params


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--device", default="mps", choices=["mps", "cpu", "cuda"])
    ap.add_argument("--lengths", type=int, nargs="+", default=[16, 32, 64, 128, 256, 512, 1024])
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--warmup", type=int, default=5, help="untimed steps before measuring")
    ap.add_argument("--steps", type=int, default=20, help="timed steps per length")
    ap.add_argument("--n-items", type=int, default=SeqConfig.n_items)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--max-step-s", type=float, default=15.0,
        help="give up on a length if one training step takes longer than this (memory swapping)",
    )
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    device = torch.device(args.device)
    hw = hardware_info(device)
    # Read the git state before measuring: the code that runs is the code present at the start,
    # and files edited while a long sweep runs must not change what the result claims.
    git = git_info()
    out = args.out or Path("results/phase0") / f"cost_{hw['lane']}_b{args.batch_size}.json"

    print(f"lane: {hw['lane']}  ({hw['chip']}, torch {hw['torch']})")
    print(f"batch {args.batch_size}, {args.steps} timed steps after {args.warmup} warmup, seed {args.seed}\n")
    print(f"{'len':>6} {'train ms/step':>14} {'train ex/s':>11} {'infer ms/batch':>15} {'infer ex/s':>11} {'mem GB':>7}")

    # The first GPU run in a process pays one-off costs (compiling GPU programs, growing memory
    # pools) that a few warmup steps do not fully absorb. One throwaway length soaks them up so
    # they are not billed to whichever length happens to be measured first.
    measure_length(min(args.lengths), args, device)

    rows = []
    for L in args.lengths:
        row = measure_length(L, args, device)
        rows.append(row)
        if row["status"] != "ok":
            print(f"{L:>6}  {row['status']}; stopping")
            break
        t, i = row["train"], row["infer"]
        mem = row["peak_mem_gb"]
        print(
            f"{L:>6} {t['ms_median']:>14.1f} {t['examples_per_s']:>11.0f} "
            f"{i['ms_median']:>15.1f} {i['examples_per_s']:>11.0f} {mem if mem is not None else '-':>7}"
            + ("  UNSTABLE TIMING" if row["unstable_timing"] else "")
        )

    result = {
        "experiment": "phase0_cost_vs_length",
        "synthetic_inputs": True,
        "padding": "none (every history full length; worst case)",
        "hardware": hw,
        "git": git,
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "config": {
            "batch_size": args.batch_size,
            "warmup_steps": args.warmup,
            "timed_steps": args.steps,
            "seed": args.seed,
            "model": {k: v for k, v in vars(SeqConfig(n_items=args.n_items)).items() if k != "max_len"},
            "optimizer": "AdamW lr=1e-3",
        },
        "rows": rows,
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2) + "\n")
    print(f"\nwrote {out}")


def measure_length(L: int, args, device: torch.device) -> dict:
    torch.manual_seed(args.seed)
    cfg = SeqConfig(n_items=args.n_items, max_len=L)
    B = args.batch_size
    row: dict = {"seq_len": L, "params": None, "status": "ok"}

    try:
        model = SequenceCTR(cfg).to(device)
        row["params"] = count_params(model)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3)

        # One fixed synthetic batch, reused every step, so data loading is not part of the timing.
        items = torch.randint(1, cfg.n_items, (B, L), device=device)
        actions = torch.randint(1, cfg.n_actions + 1, (B, L), device=device)
        target = torch.randint(1, cfg.n_items, (B,), device=device)
        labels = torch.randint(0, 2, (B,), device=device).float()

        reset_memory(device)

        def train_step():
            loss = F.binary_cross_entropy_with_logits(model(items, actions, target), labels)
            loss.backward()
            opt.step()
            opt.zero_grad(set_to_none=True)

        model.train()
        try:
            train_ms = timed(train_step, args.warmup, args.steps, device, args.max_step_s)
        except TooSlow:
            row["status"] = "too_slow"
            return row
        peak = peak_memory_gb(device)

        model.eval()
        with torch.no_grad():
            infer_ms = timed(lambda: model(items, actions, target), args.warmup, args.steps, device)

        row["train"] = summarize(train_ms, B)
        row["infer"] = summarize(infer_ms, B)
        row["peak_mem_gb"] = peak
        # Steady GPU work gives similar step times, so a typical (median) step far slower than the
        # fastest means most steps were disturbed, usually by the OS swapping memory to disk. One
        # slow outlier does not trip this. Flagged rows are excluded from extrapolation.
        row["unstable_timing"] = row["train"]["ms_median"] > 1.5 * row["train"]["ms_min"]
    except RuntimeError as e:
        if "out of memory" not in str(e).lower():
            raise
        row["status"] = "oom"
    finally:
        # Free everything before the next length so memory readings do not carry over.
        model = opt = None
        gc.collect()
        empty_cache(device)

    return row


class TooSlow(Exception):
    pass


def timed(fn, warmup: int, steps: int, device: torch.device, max_s: float = float("inf")) -> list[float]:
    """Run fn warmup+steps times; return wall-clock ms for each timed run.

    GPUs run asynchronously: a call returns before the work finishes. sync() waits for the GPU
    so each timing covers the real work, not just the time to queue it. Any single run, warmup
    included, longer than max_s raises TooSlow.
    """
    times = []
    for i in range(warmup + steps):
        t0 = time.perf_counter()
        fn()
        sync(device)
        elapsed = time.perf_counter() - t0
        if elapsed > max_s:
            raise TooSlow
        if i >= warmup:
            times.append(elapsed * 1000)
    return times


def summarize(ms: list[float], batch_size: int) -> dict:
    med = statistics.median(ms)
    return {
        "ms_median": round(med, 3),
        "ms_min": round(min(ms), 3),
        "ms_max": round(max(ms), 3),
        "examples_per_s": round(batch_size / (med / 1000), 1),
    }


def sync(device: torch.device) -> None:
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize()


def empty_cache(device: torch.device) -> None:
    if device.type == "mps":
        torch.mps.empty_cache()
    elif device.type == "cuda":
        torch.cuda.empty_cache()


def reset_memory(device: torch.device) -> None:
    empty_cache(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()


def peak_memory_gb(device: torch.device) -> float | None:
    """CUDA tracks a true peak. MPS has no peak counter, so this reads total memory the Metal
    driver holds after training steps, which includes cached blocks: an upper bound, not an exact
    peak. CPU memory is not measured (process RSS only ever grows, so it cannot be split by length).
    """
    if device.type == "cuda":
        return round(torch.cuda.max_memory_allocated() / 1e9, 3)
    if device.type == "mps":
        return round(torch.mps.driver_allocated_memory() / 1e9, 3)
    return None


def hardware_info(device: torch.device) -> dict:
    if device.type == "cuda":
        chip = torch.cuda.get_device_name(device)
    elif platform.system() == "Darwin":
        chip = _run(["sysctl", "-n", "machdep.cpu.brand_string"]) or platform.processor()
    else:
        chip = platform.processor() or platform.machine()
    ram = _run(["sysctl", "-n", "hw.memsize"]) if platform.system() == "Darwin" else None
    lane = f"{chip.lower().replace('apple ', '').replace(' ', '-')}-{device.type}"
    return {
        "lane": lane,
        "chip": chip,
        "device": device.type,
        "system_ram_gb": round(int(ram) / 1e9, 1) if ram else None,
        "os": f"{platform.system()} {platform.release()}",
        "torch": torch.__version__,
        "python": platform.python_version(),
    }


def git_info() -> dict:
    """The commit that produced a result, and whether the code differed from it.

    "dirty" covers uncommitted edits AND new files nobody has added yet: code that exists only
    on this laptop cannot be reproduced from the commit hash. Results files are excluded, since a
    run always writes one.
    """
    status = _run(["git", "status", "--porcelain", "--", ".", ":!results"])
    return {"commit": _run(["git", "rev-parse", "--short", "HEAD"]), "dirty": bool(status),
            "source_sha256": source_fingerprint()}


def source_fingerprint() -> str:
    """One hash over every source file and the dependency lock. Identifies the exact code even
    when it was not committed, which a commit hash alone cannot (BUGLOG #11)."""
    root = Path(__file__).resolve().parents[2]
    files = sorted(root.glob("rewind/**/*.py")) + [root / "pyproject.toml", root / "uv.lock"]
    h = hashlib.sha256()
    for f in files:
        h.update(f"{f.relative_to(root)}:{hashlib.sha256(f.read_bytes()).hexdigest()}\n".encode())
    return h.hexdigest()


def _run(cmd: list[str]) -> str | None:
    try:
        return subprocess.run(cmd, capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


if __name__ == "__main__":
    main()
