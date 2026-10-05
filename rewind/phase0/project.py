"""Phase 0: turn the laptop cost measurements into a predicted GPU bill for the full sweep.

Three steps, each with its assumptions written down so the prediction can be checked later:

1. Laptop cost per example at each sweep length, taken straight from measurements as a range:
   fastest step (cleanest reading; background apps only ever add time) to median step (includes
   typical interference). Each length uses the largest batch size that fit in memory.
2. Scale laptop time to rented-GPU time with an assumed speedup RANGE, not a single number.
   This is the weakest link and is labelled as an assumption; the first rented hour must
   re-measure it before the full sweep is launched.
3. Multiply by the sweep plan (examples x seeds x lengths x overhead) and an hourly price.

A quadratic t(L) = a + b*L + c*L^2 is also fitted, but only to DESCRIBE the curve (where attention,
the c*L^2 term, starts to dominate). It is not used for the bill: checked against a measurement it
was not fitted on, it under-predicted the cost at L=2048 by 25-50%.

    uv run python -m rewind.phase0.project results/phase0/cost_m5-mps_b32.json results/phase0/cost_m5-mps_b8.json
"""

import argparse
import json
from pathlib import Path

import numpy as np

# The Phase 4 sweep. Dataset sizes are the PUBLISHED figures for the Taobao display-ad log and
# have not been counted yet; the projection must be rerun once Phase 1 counts them.
SWEEP = {
    "train_examples": 19_500_000,  # ~26M published impressions x 6 of 8 days used for training
    "epochs": 1,  # click models usually overfit after one pass over the data
    "seeds": 3,  # differences between lengths may be small enough to need repeats
    "lengths": [16, 32, 64, 128, 256, 512, 1024, 2048],
    "overhead": 1.5,  # validation passes, a failed run, restarts after preemption
}

# Laptop -> A100 speedup for this small model, as a range. Rationale: memory bandwidth is
# ~2.0 TB/s on an A100 80GB vs ~0.15 TB/s on the M5, about 13x, and a 64-wide model is limited
# by moving data more than by arithmetic. The low end allows for poor utilisation at d_model=64.
SPEEDUP = {"low": 5.0, "mid": 10.0, "high": 15.0}

# Hourly A100 80GB price range (USD). Prices move; confirm on the day of rental.
PRICE_PER_HOUR = {"low": 1.0, "mid": 1.75, "high": 2.5}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cost_files", type=Path, nargs="+", help="results from rewind.phase0.cost")
    ap.add_argument("--out", type=Path, default=Path("results/phase0/projection.json"))
    args = ap.parse_args()

    rows = best_rows(args.cost_files)
    missing = [n for n in SWEEP["lengths"] if n not in rows]
    if missing:
        raise SystemExit(f"no clean measurement for lengths {missing}; measure them, do not extrapolate")

    print(f"{'len':>6} {'batch':>6} {'ms/example (fastest - median)':>32}")
    for n in SWEEP["lengths"]:
        r = rows[n]
        print(f"{n:>6} {r['batch']:>6} {r['min']:>15.3f} - {r['median']:<10.3f}")

    # Descriptive only (see module docstring).
    pts = sorted((n, r["min"]) for n, r in rows.items())
    a, b, c = fit_quadratic(pts)
    print(f"\nDescriptive fit on fastest steps: ms/example = {a:.3g} + {b:.3g}*L + {c:.3g}*L^2")
    if b > 0 and c > 0:
        print(f"attention (c*L^2) overtakes per-token work (b*L) around L = {b / c:.0f}")

    examples = SWEEP["train_examples"] * SWEEP["epochs"] * SWEEP["seeds"] * SWEEP["overhead"]

    def hours(ms_per_example: float) -> float:
        return ms_per_example * examples / 1000 / 3600

    print(f"\nSweep: {SWEEP['train_examples']:,} examples x {SWEEP['epochs']} epoch x "
          f"{SWEEP['seeds']} seeds per length, x{SWEEP['overhead']} overhead")
    print(f"{'len':>6} {'laptop hours':>16} {'A100 hours':>16}")
    per_length = []
    for n in SWEEP["lengths"]:
        lo, hi = hours(rows[n]["min"]), hours(rows[n]["median"])
        a100 = (lo / SPEEDUP["high"], hi / SPEEDUP["low"])
        per_length.append({"seq_len": n, "measured_at_batch": rows[n]["batch"],
                           "source": rows[n]["source"],
                           "laptop_hours": [round(lo, 1), round(hi, 1)],
                           "a100_hours": [round(a100[0], 1), round(a100[1], 1)]})
        print(f"{n:>6} {lo:>7.1f} - {hi:<7.1f} {a100[0]:>7.1f} - {a100[1]:<7.1f}")

    lo_total = sum(hours(rows[n]["min"]) for n in SWEEP["lengths"])
    hi_total = sum(hours(rows[n]["median"]) for n in SWEEP["lengths"])
    # Cheapest case: fastest laptop reading, biggest speedup, lowest price. Dearest: the opposite.
    a100_hours = {
        "best": lo_total / SPEEDUP["high"],
        "mid": hi_total / SPEEDUP["mid"],
        "worst": hi_total / SPEEDUP["low"],
    }
    usd = {
        "best": a100_hours["best"] * PRICE_PER_HOUR["low"],
        "mid": a100_hours["mid"] * PRICE_PER_HOUR["mid"],
        "worst": a100_hours["worst"] * PRICE_PER_HOUR["high"],
    }
    print(f"\nTotal A100 hours: {a100_hours['best']:.0f} - {a100_hours['worst']:.0f} (mid {a100_hours['mid']:.0f})")
    print(f"Total cost: ${usd['best']:.0f} - ${usd['worst']:.0f} (mid ${usd['mid']:.0f})")

    result = {
        "experiment": "phase0_sweep_projection",
        "inputs": [str(p) for p in args.cost_files],
        "measured_ms_per_example": {str(n): rows[n] for n in sorted(rows)},
        "descriptive_fit_on_fastest": {"a": a, "b": b, "c": c},
        "sweep_plan": SWEEP,
        "assumed_speedup_vs_laptop": SPEEDUP,
        "assumed_price_per_hour_usd": PRICE_PER_HOUR,
        "per_length": per_length,
        "total_a100_hours": {k: round(v, 1) for k, v in a100_hours.items()},
        "total_usd": {k: round(v) for k, v in usd.items()},
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(f"\nwrote {args.out}")


def best_rows(paths: list[Path]) -> dict[int, dict]:
    """Per length, the clean row from the largest batch size (bigger batches use the GPU more
    efficiently, so they are closer to how the rented GPU will run)."""
    best: dict[int, dict] = {}
    for path in paths:
        data = json.loads(path.read_text())
        batch = data["config"]["batch_size"]
        for r in data["rows"]:
            if r["status"] != "ok":
                continue
            n = r["seq_len"]
            if n not in best or batch > best[n]["batch"]:
                best[n] = {
                    "batch": batch,
                    "min": r["train"]["ms_min"] / batch,
                    "median": r["train"]["ms_median"] / batch,
                    "unstable_timing": r["unstable_timing"],
                    "source": path.name,
                }
    return best


def fit_quadratic(points: list[tuple[int, float]]) -> tuple[float, float, float]:
    L = np.array([p[0] for p in points], dtype=float)
    t = np.array([p[1] for p in points])
    # Least squares on RELATIVE error: short lengths are ~50x cheaper than long ones, and a plain
    # fit would ignore them entirely.
    X = np.stack([np.ones_like(L), L, L**2], axis=1)
    coef, *_ = np.linalg.lstsq(X / t[:, None], np.ones_like(t), rcond=None)
    return tuple(float(x) for x in coef)


if __name__ == "__main__":
    main()
