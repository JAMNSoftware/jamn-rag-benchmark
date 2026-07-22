#!/usr/bin/env python3
"""
Leaderboard variance across the 3 runs (RevisionPlan item 12).

Tables III/IV report means only despite triplicate runs. R2 asked for variance.
This computes per-candidate mean ±SD across runs and — more importantly — flags
which adjacent rank gaps fall inside the sampling-noise floor, so the ordering
is not over-claimed.

No API calls: reads existing judged CSVs only.

Usage (run from backend/scripts/):
    python3 compute_leaderboard_variance.py
    python3 compute_leaderboard_variance.py --judge google/gemini-3.5-flash
    python3 compute_leaderboard_variance.py --noise-floor 0.08

Output:
    Prints per-candidate mean ±SD and an adjacent-gap significance table.
    Writes leaderboard_variance.csv

Method notes:
    - A "run" score is the mean over that run's 43 items (39 for groundedness,
      blocked items excluded), so SD is the between-run spread of run means —
      the quantity that answers "would a re-run reorder the table?".
    - The reference noise floor (default 0.08) comes from the paper's own
      temperature ablation. A rank gap smaller than that is not evidence of a
      real quality difference.
    - Welch's t is reported for adjacent pairs as a secondary signal, but with
      n=3 runs it has very low power — treat the noise-floor comparison as
      primary and do not report a p-value as if it settled anything.
"""

import argparse
import csv
import os
import sys
from collections import defaultdict

import numpy as np

csv.field_size_limit(sys.maxsize)
_HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(_HERE, "results")
OUT = os.path.join(_HERE, "leaderboard_variance.csv")
INCUMBENT = "openai/gpt-5.4"


def judge_dir(judge):
    return RESULTS if judge == INCUMBENT else os.path.join(
        RESULTS, f"judge_{judge.replace('/', '_')}")


def load(judge):
    d = judge_dir(judge)
    if not os.path.isdir(d):
        return None
    out = {}
    for f in sorted(os.listdir(d)):
        if not (f.startswith("judged_") and f.endswith(".csv")):
            continue
        cand = f.replace("judged_", "").replace(".csv", "").replace("_", "/", 1)
        per_run = defaultdict(lambda: {"g": [], "c": []})
        with open(os.path.join(d, f)) as fh:
            for r in csv.DictReader(fh):
                if (r.get("justification") or "").startswith("ERROR"):
                    continue
                run = int(r["run"])
                for key, col in (("g", "groundedness_judge"),
                                 ("c", "correctness_judge")):
                    v = r[col]
                    if v not in ("", "None", "null", None):
                        per_run[run][key].append(int(v))
        if per_run:
            out[cand] = per_run
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--judge", default=INCUMBENT)
    ap.add_argument("--noise-floor", type=float, default=0.08,
                    help="sampling-noise floor from the temperature ablation")
    args = ap.parse_args()

    data = load(args.judge)
    if not data:
        sys.exit(f"no judged data for {args.judge}")

    stats = {}
    for cand, per_run in data.items():
        runs = sorted(per_run)
        combined = []
        for run in runs:
            g = per_run[run]["g"]
            c = per_run[run]["c"]
            gm = float(np.mean(g)) if g else float("nan")
            cm = float(np.mean(c)) if c else float("nan")
            combined.append((gm + cm) / 2)
        stats[cand] = {
            "runs": runs,
            "per_run": combined,
            "mean": float(np.mean(combined)),
            "sd": float(np.std(combined, ddof=1)) if len(combined) > 1 else 0.0,
        }

    order = sorted(stats.items(), key=lambda kv: -kv[1]["mean"])

    print("=" * 78)
    print(f"  LEADERBOARD WITH VARIANCE — judge: {args.judge}")
    print(f"  combined = (G + C) / 2 per run; SD is between-run spread (n=3)")
    print("=" * 78)
    print(f"  {'candidate':44s} {'mean':>7s} {'±SD':>7s}   per-run")
    rows = []
    for cand, s in order:
        pr = ", ".join(f"{v:.3f}" for v in s["per_run"])
        print(f"  {cand:44s} {s['mean']:7.3f} {s['sd']:7.3f}   [{pr}]")
        rows.append({"judge": args.judge, "candidate": cand,
                     "mean": round(s["mean"], 4), "sd": round(s["sd"], 4),
                     "n_runs": len(s["per_run"]),
                     "per_run": pr, "comparison": "", "gap": "",
                     "inside_noise_floor": ""})

    print("\n" + "=" * 78)
    print(f"  ADJACENT RANK GAPS vs the {args.noise_floor:.2f} noise floor")
    print("  A gap below the floor is NOT evidence of a real quality difference.")
    print("=" * 78)
    n_inside = 0
    for (a, sa), (b, sb) in zip(order, order[1:]):
        gap = sa["mean"] - sb["mean"]
        inside = gap < args.noise_floor
        n_inside += inside
        # Welch's t on 3+3 run means — very low power, secondary only
        va, vb = np.var(sa["per_run"], ddof=1), np.var(sb["per_run"], ddof=1)
        na, nb = len(sa["per_run"]), len(sb["per_run"])
        se = np.sqrt(va / na + vb / nb)
        t = (sa["mean"] - sb["mean"]) / se if se > 0 else float("inf")
        flag = "⚠️  INSIDE NOISE — ordering not established" if inside else "✅ exceeds noise floor"
        print(f"\n  {a.split('/')[-1]}  vs  {b.split('/')[-1]}")
        print(f"    gap = {gap:.3f}   (Welch t = {t:+.2f}, n=3+3 — low power)")
        print(f"    {flag}")
        rows.append({"judge": args.judge, "candidate": "", "mean": "", "sd": "",
                     "n_runs": "", "per_run": "",
                     "comparison": f"{a} > {b}", "gap": round(gap, 4),
                     "inside_noise_floor": str(inside)})

    print(f"\n  {n_inside} of {len(order) - 1} adjacent gaps fall inside the "
          f"{args.noise_floor:.2f} noise floor.")
    print("  Report those pairs as statistically indistinguishable rather than ranked.")

    with open(OUT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["judge", "candidate", "mean", "sd",
                                          "n_runs", "per_run", "comparison",
                                          "gap", "inside_noise_floor"])
        w.writeheader()
        w.writerows(rows)
    print(f"\nWritten to {OUT}")


if __name__ == "__main__":
    main()
