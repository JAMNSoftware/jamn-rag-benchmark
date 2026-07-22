#!/usr/bin/env python3
"""
Jury analysis for the 3-judge experiment (SmartEdu 2026 camera-ready, Task B).

Consumes the published GPT-5.4 judgments (results/judged_*.csv) plus any
per-judge sweeps written by judge_runner_jury.py
(results/judge_<slug>/judged_*.csv) and produces every reporting artifact the
handoff requires:

  1. Per-judge leaderboards (never only the aggregate — a loose judge must
     stay visible rather than being folded into a mean)
  2. Jury aggregate, with the aggregation rule stated explicitly
  3. Spearman rank correlation between each judge pair, computed with each
     judge's OWN candidate excluded from the ranking
  4. Delta-vs-controls: per candidate, Delta = GPT-5.4 score - other judge's
     score. The 6 non-GPT candidates give the baseline distribution of
     judge-to-judge disagreement; we report whether GPT-5.4's Delta on ITSELF
     is an outlier against that distribution (z-score and rank).
  5. Inter-judge Krippendorff's alpha / Fleiss' kappa, reusing the Task A
     implementations unchanged (imported from compute_interrater), reported
     alongside the human inter-rater numbers.

Usage (run from backend/scripts/):
    python3 analyze_jury.py
    python3 analyze_jury.py --judges openai/gpt-5.4 google/gemini-3.5-flash
    python3 analyze_jury.py --aggregate median

Output:
    Prints all tables to stdout.
    Writes jury_results.csv (long format: table, judge, candidate, stat, value).

Notes:
    - The incumbent GPT-5.4 judgments are read from the flat results/judged_*.csv
      (the published files); they are never written to.
    - Self-score exclusion: a judge model that is also a candidate has its own
      candidate row dropped before rank correlation.
    - Groundedness is null for preflight-blocked items by rubric design, so
      per-candidate G means are over 117 rows and C over 129.
"""

import argparse
import csv
import itertools
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from compute_interrater import krippendorff_alpha_ordinal, fleiss_kappa

csv.field_size_limit(sys.maxsize)

_HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(_HERE, "results")
OUTPUT_CSV = "jury_results.csv"

INCUMBENT_JUDGE = "openai/gpt-5.4"

# Human inter-rater numbers from compute_interrater.py, for the 3-vs-3
# symmetry comparison (the headline of this experiment).
HUMAN_REFERENCE = {
    "groundedness": {"alpha": 0.8935, "fleiss": 0.8483, "n": 39},
    "correctness": {"alpha": 0.6874, "fleiss": 0.5185, "n": 43},
}


def judge_slug(model):
    return model.replace("/", "_")


def candidate_from_filename(fname):
    """judged_meta-llama_llama-3.2-3b-instruct.csv -> meta-llama/llama-3.2-3b-instruct"""
    return (fname.replace("judged_", "").replace(".csv", "").replace("_", "/", 1))


def load_judge(judge_model):
    """Returns {candidate: {(item_id, run): {"g": int|None, "c": int|None}}}."""
    if judge_model == INCUMBENT_JUDGE:
        d = RESULTS_DIR
        files = [f for f in os.listdir(d)
                 if f.startswith("judged_") and f.endswith(".csv")]
    else:
        d = os.path.join(RESULTS_DIR, f"judge_{judge_slug(judge_model)}")
        if not os.path.isdir(d):
            return None
        files = [f for f in os.listdir(d)
                 if f.startswith("judged_") and f.endswith(".csv")]
    if not files:
        return None

    out = {}
    for fname in sorted(files):
        cand = candidate_from_filename(fname)
        rows = {}
        with open(os.path.join(d, fname)) as fh:
            for r in csv.DictReader(fh):
                key = (int(r["item_id"]), int(r["run"]))
                def num(v):
                    return int(v) if v not in ("", "None", "null", None) else None
                rows[key] = {"g": num(r["groundedness_judge"]),
                             "c": num(r["correctness_judge"])}
        out[cand] = rows
    return out


def leaderboard(judge_data):
    """{candidate: {"g": mean, "c": mean, "combined": mean, "n_g":, "n_c":}}"""
    lb = {}
    for cand, rows in judge_data.items():
        g = [v["g"] for v in rows.values() if v["g"] is not None]
        c = [v["c"] for v in rows.values() if v["c"] is not None]
        gm = float(np.mean(g)) if g else float("nan")
        cm = float(np.mean(c)) if c else float("nan")
        lb[cand] = {"g": gm, "c": cm, "combined": (gm + cm) / 2,
                    "n_g": len(g), "n_c": len(c)}
    return lb


def spearman(x, y):
    """Spearman rho via Pearson on ranks (average ranks for ties)."""
    def rank(v):
        v = np.asarray(v, dtype=float)
        order = v.argsort()
        r = np.empty(len(v), dtype=float)
        r[order] = np.arange(1, len(v) + 1)
        # average ranks within tie groups
        for val in np.unique(v):
            mask = v == val
            if mask.sum() > 1:
                r[mask] = r[mask].mean()
        return r
    rx, ry = rank(x), rank(y)
    if np.std(rx) == 0 or np.std(ry) == 0:
        return float("nan")
    return float(np.corrcoef(rx, ry)[0, 1])


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--judges", nargs="+", default=[
        INCUMBENT_JUDGE, "google/gemini-3.5-flash", "qwen/qwen3.5-397b-a17b"])
    ap.add_argument("--aggregate", choices=["mean", "median"], default="mean",
                    help="jury aggregation rule (stated explicitly in output)")
    ap.add_argument("--metric", choices=["combined", "g", "c"], default="combined",
                    help="score used for leaderboards / rank correlation")
    args = ap.parse_args()

    loaded, missing = {}, []
    for j in args.judges:
        d = load_judge(j)
        if d is None:
            missing.append(j)
        else:
            loaded[j] = d

    if missing:
        print(f"NOT YET RUN (no sweep found): {', '.join(missing)}")
    if not loaded:
        sys.exit("No judge data found at all — run judge_runner_jury.py first.")

    # --- coverage guard -------------------------------------------------
    # A sweep can die partway (rate limit, exhausted credits) leaving files
    # full of ERROR rows and null scores. Averaging those silently yields a
    # confident-looking but meaningless leaderboard, so any judge missing a
    # meaningful share of its rows is EXCLUDED here rather than reported.
    MIN_COVERAGE = 0.90
    incomplete = {}
    for j, data in list(loaded.items()):
        thin = {}
        for cand, rows in data.items():
            scored = sum(1 for v in rows.values()
                         if v["g"] is not None or v["c"] is not None)
            if not rows or scored / len(rows) < MIN_COVERAGE:
                thin[cand] = (scored, len(rows))
        if thin:
            incomplete[j] = thin
            del loaded[j]

    if incomplete:
        print("🚨 INCOMPLETE SWEEPS — EXCLUDED FROM ALL ANALYSIS BELOW")
        for j, thin in incomplete.items():
            print(f"  {j}: {len(thin)} candidate(s) below {MIN_COVERAGE:.0%} coverage")
            for cand, (s, n) in sorted(thin.items()):
                print(f"      {cand:45s} {s:>4d}/{n} scored")
        print("  Finish the sweep (repair_judged_errors.py resumes error rows),")
        print("  then re-run. Partial judge data is NOT reported as a leaderboard.\n")

    if not loaded:
        sys.exit("No judge has complete data — nothing trustworthy to report.")
    print(f"Judges loaded: {', '.join(loaded)}\n")

    rows_out = []

    def add(table, judge, candidate, stat, value):
        rows_out.append({"table": table, "judge": judge, "candidate": candidate,
                         "stat": stat,
                         "value": round(value, 4) if isinstance(value, float)
                                  and value == value else value})

    # ---------- 1. per-judge leaderboards ----------
    lbs = {j: leaderboard(d) for j, d in loaded.items()}
    all_candidates = sorted(set(itertools.chain.from_iterable(
        lb.keys() for lb in lbs.values())))

    for j, lb in lbs.items():
        print("=" * 72)
        print(f"  LEADERBOARD — judge: {j}")
        print("=" * 72)
        print(f"  {'candidate':45s} {'G':>7s} {'C':>7s} {'comb':>7s}")
        for cand, s in sorted(lb.items(), key=lambda kv: -kv[1]["combined"]):
            self_mark = "  <- SELF" if cand == j else ""
            print(f"  {cand:45s} {s['g']:7.3f} {s['c']:7.3f} "
                  f"{s['combined']:7.3f}{self_mark}")
            for k in ("g", "c", "combined"):
                add("leaderboard", j, cand, k, s[k])
        print()

    # ---------- 2. jury aggregate ----------
    if len(lbs) > 1:
        print("=" * 72)
        print(f"  JURY AGGREGATE — rule: {args.aggregate} of per-judge "
              f"{args.metric} across {len(lbs)} judges")
        print("  (report alongside the per-judge tables above, never instead of them)")
        print("=" * 72)
        agg = {}
        for cand in all_candidates:
            vals = [lb[cand][args.metric] for lb in lbs.values() if cand in lb]
            agg[cand] = float(np.mean(vals) if args.aggregate == "mean"
                              else np.median(vals))
        for cand, v in sorted(agg.items(), key=lambda kv: -kv[1]):
            print(f"  {cand:45s} {v:7.3f}")
            add("jury_aggregate", args.aggregate, cand, args.metric, v)
        print()

    # ---------- 3. Spearman, self-scores excluded ----------
    if len(lbs) > 1:
        print("=" * 72)
        print("  SPEARMAN RANK CORRELATION between judges")
        print("  (each pair ranked over candidates with BOTH judges' own")
        print("   candidate rows excluded — self-scores never enter the rank)")
        print("=" * 72)
        for ja, jb in itertools.combinations(lbs, 2):
            shared = [c for c in all_candidates
                      if c in lbs[ja] and c in lbs[jb] and c != ja and c != jb]
            x = [lbs[ja][c][args.metric] for c in shared]
            y = [lbs[jb][c][args.metric] for c in shared]
            rho = spearman(x, y)
            print(f"  {ja} vs {jb}: rho={rho:.4f}  (n={len(shared)} candidates)")
            order_a = [c for c in sorted(shared, key=lambda c: -lbs[ja][c][args.metric])]
            order_b = [c for c in sorted(shared, key=lambda c: -lbs[jb][c][args.metric])]
            print(f"      {ja} order: {' > '.join(c.split('/')[-1] for c in order_a)}")
            print(f"      {jb} order: {' > '.join(c.split('/')[-1] for c in order_b)}")
            print(f"      ordering identical: {order_a == order_b}")
            add("spearman", f"{ja}|{jb}", "-", "rho", rho)
            add("spearman", f"{ja}|{jb}", "-", "n_candidates", len(shared))
            add("spearman", f"{ja}|{jb}", "-", "ordering_identical",
                str(order_a == order_b))
        print()

    # ---------- 4. delta-vs-controls ----------
    if INCUMBENT_JUDGE in lbs and len(lbs) > 1:
        print("=" * 72)
        print("  DELTA-VS-CONTROLS  (Delta = GPT-5.4 score - other judge's score)")
        print("  Controls = the 6 candidates that are NOT the incumbent judge.")
        print("  Question: is GPT-5.4's Delta ON ITSELF an outlier vs that spread?")
        print("=" * 72)
        base = lbs[INCUMBENT_JUDGE]
        for jb in lbs:
            if jb == INCUMBENT_JUDGE:
                continue
            other = lbs[jb]
            deltas = {c: base[c][args.metric] - other[c][args.metric]
                      for c in all_candidates if c in base and c in other}
            if INCUMBENT_JUDGE not in deltas:
                continue
            controls = {c: d for c, d in deltas.items() if c != INCUMBENT_JUDGE}
            cvals = np.array(list(controls.values()))
            self_d = deltas[INCUMBENT_JUDGE]
            mu, sd = float(cvals.mean()), float(cvals.std(ddof=1))
            z = (self_d - mu) / sd if sd > 0 else float("nan")
            n_above = int((cvals >= self_d).sum())
            print(f"\n  vs {jb}:")
            for c, d in sorted(deltas.items(), key=lambda kv: -kv[1]):
                tag = "  <- SELF" if c == INCUMBENT_JUDGE else ""
                print(f"    {c:45s} Delta={d:+7.3f}{tag}")
                add("delta_vs_controls", jb, c, "delta", d)
            print(f"    control mean={mu:+.3f}  SD={sd:.3f}  "
                  f"(n={len(cvals)} controls)")
            print(f"    GPT-5.4 self-Delta={self_d:+.3f}  z={z:+.2f}  "
                  f"controls >= self-Delta: {n_above}/{len(cvals)}")
            verdict = ("OUTLIER vs controls" if abs(z) > 2
                       else "within the control spread — no detectable "
                            "self-preference at this scale")
            print(f"    verdict: {verdict}")
            add("delta_vs_controls", jb, "_control_mean", "mean", mu)
            add("delta_vs_controls", jb, "_control_sd", "sd", sd)
            add("delta_vs_controls", jb, INCUMBENT_JUDGE, "self_delta_z", z)
            add("delta_vs_controls", jb, INCUMBENT_JUDGE, "verdict", verdict)
        print()

    # ---------- 5. inter-judge agreement (Task A code, unchanged) ----------
    if len(lbs) >= 2:
        print("=" * 72)
        print("  INTER-JUDGE AGREEMENT  (same estimators as the human raters)")
        print("=" * 72)
        judges = list(loaded)
        for metric, key in (("groundedness", "g"), ("correctness", "c")):
            # one 'item' = (candidate, item_id, run); keep only units where
            # every judge produced a score
            units = []
            common_cands = set.intersection(*(set(d) for d in loaded.values()))
            for cand in sorted(common_cands):
                keys = set.intersection(
                    *(set(loaded[j][cand]) for j in judges))
                for k in sorted(keys):
                    vals = [loaded[j][cand][k][key] for j in judges]
                    if all(v is not None for v in vals):
                        units.append(vals)
            if len(units) < 2:
                print(f"  {metric}: too few complete units ({len(units)})")
                continue
            alpha = krippendorff_alpha_ordinal(units)
            fk = fleiss_kappa(units)
            hr = HUMAN_REFERENCE[metric]
            print(f"\n  {metric.upper()} (n={len(units)} judged units, "
                  f"{len(judges)} judges)")
            print(f"    Krippendorff's alpha (ordinal): {alpha:.4f}   "
                  f"[3 humans: {hr['alpha']:.4f}, n={hr['n']}]")
            print(f"    Fleiss' kappa                 : {fk:.4f}   "
                  f"[3 humans: {hr['fleiss']:.4f}]")
            add("inter_judge", "|".join(judges), metric,
                "krippendorff_alpha_ordinal", alpha)
            add("inter_judge", "|".join(judges), metric, "fleiss_kappa", fk)
            add("inter_judge", "|".join(judges), metric, "n_units", len(units))
            add("inter_judge", "humans", metric,
                "krippendorff_alpha_ordinal", hr["alpha"])
            add("inter_judge", "humans", metric, "fleiss_kappa", hr["fleiss"])
        print()

    # ---------- 6. calibration sanity checks ----------
    print("=" * 72)
    print("  CALIBRATION ENTRANCE EXAMS")
    print("=" * 72)
    for j in loaded:
        path = os.path.join(RESULTS_DIR, f"judge_{judge_slug(j)}", "calibration.json")
        if not os.path.exists(path):
            print(f"  {j}: no calibration.json "
                  f"({'incumbent — pre-dates this experiment' if j == INCUMBENT_JUDGE else 'RUN IT'})")
            continue
        with open(path) as f:
            cal = json.load(f)
        print(f"  {j}: all_pass={cal['all_pass']}  provider_pin={cal.get('provider_pin')}")
        for it in cal["items"]:
            if "error" in it:
                print(f"    {it['name']}: ERROR {it['error']}")
            else:
                print(f"    {it['name']}: expected {it['expected']} got {it['got']} "
                      f"{'PASS' if it['pass'] else 'FAIL'}")
        add("calibration", j, "-", "all_pass", str(cal["all_pass"]))
    print()

    out_path = os.path.join(_HERE, OUTPUT_CSV)
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["table", "judge", "candidate", "stat", "value"])
        w.writeheader()
        w.writerows(rows_out)
    print(f"Written to {out_path}\n")

    # ---------- headline noise-floor caveat ----------
    print("REMINDER for the write-up: GPT-5.4's margin over rank 2 is 0.067,")
    print("below the ~0.08 sampling-noise floor from the temperature ablation.")
    print("Frame this as a ranking-robustness + panel-agreement study, NOT as")
    print("bias quantification. Self-preference can be neither confirmed nor")
    print("excluded at this scale, whichever way the numbers fall.")


if __name__ == "__main__":
    main()
