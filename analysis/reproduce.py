#!/usr/bin/env python3
"""
Reproduce the paper's headline numbers from this repository's data alone.

Self-contained: standard library + numpy only, no network, no external paths.
Run from the repository root:

    python3 analysis/reproduce.py

It recomputes, and checks against the published values:
  1. Inter-rater agreement among the three human raters (Krippendorff's ordinal
     alpha), Q4_K_M production model.
  2. Corrected judge-human agreement (Cohen's linear-weighted kappa) for the
     Q4_K_M and Instruct configurations.
  3. Retrieval hit-rate (Hit@1, Hit@3, Hit@6, MRR) over the answerable items.
  4. Multi-model leaderboard (combined G/C per model).

The other scripts in analysis/ (compute_interrater.py, analyze_jury.py, ...)
are the exact scripts used in the study, provided for transparency; they read
from the source project layout. This script is the portable reproduction path.
"""

import csv
import json
import os
import statistics
import sys

import numpy as np

csv.field_size_limit(sys.maxsize)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BLOCKED = {5, 34, 36, 39}


def krippendorff_alpha_ordinal(units, categories=(0, 1, 2)):
    """Ordinal Krippendorff's alpha via the coincidence matrix."""
    idx = {c: i for i, c in enumerate(categories)}
    k = len(categories)
    o = np.zeros((k, k))
    for u in units:
        m = len(u)
        if m < 2:
            continue
        for a in range(m):
            for b in range(m):
                if a != b:
                    o[idx[u[a]], idx[u[b]]] += 1.0 / (m - 1)
    n_c = o.sum(axis=1)
    n = n_c.sum()
    if n <= 1:
        return float("nan")
    d2 = np.zeros((k, k))
    for c in range(k):
        for d in range(c + 1, k):
            d2[c, d] = d2[d, c] = (n_c[c:d + 1].sum() - (n_c[c] + n_c[d]) / 2.0) ** 2
    do = np.sum(o * d2) / n
    de = np.sum(np.outer(n_c, n_c) * d2) / (n * (n - 1))
    return 1.0 - do / de if de else 1.0


def linear_weighted_kappa(a, b, categories=(0, 1, 2)):
    n = len(a)
    if n == 0:
        return float("nan")
    md = max(categories) - min(categories)
    w = lambda i, j: 1.0 - abs(i - j) / md
    c1 = {c: a.count(c) / n for c in categories}
    c2 = {c: b.count(c) / n for c in categories}
    po = sum(w(x, y) for x, y in zip(a, b)) / n
    pe = sum(c1[i] * c2[j] * w(i, j) for i in categories for j in categories)
    return (po - pe) / (1 - pe) if pe != 1 else 1.0


def median_int(vals):
    c = [int(v) for v in vals if v not in (None, "")]
    return round(statistics.median(c)) if c else None


def load_human(config, run=None):
    """-> {item_id: {'g':[r1,r2,r3], 'c':[...]}} for a config (optionally one run)."""
    out = {}
    with open(os.path.join(ROOT, "data", "human_ratings.csv")) as f:
        for r in csv.DictReader(f):
            if r["config"] != config:
                continue
            if run is not None and r["run"] != str(run):
                continue
            g = [r["ground_r1"], r["ground_r2"], r["ground_r3"]]
            c = [r["correct_r1"], r["correct_r2"], r["correct_r3"]]
            g = [int(x) for x in g if x not in ("", None)]
            c = [int(x) for x in c if x not in ("", None)]
            out[int(r["item_id"])] = {"g": g, "c": c}
    return out


def check(label, got, expected, tol=0.02):
    ok = abs(got - expected) <= tol
    print(f"  {'PASS' if ok else 'FAIL'}  {label:52s} got {got:.3f}  (paper {expected:.3f})")
    return ok


def section_1_interrater():
    print("\n[1] Inter-rater agreement among R1/R2/R3 (Q4_K_M, run 1)")
    h = load_human("Q4_K_M", run=1)
    g = [h[i]["g"] for i in sorted(h) if i not in BLOCKED and len(h[i]["g"]) == 3]
    c = [h[i]["c"] for i in sorted(h) if len(h[i]["c"]) == 3]
    return all([
        check("Groundedness alpha (n=%d)" % len(g), krippendorff_alpha_ordinal(g), 0.894),
        check("Correctness alpha (n=%d)" % len(c), krippendorff_alpha_ordinal(c), 0.687),
    ])


def load_judge(path, run=None):
    out = {}
    with open(os.path.join(ROOT, path)) as f:
        for r in csv.DictReader(f):
            if run is not None and int(r.get("run", 1)) != run:
                continue
            num = lambda v: int(v) if v not in ("", "None", "null", None) else None
            out[int(r["item_id"])] = {"g": num(r["groundedness_judge"]),
                                      "c": num(r["correctness_judge"])}
    return out


def section_2_kappa():
    print("\n[2] Corrected judge-human kappa (temperature 0, chunk text supplied)")
    ok = True
    for cfg, path, run, eg, ec in [
        ("Q4_K_M", "results/judged/gpt-5.4/benchmark_judge_gpt54_phase4_rerun_temp0_chunks.csv", 1, 0.400, 0.144),
        ("Instruct", "results/judged/gpt-5.4/judged_meta-llama_llama-3.2-3b-instruct.csv", 1, 0.389, 0.463),
    ]:
        h = load_human(cfg, run=1) if cfg == "Q4_K_M" else load_human(cfg, run=1)
        j = load_judge(path, run=run)
        hg, jg, hc, jc = [], [], [], []
        for i in sorted(h):
            hgc = median_int(h[i]["g"]) if len(h[i]["g"]) else None
            hcc = median_int(h[i]["c"]) if len(h[i]["c"]) else None
            if i in j:
                if i not in BLOCKED and hgc is not None and j[i]["g"] is not None:
                    hg.append(hgc); jg.append(j[i]["g"])
                if hcc is not None and j[i]["c"] is not None:
                    hc.append(hcc); jc.append(j[i]["c"])
        ok &= check(f"{cfg} kappa_G", linear_weighted_kappa(hg, jg), eg)
        ok &= check(f"{cfg} kappa_C", linear_weighted_kappa(hc, jc), ec)
    return ok


def section_3_retrieval():
    print("\n[3] Retrieval hit-rate (answerable items)")
    d = json.load(open(os.path.join(ROOT, "data", "gold_chunks.json")))
    items = d.get("items", d)
    vals = list(items.values()) if isinstance(items, dict) else items
    lab = [v for v in vals if v.get("gold_chunk_ids")]
    hit1 = hit3 = hit6 = rr = 0.0
    for v in lab:
        ranked = [c["chunk_id"] for c in v["candidates"]]
        gold = set(v["gold_chunk_ids"])
        hit1 += bool(gold & set(ranked[:1]))
        hit3 += bool(gold & set(ranked[:3]))
        hit6 += bool(gold & set(ranked[:6]))
        first = next((i + 1 for i, c in enumerate(ranked) if c in gold), None)
        rr += 1.0 / first if first else 0.0
    n = len(lab)
    return all([
        check("Hit@1 (n=%d)" % n, hit1 / n, 0.667),
        check("Hit@3", hit3 / n, 0.889),
        check("MRR", rr / n, 0.773),
    ])


def section_4_leaderboard():
    print("\n[4] Multi-model leaderboard (GPT-5.4 judge, combined = (G+C)/2)")
    d = os.path.join(ROOT, "results", "judged", "gpt-5.4")
    board = {}
    for f in sorted(os.listdir(d)):
        if not (f.startswith("judged_meta") or f.startswith("judged_openai")
                or f.startswith("judged_anthropic") or f.startswith("judged_deepseek")
                or f.startswith("judged_cohere")):
            continue
        rows = list(csv.DictReader(open(os.path.join(d, f))))
        g = [int(r["groundedness_judge"]) for r in rows if r["groundedness_judge"] not in ("", "None", "null")]
        c = [int(r["correctness_judge"]) for r in rows if r["correctness_judge"] not in ("", "None", "null")]
        model = f.replace("judged_", "").replace(".csv", "")
        board[model] = (np.mean(g) + np.mean(c)) / 2
    print("  ranked combined scores:")
    for m, s in sorted(board.items(), key=lambda kv: -kv[1]):
        print(f"      {s:.3f}  {m}")
    top = max(board.values())
    return check("top model combined (GPT-5.4)", top, 1.833)


def main():
    print("=" * 70)
    print("  Reproducing paper headline numbers from repository data")
    print("=" * 70)
    results = [section_1_interrater(), section_2_kappa(),
               section_3_retrieval(), section_4_leaderboard()]
    print("\n" + "=" * 70)
    print(f"  {sum(results)}/{len(results)} checks passed"
          + ("  — all reproduced" if all(results) else "  — see FAIL lines"))
    print("=" * 70)
    sys.exit(0 if all(results) else 1)


if __name__ == "__main__":
    main()
