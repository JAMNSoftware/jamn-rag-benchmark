#!/usr/bin/env python3
"""
Panel-agreement comparison on a MATCHED item set — human raters vs LLM judges.

Why this exists: `analyze_jury.py` computes inter-judge agreement across all 903
judged rows (7 candidate models), while the human inter-rater figures come from
a single model's outputs. Agreement statistics depend on the score distribution
of the items being rated, so those two numbers are NOT directly comparable — a
judge panel scoring everything from Llama-1B's near-unusable output to GPT-5.4's
faces a very different item mix than humans who only saw one model.

This script removes that confound by restricting BOTH panels to the one
candidate that humans and judges both scored:

    meta-llama/llama-3.2-3b-instruct, all 3 runs, all 43 items

Humans: RAG_Manual_Scoring.xlsx sheets "llama-3.23b-instruct Run 1/2/3"
Judges: results/judged_meta-llama_llama-3.2-3b-instruct.csv (published GPT-5.4)
        results/judge_<slug>/judged_meta-llama_llama-3.2-3b-instruct.csv

A "unit" is one (item, run) pair, identical for both panels — 129 units for
correctness, 117 for groundedness (blocked items 5/34/36/39 excluded).

Usage (run from backend/scripts/):
    python3 compare_panels.py
    python3 compare_panels.py --judges openai/gpt-5.4 google/gemini-3.5-flash

Output:
    Prints the matched comparison; writes panel_comparison.csv
"""

import argparse
import csv
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from compute_interrater import krippendorff_alpha_ordinal, fleiss_kappa

csv.field_size_limit(sys.maxsize)

_HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(_HERE, "results")
# Not distributed with this repo; set RAG_SCORING_XLSX to your own copy. The
# published per-item ratings are in data/human_ratings.csv (see reproduce.py).
EXCEL = os.environ.get("RAG_SCORING_XLSX", "RAG_Manual_Scoring.xlsx")
CANDIDATE_CSV = "judged_meta-llama_llama-3.2-3b-instruct.csv"
SHEETS = {1: "llama-3.23b-instruct Run 1",
          2: "llama-3.23b-instruct Run 2",
          3: "llama-3.23b-instruct Run 3"}
BLOCKED = {5, 34, 36, 39}
OUT = os.path.join(_HERE, "panel_comparison.csv")

COL_ITEM, COL_G, COL_C = 0, (6, 7, 8), (10, 11, 12)
INCUMBENT = "openai/gpt-5.4"


def load_humans():
    """-> {(item, run): {"g": [r1,r2,r3] | None, "c": [...] | None}}"""
    import openpyxl
    wb = openpyxl.load_workbook(EXCEL, data_only=True)
    out = {}
    for run, sheet in SHEETS.items():
        if sheet not in wb.sheetnames:
            sys.exit(f"missing sheet: {sheet}")
        ws = wb[sheet]
        for row in ws.iter_rows(min_row=4, values_only=True):
            if not row or row[COL_ITEM] is None:
                continue
            try:
                item = int(row[COL_ITEM])
            except (TypeError, ValueError):
                continue

            def triple(cols):
                vals = []
                for c in cols:
                    v = row[c] if len(row) > c else None
                    if v is None or v == "":
                        return None
                    vals.append(int(v))
                return vals
            out[(item, run)] = {"g": triple(COL_G), "c": triple(COL_C)}
    return out


def load_judge(judge):
    d = RESULTS if judge == INCUMBENT else os.path.join(
        RESULTS, f"judge_{judge.replace('/', '_')}")
    path = os.path.join(d, CANDIDATE_CSV)
    if not os.path.exists(path):
        return None
    out = {}
    with open(path) as f:
        for r in csv.DictReader(f):
            if (r.get("justification") or "").startswith("ERROR"):
                continue

            def num(v):
                return int(v) if v not in ("", "None", "null", None) else None
            out[(int(r["item_id"]), int(r["run"]))] = {
                "g": num(r["groundedness_judge"]), "c": num(r["correctness_judge"])}
    return out


def agreement(units):
    if len(units) < 2:
        return None
    return (krippendorff_alpha_ordinal(units), fleiss_kappa(units), len(units))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--judges", nargs="+",
                    default=[INCUMBENT, "google/gemini-3.5-flash",
                             "qwen/qwen3.5-397b-a17b"])
    args = ap.parse_args()

    humans = load_humans()
    judges, missing = {}, []
    for j in args.judges:
        d = load_judge(j)
        (judges.setdefault(j, d) if d else missing.append(j))
    if missing:
        print(f"not available (sweep incomplete): {', '.join(missing)}")
    if not judges:
        sys.exit("no judge data for the matched candidate")
    print(f"Matched candidate: meta-llama/llama-3.2-3b-instruct, runs 1-3")
    print(f"Judges included: {', '.join(judges)}\n")

    rows_out = []
    for metric, key in (("groundedness", "g"), ("correctness", "c")):
        # units where BOTH panels are complete — identical basis for both
        keys = []
        for k in sorted(humans):
            if key == "g" and k[0] in BLOCKED:
                continue
            h = humans[k][key]
            if h is None:
                continue
            if any(k not in judges[j] or judges[j][k][key] is None for j in judges):
                continue
            keys.append(k)

        h_units = [humans[k][key] for k in keys]
        j_units = [[judges[j][k][key] for j in judges] for k in keys]

        ha, hf, hn = agreement(h_units)
        ja, jf, jn = agreement(j_units)

        print("=" * 70)
        print(f"  {metric.upper()} — matched units: {len(keys)} "
              f"(same items & runs for both panels)")
        print("=" * 70)
        print(f"  {'panel':28s} {'Krippendorff α':>15s} {'Fleiss κ':>10s} {'n':>6s}")
        print(f"  {'3 human raters':28s} {ha:15.4f} {hf:10.4f} {hn:6d}")
        print(f"  {f'{len(judges)} LLM judges':28s} {ja:15.4f} {jf:10.4f} {jn:6d}")
        gap = ha - ja
        print(f"\n  Δ (human − judge) α = {gap:+.4f} — "
              f"{'humans agree more' if gap > 0 else 'judges agree more'}")

        for panel, a, f_, n in (("humans", ha, hf, hn),
                                (f"judges_{len(judges)}", ja, jf, jn)):
            rows_out.append({"metric": metric, "panel": panel,
                             "krippendorff_alpha_ordinal": round(a, 4),
                             "fleiss_kappa": round(f_, 4), "n_units": n,
                             "basis": "matched: llama-3.2-3b-instruct runs 1-3"})
        print()

    with open(OUT, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["metric", "panel",
                                          "krippendorff_alpha_ordinal",
                                          "fleiss_kappa", "n_units", "basis"])
        w.writeheader()
        w.writerows(rows_out)
    print(f"Written to {OUT}")
    print("\nNOTE: report THIS as the panel-agreement comparison. The all-903-row")
    print("inter-judge figure from analyze_jury.py spans 7 candidate models and is")
    print("not comparable to a human panel that scored only one.")


if __name__ == "__main__":
    main()
