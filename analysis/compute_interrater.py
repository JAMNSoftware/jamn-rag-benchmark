#!/usr/bin/env python3
"""
Inter-rater agreement among the three human raters (R1/R2/R3).

Addresses SmartEdu 2026 R2's complaint that Table II ("...with Inter-Rater
Agreement") reports only human-consensus-vs-judge kappa and nothing about
agreement among the humans themselves.

Setup:
    pip install openpyxl numpy

Usage (run from backend/scripts/):
    python3 compute_interrater.py

Input:
    RAG_Manual_Scoring.xlsx, sheet "Phase4 Rerun — Run 1" — the sheet whose
    R1–R3 averages reproduce the paper's reported human means (G=1.632,
    C=1.240). Runs 2 and 3 were not human-scored.

Output:
    Prints all statistics to stdout.
    Writes interrater_results.csv (one row per statistic).

Notes:
    - Groundedness: blocked items (5, 34, 36, 39) are excluded (no chunks
      retrieved), n=39. Correctness: all items, n=43.
    - Krippendorff's alpha uses the ordinal metric (scores are ordinal 0/1/2).
    - Also verifies the two consensus definitions used by the pipeline:
      mean of R1–R3 (paper Tables I & II) vs median-rounded (ordinal input
      to Cohen's weighted kappa in compute_kappa.py). Both are checked
      against their known values; neither is "corrected".
"""

import csv
import os
import statistics
import sys

import numpy as np

# The human-scoring workbook is not distributed with this repo. Point this at
# your own copy via the RAG_SCORING_XLSX environment variable. The published
# per-item ratings are available portably in data/human_ratings.csv, which
# analysis/reproduce.py uses instead.
EXCEL_PATH = os.environ.get("RAG_SCORING_XLSX", "RAG_Manual_Scoring.xlsx")
SHEET_NAME = "Phase4 Rerun — Run 1"
OUTPUT_CSV = "interrater_results.csv"

BLOCKED_IDS = {5, 34, 36, 39}

# Excel column indices (0-based, row 4 onward is data)
COL_ITEM = 0      # A: item number
COL_G_R1 = 6      # G: Groundedness R1
COL_G_R2 = 7      # H: Groundedness R2
COL_G_R3 = 8      # I: Groundedness R3
COL_C_R1 = 10     # K: Correctness R1
COL_C_R2 = 11     # L: Correctness R2
COL_C_R3 = 12     # M: Correctness R3

CATEGORIES = [0, 1, 2]


def median_int(values):
    clean = [int(v) for v in values if v is not None and v != ""]
    if not clean:
        return None
    return round(statistics.median(clean))


def cohen_kappa(rater1, rater2, categories=CATEGORIES):
    n = len(rater1)
    if n == 0:
        return float("nan")
    p_o = sum(a == b for a, b in zip(rater1, rater2)) / n
    counts1 = {c: rater1.count(c) / n for c in categories}
    counts2 = {c: rater2.count(c) / n for c in categories}
    p_e = sum(counts1[c] * counts2[c] for c in categories)
    if p_e == 1.0:
        return 1.0
    return (p_o - p_e) / (1.0 - p_e)


def linear_weighted_kappa(rater1, rater2, categories=CATEGORIES):
    n = len(rater1)
    if n == 0:
        return float("nan")
    max_diff = max(categories) - min(categories)

    def weight(i, j):
        return 1.0 - abs(i - j) / max_diff

    counts1 = {c: rater1.count(c) / n for c in categories}
    counts2 = {c: rater2.count(c) / n for c in categories}
    p_o = sum(weight(a, b) for a, b in zip(rater1, rater2)) / n
    p_e = sum(counts1[i] * counts2[j] * weight(i, j)
              for i in categories for j in categories)
    if p_e == 1.0:
        return 1.0
    return (p_o - p_e) / (1.0 - p_e)


def fleiss_kappa(ratings_by_item, categories=CATEGORIES):
    """ratings_by_item: list of per-item rating lists (all same length m)."""
    n = len(ratings_by_item)
    if n == 0:
        return float("nan")
    m = len(ratings_by_item[0])
    counts = np.zeros((n, len(categories)))
    cat_idx = {c: i for i, c in enumerate(categories)}
    for i, ratings in enumerate(ratings_by_item):
        for r in ratings:
            counts[i, cat_idx[r]] += 1
    p_j = counts.sum(axis=0) / (n * m)
    p_i = (np.sum(counts ** 2, axis=1) - m) / (m * (m - 1))
    p_bar = p_i.mean()
    p_e = np.sum(p_j ** 2)
    if p_e == 1.0:
        return 1.0
    return (p_bar - p_e) / (1.0 - p_e)


def krippendorff_alpha_ordinal(ratings_by_item, categories=CATEGORIES):
    """Krippendorff's alpha with the ordinal metric, via the coincidence
    matrix (Krippendorff 2011, "Computing Krippendorff's Alpha-Reliability").
    Handles complete data from >=2 raters; categories must be sorted."""
    cat_idx = {c: i for i, c in enumerate(categories)}
    k = len(categories)
    o = np.zeros((k, k))  # coincidence matrix
    for ratings in ratings_by_item:
        m_u = len(ratings)
        if m_u < 2:
            continue
        for a in range(m_u):
            for b in range(m_u):
                if a != b:
                    o[cat_idx[ratings[a]], cat_idx[ratings[b]]] += 1.0 / (m_u - 1)
    n_c = o.sum(axis=1)          # marginals
    n_total = n_c.sum()
    if n_total <= 1:
        return float("nan")

    # ordinal squared distance between categories c<=d:
    # (sum of marginals from c..d, minus half the two endpoints)^2
    delta2 = np.zeros((k, k))
    for c in range(k):
        for d in range(c + 1, k):
            delta2[c, d] = (n_c[c:d + 1].sum() - (n_c[c] + n_c[d]) / 2.0) ** 2
            delta2[d, c] = delta2[c, d]

    d_o = np.sum(o * delta2) / n_total
    d_e = np.sum(np.outer(n_c, n_c) * delta2) / (n_total * (n_total - 1))
    if d_e == 0:
        return 1.0
    return 1.0 - d_o / d_e


def main():
    try:
        import openpyxl
    except ImportError:
        sys.exit("Run: pip install openpyxl")

    script_dir = os.path.dirname(os.path.abspath(__file__))

    # --- Load human scores from Excel ---
    wb = openpyxl.load_workbook(EXCEL_PATH, data_only=True)
    if SHEET_NAME not in wb.sheetnames:
        sys.exit(f"Sheet not found: {SHEET_NAME!r}. Available: {wb.sheetnames}")
    ws = wb[SHEET_NAME]

    human = {}
    for row in ws.iter_rows(min_row=4, values_only=True):
        item_id = row[COL_ITEM]
        if item_id is None:
            continue
        item_id = int(item_id)
        human[item_id] = {
            "g_r1": row[COL_G_R1], "g_r2": row[COL_G_R2], "g_r3": row[COL_G_R3],
            "c_r1": row[COL_C_R1], "c_r2": row[COL_C_R2], "c_r3": row[COL_C_R3],
        }

    if len(human) != 43:
        sys.exit(f"Expected 43 items, found {len(human)} — check sheet/rows")

    def clean_triple(vals):
        out = []
        for v in vals:
            if v is None or v == "":
                return None
            out.append(int(v))
        return out

    g_items, c_items = {}, {}
    for item_id, h in sorted(human.items()):
        g = clean_triple([h["g_r1"], h["g_r2"], h["g_r3"]])
        c = clean_triple([h["c_r1"], h["c_r2"], h["c_r3"]])
        if item_id not in BLOCKED_IDS:
            if g is None:
                sys.exit(f"Item {item_id}: missing groundedness rating (non-blocked)")
            g_items[item_id] = g
        if c is None:
            sys.exit(f"Item {item_id}: missing correctness rating")
        c_items[item_id] = c

    print(f"Loaded {len(human)} items from sheet {SHEET_NAME!r}")
    print(f"Groundedness n={len(g_items)} (blocked {sorted(BLOCKED_IDS)} excluded)")
    print(f"Correctness  n={len(c_items)}")

    results = []  # rows for the CSV

    def add(metric, statistic, value, n, note=""):
        results.append({
            "metric": metric, "statistic": statistic,
            "value": round(value, 4) if value == value else "",
            "n": n, "note": note,
        })

    for metric, items in (("groundedness", g_items), ("correctness", c_items)):
        ratings = [items[i] for i in sorted(items)]
        n = len(ratings)
        label = metric.upper()
        print(f"\n{'=' * 58}\n  {label} — inter-rater agreement (n={n})\n{'=' * 58}")

        alpha = krippendorff_alpha_ordinal(ratings)
        fk = fleiss_kappa(ratings)
        print(f"  Krippendorff's alpha (ordinal) : {alpha:.4f}")
        print(f"  Fleiss' kappa                  : {fk:.4f}")
        add(metric, "krippendorff_alpha_ordinal", alpha, n)
        add(metric, "fleiss_kappa", fk, n)

        pairs = [("R1", "R2", 0, 1), ("R1", "R3", 0, 2), ("R2", "R3", 1, 2)]
        wk_values = []
        for name_a, name_b, a, b in pairs:
            ra = [r[a] for r in ratings]
            rb = [r[b] for r in ratings]
            wk = linear_weighted_kappa(ra, rb)
            uk = cohen_kappa(ra, rb)
            wk_values.append(wk)
            print(f"  Cohen's kappa {name_a}-{name_b}: linear-weighted={wk:.4f}  unweighted={uk:.4f}")
            add(metric, f"cohen_linear_weighted_{name_a}_{name_b}", wk, n)
            add(metric, f"cohen_unweighted_{name_a}_{name_b}", uk, n)
        add(metric, "cohen_linear_weighted_pairwise_mean",
            sum(wk_values) / len(wk_values), n)

        print(f"  Per-rater mean (SD):")
        for idx, name in enumerate(["R1", "R2", "R3"]):
            col = [r[idx] for r in ratings]
            mean, sd = np.mean(col), np.std(col, ddof=1)
            print(f"    {name}: {mean:.4f} ({sd:.4f})")
            add(metric, f"mean_{name}", mean, n)
            add(metric, f"sd_{name}", sd, n)

        # exact 3-way agreement, for context
        unanimous = sum(1 for r in ratings if len(set(r)) == 1)
        print(f"  Unanimous items: {unanimous}/{n} ({100 * unanimous / n:.1f}%)")
        add(metric, "unanimous_fraction", unanimous / n, n)

        # --- consensus-definition verification (do not "fix" either) ---
        grand_mean = float(np.mean([v for r in ratings for v in r]))
        median_mean = float(np.mean([median_int(r) for r in ratings]))
        print(f"  Consensus check — mean of all ratings   : {grand_mean:.4f}"
              f"  (paper Tables I/II definition)")
        print(f"  Consensus check — mean of median-rounded: {median_mean:.4f}"
              f"  (compute_kappa.py ordinal input)")
        add(metric, "consensus_mean_of_ratings", grand_mean, n,
            "paper Tables I & II definition (mean of R1-R3)")
        add(metric, "consensus_mean_of_median_rounded", median_mean, n,
            "ordinal consensus fed to weighted kappa (compute_kappa.py median_int)")

    out_path = os.path.join(script_dir, OUTPUT_CSV)
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["metric", "statistic", "value", "n", "note"])
        writer.writeheader()
        writer.writerows(results)
    print(f"\nWritten to {out_path}\n")


if __name__ == "__main__":
    main()
