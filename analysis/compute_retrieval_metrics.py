#!/usr/bin/env python3
"""
Task C step 2 — retrieval hit-rate metrics (SmartEdu 2026 camera-ready).

REFUSES TO RUN until gold_chunks.json carries human labels. Never guesses a
gold chunk: a fabricated label set invalidates the metric.

Run build_gold_chunk_stub.py first, label gold_chunk_ids in gold_chunks.json
against chunk_corpus.md, then run this.

Usage (run from backend/scripts/):
    python3 compute_retrieval_metrics.py

Output:
    Prints overall + per-section metrics.
    Writes retrieval_metrics.csv

Metrics over the 39 non-blocked items (blocked items retrieve nothing):
    Hit@1, Hit@3, Hit@6, MRR, recall@6
An item counts as a hit at k if ANY labeled gold chunk appears in the top-k.
recall@6 is the fraction of that item's gold chunks present in the top 6,
averaged over items. Items with empty_retrieval count as misses.

⚠️  READ BEFORE CITING THESE NUMBERS — corpus-size caveat:
The knowledge corpus is only 10 chunks and retrieval returns a fixed top-6
(the 0.3 score threshold never binds; observed cosines span 0.505–0.880, and
all 39 non-blocked items return exactly 6 chunks). So every item retrieves
6 of 10 chunks and set-membership metrics are nearly saturated by
construction. Three measured facts sharpen that warning:

  * A random retriever scores Hit@6 = 60% when an item has ONE gold chunk,
    but 87% at two gold chunks and 97% at three. The baseline is therefore
    not a constant 60% — it depends on the label set, and this script
    computes it exactly from the actual gold counts (see baselines_for).
  * The 39 items produce only 13 DISTINCT retrieved sets, and one single
    set covers 17 of the 39 items. Set-membership metrics have very little
    variance available to them here regardless of baseline.
  * Ordering, by contrast, is almost fully item-specific: 38 of 39 items
    have a distinct ORDERED top-6.

Lead with Hit@1 and MRR, which measure ranking quality, and disclose the
corpus size. Hit@6/recall@6 must never be reported without the baseline
printed beside them.
"""

import csv
import json
import os
import sys
from collections import Counter, defaultdict
from math import comb

import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
STUB = os.path.join(_HERE, "gold_chunks.json")
OUTPUT_CSV = os.path.join(_HERE, "retrieval_metrics.csv")

CORPUS_SIZE = 10   # chunks; verified by build_gold_chunk_stub.py
TOP_K = 6


def main():
    if not os.path.exists(STUB):
        sys.exit("gold_chunks.json not found — run build_gold_chunk_stub.py first.")

    with open(STUB) as f:
        stub = json.load(f)
    items = stub["items"]

    labeled, unlabeled, no_answer = {}, [], []
    for item_id, v in items.items():
        if v.get("no_answer_in_corpus"):
            no_answer.append(item_id)
        elif v.get("gold_chunk_ids"):
            labeled[item_id] = v
        else:
            unlabeled.append(item_id)

    if unlabeled:
        print(f"🛑 BLOCKED — {len(unlabeled)} of {len(items)} items have no gold "
              f"chunk labels yet.\n")
        print("   Unlabeled item ids: " + ", ".join(sorted(unlabeled, key=int)))
        print("\n   Label gold_chunk_ids in gold_chunks.json against "
              "chunk_corpus.md, then re-run.")
        print("   (Items with genuinely no answer in the corpus: set "
              "no_answer_in_corpus=true instead.)")
        print("\n   No metrics computed. Nothing was written — a fabricated "
              "label set would invalidate the metric.")
        sys.exit(1)

    if not labeled:
        sys.exit("No labeled items with gold chunks — nothing to compute.")

    print(f"Labeled items: {len(labeled)}  |  marked no-answer-in-corpus: "
          f"{len(no_answer)}  |  total: {len(items)}")
    print(f"Corpus: {CORPUS_SIZE} chunks, fixed top-{TOP_K} retrieval\n")

    def metrics_for(subset):
        hits = {1: [], 3: [], 6: []}
        rr, rec = [], []
        for v in subset:
            ranked = [c["chunk_id"] for c in v["candidates"]]
            gold = set(v["gold_chunk_ids"])
            for k in hits:
                hits[k].append(1.0 if gold & set(ranked[:k]) else 0.0)
            first = next((i + 1 for i, c in enumerate(ranked) if c in gold), None)
            rr.append(1.0 / first if first else 0.0)
            rec.append(len(gold & set(ranked[:TOP_K])) / len(gold) if gold else 0.0)
        return {
            "Hit@1": float(np.mean(hits[1])),
            "Hit@3": float(np.mean(hits[3])),
            "Hit@6": float(np.mean(hits[6])),
            "MRR": float(np.mean(rr)),
            "recall@6": float(np.mean(rec)),
            "n": len(subset),
        }

    def baselines_for(subset):
        """Exact random-retriever baseline, conditioned on each item's own gold count.

        The random retriever draws a uniformly random ORDERED 6-subset of the 10
        chunks. Both corrections below matter and neither is cosmetic:

        1. Hit@k depends on how many gold chunks an item has. With g gold chunks,
           P(hit@k) = 1 - C(N-g, k)/C(N, k), NOT g-independent. For k=6 this is
           0.600 at g=1 but 0.867 at g=2 and 0.967 at g=3 — so an item with two
           gold chunks is nearly guaranteed a Hit@6 by chance alone. Averaging a
           flat 60% over a set that contains multi-gold items understates the
           baseline and overstates how much the retriever is adding.
        2. MRR is truncated at rank 6 because only 6 chunks are ever returned, so
           the baseline must be truncated too. E[RR] over a full ranking of all 10
           is 0.293; over the top-6 it is 0.245 at g=1.

        recall@6 is g-independent at TOP_K/CORPUS_SIZE by linearity of expectation.
        """
        per_item = {m: [] for m in ("Hit@1", "Hit@3", "Hit@6", "MRR", "recall@6")}
        for v in subset:
            g = len(set(v["gold_chunk_ids"]))
            for k, m in ((1, "Hit@1"), (3, "Hit@3"), (TOP_K, "Hit@6")):
                miss = (comb(CORPUS_SIZE - g, k) / comb(CORPUS_SIZE, k)
                        if CORPUS_SIZE - g >= k else 0.0)
                per_item[m].append(1.0 - miss)
            err = 0.0
            for r in range(1, TOP_K + 1):
                p_none = (comb(CORPUS_SIZE - g, r - 1) / comb(CORPUS_SIZE, r - 1)
                          if CORPUS_SIZE - g >= r - 1 else 0.0)
                err += p_none * (g / (CORPUS_SIZE - r + 1)) * (1.0 / r)
            per_item["MRR"].append(err)
            per_item["recall@6"].append(TOP_K / CORPUS_SIZE)
        return {m: float(np.mean(vals)) for m, vals in per_item.items()}

    baselines = baselines_for(list(labeled.values()))
    gold_counts = Counter(len(set(v["gold_chunk_ids"])) for v in labeled.values())
    print("Gold chunks per item: "
          + ", ".join(f"{g} gold ×{n}" for g, n in sorted(gold_counts.items())))
    print("Random baselines below are exact for that gold-count mix, "
          "not a flat single-gold assumption.\n")

    rows = []
    overall = metrics_for(list(labeled.values()))
    print("=" * 68)
    print(f"  OVERALL (n={overall['n']})")
    print("=" * 68)
    print(f"  {'metric':10s} {'value':>8s} {'random':>8s}  interpretation")
    for m in ("Hit@1", "Hit@3", "Hit@6", "MRR", "recall@6"):
        b = baselines[m]
        headroom = 1.0 - b            # how much a perfect retriever could add
        lift = (overall[m] - b) / headroom if headroom > 1e-9 else float("nan")
        if overall[m] <= b + 1e-9:
            note = f"AT OR BELOW random ({b:.3f}) — no retrieval signal"
        elif b >= 0.55:
            note = f"random={b:.0%}, only {headroom:.0%} headroom — do NOT cite bare"
        else:
            note = f"discriminative — closes {lift:.0%} of the gap to perfect"
        print(f"  {m:10s} {overall[m]:8.3f} {b:8.3f}  {note}")
        rows.append({"scope": "overall", "section": "", "metric": m,
                     "value": round(overall[m], 4),
                     "random_baseline": round(b, 4), "n": overall["n"]})

    by_section = defaultdict(list)
    for v in labeled.values():
        by_section[v["section"]].append(v)

    print("\n" + "=" * 68)
    print("  PER SECTION")
    print("=" * 68)
    for sec in sorted(by_section):
        s = metrics_for(by_section[sec])
        sb = baselines_for(by_section[sec])   # baseline follows THIS section's gold mix
        print(f"\n  {sec}  (n={s['n']})")
        for m in ("Hit@1", "Hit@3", "Hit@6", "MRR", "recall@6"):
            print(f"    {m:10s} {s[m]:.3f}   (random {sb[m]:.3f})")
            rows.append({"scope": "section", "section": sec, "metric": m,
                         "value": round(s[m], 4),
                         "random_baseline": round(sb[m], 4), "n": s["n"]})

    with open(OUTPUT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["scope", "section", "metric", "value",
                                          "random_baseline", "n"])
        w.writeheader()
        w.writerows(rows)
    print(f"\nWritten to {OUTPUT_CSV}")
    print("\n⚠️  When citing: state the corpus is 10 chunks with fixed top-6 "
          "retrieval.\n   Hit@6/recall@6 sit against a 60% random baseline; lead "
          "with Hit@1 and MRR.")


if __name__ == "__main__":
    main()
