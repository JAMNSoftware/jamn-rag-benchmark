# JAMN Bot RAG Benchmark

Reproducibility artifacts for the paper **"Unjamming Deferrals: Supporting Students
Through JAMN Bot, a Mobile RAG-Grounded Policy Assistant"** (IEEE SmartEdu 2026).

This repository contains the benchmark dataset, the LLM-judge prompt, all model outputs and
judge scores, the human ratings, and the analysis needed to reproduce every headline number
in the paper. It is **data and analysis only**; the JAMN Bot application itself is a separate
system.

## Reproduce the headline numbers

Requirements: Python 3.9+ and `numpy`. No network access or API keys are needed — every
number below is recomputed from the files in this repository.

```bash
pip install numpy
python3 analysis/reproduce.py
```

Expected output: `8/8 checks passed`, covering inter-rater agreement, corrected judge–human
kappa (both serving configurations), retrieval hit-rate, and the multi-model leaderboard.

## What is here

```
data/
  ground_truths.json        43 benchmark items: id, section, blocked flag, question, gold answer
  knowledge_base/           the two source policy documents (policies.md, procedures.md)
  chunk_corpus.md           the 10 retrievable chunks derived from the knowledge base
  gold_chunks.json          per-item gold-chunk labels for retrieval evaluation + retriever top-6
  human_ratings.csv         anonymized per-item R1/R2/R3 Groundedness & Correctness scores
judge/
  system_prompt.md          the rubric + 3 calibration examples given to every judge (temperature 0)
results/
  benchmark/                per-model answers + retrieved chunk IDs + scores (7 models x 43 x 3 runs)
  judged/                   per-model GPT-5.4 judge scores; gemini/ and qwen/ hold the two
                            independent judges; benchmark_judge_gpt54_phase4_rerun_temp0_chunks.csv
                            is the corrected Q4_K_M judging (temperature 0, chunk text supplied)
  ablation/                 temperature sweep (0.0-1.0) on Llama 3.2 3B Instruct
metrics/                    precomputed result tables (inter-rater, retrieval, jury, panel, variance)
analysis/
  reproduce.py              self-contained; regenerates the headline numbers from data/ + results/
  compute_interrater.py     as-run analysis scripts, provided for transparency (see note below)
  analyze_jury.py           ...
  ...
```

## Key facts for interpreting the data

- **Benchmark:** 43 items in five categories — Exam Deferral (10), Privacy (5), Scheduling
  (5), Cross-Policy (15), Adversarial/Out-of-Scope (8). Items 5, 34, 36, 39 are blocked by a
  preflight regex before retrieval, so Groundedness is scored over n=39 and Correctness over
  n=43.
- **Corpus size:** the knowledge base is deliberately small — **10 chunks** — and the
  retriever returns a fixed top-6. Retrieval set-membership metrics (Hit@6, recall@6)
  therefore sit on a high random baseline; the ranking metrics (Hit@1, MRR) carry the signal.
- **Judging:** all judge scoring in the study was at **temperature 0**. The originally
  submitted Q4_K_M judging used temperature 0.3 and, for the production pipeline, passed the
  judge chunk identifiers rather than chunk text; it has been superseded by
  `benchmark_judge_gpt54_phase4_rerun_temp0_chunks.csv` and is not included here.
- **Judges:** GPT-5.4 (also a candidate), Gemini 3.5 Flash, and Qwen3.5 397B. All three are
  proprietary hosted models; no reproducibility or open-weight claim is made about any judge.
- **Human ratings:** produced by three members of the research team (R1/R2/R3), who rated
  model outputs against the shared rubric. These are author-generated annotations of machine
  output, not human-subjects data.

## A note on the `analysis/` scripts

`reproduce.py` is fully portable and reads only this repository. The other scripts
(`compute_interrater.py`, `analyze_jury.py`, `compare_panels.py`,
`compute_retrieval_metrics.py`, `compute_leaderboard_variance.py`) are the **exact scripts
used in the study**, included for transparency. Some reference the original project layout or
the source scoring workbook and are not meant to run unmodified here; `reproduce.py` is the
supported reproduction path.

## Citation

See [CITATION.cff](CITATION.cff). Please cite the paper if you use these artifacts.

## License

- **Data, judge prompt, and results** (`data/`, `judge/`, `results/`, `metrics/`): released
  under [CC BY 4.0](LICENSE-DATA).
- **Analysis code** (`analysis/`): released under the [MIT License](LICENSE-CODE).
