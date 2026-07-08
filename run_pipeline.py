#!/usr/bin/env python3
"""
End-to-end pipeline: RAG evaluation → analysis in one shot.

Runs run_multi_evaluation_lazy for each ticker, then immediately runs
run_multi_analysis on each output file. The generation model is released
from memory before the corpus and judge model are loaded.

Usage:
    python run_pipeline.py

Edit the CONFIG block below to change tickers, levels, modes, etc.
"""

import gc
import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent))

import mlx.core as mx
from mlx_lm import load

from evaluation.build_evaluation_dataset import load_dataset
from evaluation.run_rag_lazy import DATASET_PATH, run_multi_evaluation_lazy, write_config, COLLECTIONS
from evaluation.evaluation_run import run_multi_analysis, load_corpus

# ── CONFIG ────────────────────────────────────────────────────────────────────
TICKERS         = ["pypl", "tsla"]
OUTPUT_DIR      = "/Users/nadavsmacbookair/Desktop/Thesis/data/eval_results"
MERGED_PATH     = Path(OUTPUT_DIR) / "eval_multi_merged_PYPL_TSLA.json"
TOP_K           = 5
USE_LLM_JUDGE   = False   # set True to load Phi-4 and judge answers post-generation

# Full grid: every level × every retrieval mode × every combination of
# enhance_query_flag / section_alpha (via write_config). Year filtering is
# not a config toggle -- it's deterministic per query (see write_config's
# docstring in evaluation/run_rag_lazy.py).
LEVELS          = list(COLLECTIONS.keys())
RETRIEVAL_MODES = ["hybrid", "dense", "sparse"]
# ──────────────────────────────────────────────────────────────────────────────

# ── 1. Dataset ────────────────────────────────────────────────────────────────
print("Loading precomputed dataset...")
dataset, _ = load_dataset(DATASET_PATH)
if dataset.empty:
    raise SystemExit(f"Dataset not found at {DATASET_PATH}. Run build_evaluation_dataset.py first.")
print(f"  {len(dataset)} rows across {dataset['ticker'].nunique()} ticker(s)")

# ── 2. Generation model ───────────────────────────────────────────────────────
print("\nSkipping generation model (retrieval-only run).")
#gen_model, gen_tokenizer = load("mlx-community/Qwen3.5-9B-OptiQ-4bit")
gen_model, gen_tokenizer = None, None

# ── 3. RAG evaluation (one JSON per ticker) ───────────────────────────────────
eval_paths: list[str] = []
for ticker in TICKERS:
    cfg = write_config(tickers=[ticker], levels=LEVELS, retrieval_modes=RETRIEVAL_MODES)
    print(f"\n{'='*60}\nEvaluating ticker: {ticker.upper()}  |  {len(cfg)} configs\n{'='*60}")
    out_path = run_multi_evaluation_lazy(
        configs=cfg,
        dataset=dataset,
        gen_model=gen_model,
        gen_tokenizer=gen_tokenizer,
        output_dir=OUTPUT_DIR,
        top_k=TOP_K,
    )
    eval_paths.append(out_path)

# ── 4. Free (no-op here since no model was loaded) ────────────────────────────
del gen_model, gen_tokenizer
gc.collect()
mx.clear_cache()

# # ── 5. Corpus ─────────────────────────────────────────────────────────────────
# print("Loading corpus for analysis...")
# corpus, fp_index, corpus_text_map = load_corpus()
# print(f"  {len(corpus)} chunks loaded")

# # ── 6. Analysis (one Excel per ticker; retrieval-only, no LLM judge) ─────────
# # The LLM judge is a deliberately separate stage -- run apply_llm_judge in
# # evaluation_run.py afterwards, pointed at the analysis_*.pkl saved below.
# for path in eval_paths:
#     print(f"\n{'='*60}\nAnalysing: {Path(path).name}\n{'='*60}")
#     run_multi_analysis(Path(path), corpus, fp_index, corpus_text_map)

# # ── 8. Merge new outputs into combined JSON ───────────────────────────────────
# print(f"\nMerging into {MERGED_PATH.name} …")
# if MERGED_PATH.exists():
#     with open(MERGED_PATH) as f:
#         merged = json.load(f)
# else:
#     merged = {}

# all_cols: set[str] = set(merged.keys())
# for ep in eval_paths:
#     with open(ep) as f:
#         d = json.load(f)
#     all_cols.update(d.keys())

# for col in all_cols:
#     if col not in merged:
#         merged[col] = {}

# global_idx = max((int(k) for k in merged.get("run_id", {}).keys()), default=-1) + 1
# added = 0
# for ep in eval_paths:
#     with open(ep) as f:
#         d = json.load(f)
#     for li in sorted(d["run_id"].keys(), key=int):
#         gi = str(global_idx)
#         for col in all_cols:
#             merged[col][gi] = d.get(col, {}).get(li)
#         global_idx += 1
#         added += 1

# with open(MERGED_PATH, "w") as f:
#     json.dump(merged, f)

# print(f"  Added {added} rows → {MERGED_PATH.name}  ({global_idx} total rows)")
# print("\n── Pipeline complete ──")
# print(f"  Eval JSONs  : {[Path(p).name for p in eval_paths]}")
# print(f"  Analysis dir: {OUTPUT_DIR}/analysis/")
# print(f"  Merged JSON : {MERGED_PATH.name}")
