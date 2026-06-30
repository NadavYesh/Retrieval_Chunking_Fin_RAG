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
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent))

import mlx.core as mx
from mlx_lm import load

from evaluation.build_evaluation_dataset import DATASET_PATH, load_dataset
from evaluation.run_rag_lazy import run_multi_evaluation_lazy, write_config
from evaluation.evaluation_run import run_multi_analysis, load_corpus

# ── CONFIG ────────────────────────────────────────────────────────────────────
TICKERS    = ["pypl", "tsla"]
LEVELS     = [
    "--limited --level 1 DENSE",
    "--limited --level 2 DENSE",
    "--limited --level 3 DENSE",
]
MODES      = ["hybrid", "dense"]
OUTPUT_DIR = "/Users/nadavsmacbookair/Desktop/Thesis/data/eval_results"
TOP_K      = 5
USE_LLM_JUDGE = False   # set True to load Phi-4 and judge answers post-generation
# ──────────────────────────────────────────────────────────────────────────────

# ── 1. Dataset ────────────────────────────────────────────────────────────────
print("Loading precomputed dataset...")
dataset, _ = load_dataset(DATASET_PATH)
if dataset.empty:
    raise SystemExit(f"Dataset not found at {DATASET_PATH}. Run build_evaluation_dataset.py first.")
print(f"  {len(dataset)} rows across {dataset['ticker'].nunique()} ticker(s)")

# ── 2. Generation model ───────────────────────────────────────────────────────
print("\nLoading generation model...")
gen_model, gen_tokenizer = load("mlx-community/Qwen3.5-9B-OptiQ-4bit")

# ── 3. RAG evaluation (one JSON per ticker) ───────────────────────────────────
eval_paths: list[str] = []
for ticker in TICKERS:
    cfg = write_config(tickers=[ticker], levels=LEVELS, retrieval_modes=MODES)
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

# ── 4. Free generation model before loading corpus + judge ────────────────────
print("\nFreeing generation model...")
del gen_model, gen_tokenizer
gc.collect()
mx.clear_cache()

# ── 5. Corpus ─────────────────────────────────────────────────────────────────
print("Loading corpus for analysis...")
corpus, fp_index, corpus_text_map = load_corpus()
print(f"  {len(corpus)} chunks loaded")

# ── 6. Optional judge model ───────────────────────────────────────────────────
judge_model, judge_tok = None, None
if USE_LLM_JUDGE:
    from mlx_lm import load as mlx_load
    print("\nLoading Phi-4 judge model...")
    judge_model, judge_tok = mlx_load("mlx-community/phi-4-4bit")

# ── 7. Analysis (one Excel per ticker) ───────────────────────────────────────
for path in eval_paths:
    print(f"\n{'='*60}\nAnalysing: {Path(path).name}\n{'='*60}")
    run_multi_analysis(Path(path), corpus, fp_index, corpus_text_map, judge_model, judge_tok)

print("\n── Pipeline complete ──")
print(f"  Eval JSONs  : {[Path(p).name for p in eval_paths]}")
print(f"  Analysis dir: {OUTPUT_DIR}/analysis/")
