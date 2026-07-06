#!/usr/bin/env python3
"""
Rebuilds the precomputed evaluation dataset using mlx-community/Qwen3.5-9B-OptiQ-4bit
as the generation model (enhance_query + extract_metadata's optimized_query rewrite),
instead of the usual mlx-community/Llama-3.2-3B-Instruct-4bit.

Writes to a separate dataset file so the Qwen-enhanced rows can be compared against
the existing Llama-enhanced ones without overwriting them.

Usage:
    python build_dataset_qwen.py
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent))

from mlx_lm import load
from mlx_embeddings.utils import load as emb_load

from evaluation.build_evaluation_dataset import build_dataset

TICKERS     = ["tsla", "pypl"]
OUTPUT_PATH = "/Users/nadavsmacbookair/Desktop/Thesis/data/eval_dataset/limited-new-precomputed-qwen.parquet"

print("Loading generation model (Qwen3.5-9B-OptiQ-4bit)...")
gen_model, gen_tokenizer = load("mlx-community/Qwen3.5-9B-OptiQ-4bit")
print("Loading embedding model...")
embed_model, embed_tokenizer = emb_load("mlx-community/embeddinggemma-300m-bf16")

build_dataset(
    tickers=TICKERS,
    gen_model=gen_model,
    gen_tokenizer=gen_tokenizer,
    embed_model=embed_model,
    embed_tokenizer=embed_tokenizer,
    output_path=OUTPUT_PATH,
)
