"""
Precomputes query enhancement, metadata extraction, and dense embeddings for FinDER
questions and saves them to a Parquet dataset. The dataset is additive: re-running
with new tickers appends rows without recomputing existing (finder_id, ticker) pairs.

Metadata (ticker/year/form_type) is always extracted from the original query --
never the enhanced one -- so enhancement's effect stays isolated to retrieval
(the embedding vector and BM25 query text) and never cascades into the
metadata payload filters. meta_orig and meta_enhanced are therefore always
identical; the column is kept for schema stability, not because the two can
diverge.

Each row also records a git snapshot (commit + dirty flag) for traceability.
"""

import json
import math
import subprocess
from datetime import datetime
from pathlib import Path
import sys

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.parent))

from FinDER import run_finder
from evaluation.rag_functions import enhance_query_batch, extract_metadata_batch, embed_query_batch

DATASET_PATH = "/Users/nadavsmacbookair/Desktop/Thesis/data/eval_dataset/precomputed-qwen.parquet"

# Questions per batched forward pass before an incremental Parquet write. Keeps the
# crash-recovery granularity of the old per-question save loop while still letting
# each LLM call see many prompts at once.
BUILD_CHUNK = 8


def _git_snapshot() -> dict:
    commit = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True).strip()
    dirty  = subprocess.call(
        ["git", "diff", "--quiet", "HEAD"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    ) != 0
    return {"git_commit": commit, "git_dirty": dirty}


def load_dataset(path: str = DATASET_PATH) -> tuple[pd.DataFrame, set]:
    p = Path(path)
    if not p.exists():
        return pd.DataFrame(), set()
    df       = pd.read_parquet(p)
    existing = set(zip(df["finder_id"], df["ticker"]))
    return df, existing


def build_dataset(
    tickers:          list[str],
    gen_model,
    gen_tokenizer,
    embed_model,
    embed_tokenizer,
    output_path: str = DATASET_PATH,
) -> pd.DataFrame:
    snapshot = _git_snapshot()
    if snapshot["git_dirty"]:
        print(f"WARNING: uncommitted changes — entries will not be exactly reproducible "
              f"(commit={snapshot['git_commit']})")
    else:
        print(f"Git snapshot: {snapshot['git_commit']}")

    df, existing = load_dataset(output_path)
    rows = [] if df.empty else df.to_dict("records")
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    for ticker in tickers:
        finder_df = run_finder(tickers=[ticker])
        finder_df = finder_df[:finder_df.shape[0]-1]
        if finder_df.empty:
            print(f"[skip] no FinDER questions for {ticker}")
            continue

        new_rows = [row for _, row in finder_df.iterrows()
                    if (row.get("_id", ""), ticker) not in existing]
        print(f"\n{ticker.upper()}: {len(finder_df)} total | {len(new_rows)} new to compute")

        n_chunks = math.ceil(len(new_rows) / BUILD_CHUNK)
        for c_idx in range(n_chunks):
            lo    = c_idx * BUILD_CHUNK
            chunk = new_rows[lo:lo + BUILD_CHUNK]
            queries = [row.get("query", "") for row in chunk]
            print(f"\n  [chunk {c_idx+1}/{n_chunks}] {len(chunk)} question(s), rows {lo+1}-{lo+len(chunk)}")

            # 1. Query enhancement — one batched forward pass for the whole chunk
            enhanced_list = enhance_query_batch(queries, gen_model, gen_tokenizer, completion_batch_size=1, prefill_batch_size=2)

            # 2. Metadata extraction -- always from the original query. Ticker/
            metas = extract_metadata_batch(queries)

            for meta_orig in metas:
                if meta_orig.get("ticker") != ticker:
                    print(f"  [ticker override] extracted={meta_orig.get('ticker')!r} -> bucket {ticker!r}")
                    meta_orig["ticker"] = ticker

            # 3. Embeddings. Original and enhanced texts are embedded in a single
            # padded pass; an enhancement that came back identical to its original
            # (or was a fallback passthrough) reuses that vector rather than being
            # embedded twice.
            to_embed:  list[str] = list(queries)
            enh_slot:  list[int | None] = []
            for orig, enh in zip(queries, enhanced_list):
                if enh == orig:
                    enh_slot.append(None)
                else:
                    enh_slot.append(len(to_embed))
                    to_embed.append(enh)
            vecs = embed_query_batch(to_embed, embed_model, embed_tokenizer)

            for i, row in enumerate(chunk):
                finder_id  = row.get("_id", "")
                orig_query = queries[i]
                enhanced   = enhanced_list[i]
                meta_orig  = metas[i]
                meta_enh   = meta_orig.copy()
                vec_orig   = vecs[i]
                vec_enh    = vec_orig if enh_slot[i] is None else vecs[enh_slot[i]]

                print(f"  [{lo+i+1}/{len(new_rows)}] {orig_query[:80]}...")
                print(f"  [enhance] → {enhanced[:80]}...")
                print(f"  [meta] ticker={meta_orig.get('ticker')} year={meta_orig.get('year')}")

                truth_ref = row.get("truth_ref", "")
                entry = {
                    "finder_id":      finder_id,
                    "ticker":         ticker,
                    "query":          orig_query,
                    "enhanced_query": enhanced,
                    "truth_answer":   row.get("truth_answer", ""),
                    "truth_ref":      truth_ref if isinstance(truth_ref, str) else json.dumps(truth_ref.tolist() if hasattr(truth_ref, "tolist") else list(truth_ref)),
                    "category":       row.get("category", ""),
                    "query_type":     row.get("type", ""),
                    "meta_orig":      json.dumps(meta_orig),
                    "meta_enhanced":  json.dumps(meta_enh),
                    "vec_orig":       vec_orig,
                    "vec_enhanced":   vec_enh,
                    "git_commit":     snapshot["git_commit"],
                    "git_dirty":      snapshot["git_dirty"],
                    "built_at":       datetime.now().isoformat(),
                }

                rows.append(entry)
                existing.add((finder_id, ticker))

            # saving procedure — once per chunk, so a crash loses at most BUILD_CHUNK questions
            pd.DataFrame(rows).to_parquet(output_path, index=False)
            print(f"  [saved] {len(rows)} rows → {output_path}")

    final = pd.DataFrame(rows)
    print(f"\nDone. Dataset: {len(final)} rows across "
          f"{final['ticker'].nunique() if not final.empty else 0} ticker(s).")
    return final


if __name__ == "__main__":
    from mlx_lm import load
    from mlx_embeddings.utils import load as emb_load

    tickers = ["mdlz","mondelez","ma","mastercard","yum","nem","newmont","iff"]

    print("Loading generation model...")
    gen_model, gen_tokenizer = load("mlx-community/Qwen3.5-9B-OptiQ-4bit")
    print("Loading embedding model...")
    embed_model, embed_tokenizer = emb_load("mlx-community/embeddinggemma-300m-bf16")

    build_dataset(
        tickers=tickers,
        gen_model=gen_model,
        gen_tokenizer=gen_tokenizer,
        embed_model=embed_model,
        embed_tokenizer=embed_tokenizer,
    )
