"""
Precomputes query enhancement, metadata extraction, and dense embeddings for FinDER
questions and saves them to a Parquet dataset. The dataset is additive: re-running
with new tickers appends rows without recomputing existing (finder_id, ticker) pairs.

Each row also records a git snapshot (commit + dirty flag) for traceability.
"""

import json
import subprocess
from datetime import datetime
from pathlib import Path
import sys

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.parent))

from FinDER import run_finder
from evaluation.rag_functions import enhance_query, extract_metadata, embed_query

DATASET_PATH = "/Users/nadavsmacbookair/Desktop/Thesis/data/eval_dataset/limited-new-precomputed.parquet"


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
        if finder_df.empty:
            print(f"[skip] no FinDER questions for {ticker}")
            continue

        new_rows = [row for _, row in finder_df.iterrows()
                    if (row.get("_id", ""), ticker) not in existing]
        print(f"\n{ticker.upper()}: {len(finder_df)} total | {len(new_rows)} new to compute")

        for q_idx, row in enumerate(new_rows):
            finder_id  = row.get("_id", "")
            orig_query = row.get("query", "")
            print(f"\n  [{q_idx+1}/{len(new_rows)}] {orig_query[:80]}...")

            # 1. Query enhancement
            enhanced  = enhance_query(orig_query, gen_model, gen_tokenizer)
            same_text = (enhanced == orig_query)
            print(f"  [enhance] → {enhanced[:80]}...")

            # 2. Metadata extraction
            meta_orig = extract_metadata(orig_query, gen_model, gen_tokenizer)
            meta_enh  = meta_orig.copy() if same_text else extract_metadata(enhanced, gen_model, gen_tokenizer)
            print(f"  [meta/orig] ticker={meta_orig.get('ticker')} year={meta_orig.get('year')}")
            if not same_text:
                print(f"  [meta/enh]  ticker={meta_enh.get('ticker')}  year={meta_enh.get('year')}")

            # 3. Embeddings
            vec_orig = embed_query(orig_query, embed_model, embed_tokenizer)
            vec_enh  = vec_orig if same_text else embed_query(enhanced, embed_model, embed_tokenizer)

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

            # saving procedure
            pd.DataFrame(rows).to_parquet(output_path, index=False)
            print(f"  [saved] {len(rows)} rows → {output_path}")

    final = pd.DataFrame(rows)
    print(f"\nDone. Dataset: {len(final)} rows across "
          f"{final['ticker'].nunique() if not final.empty else 0} ticker(s).")
    return final


if __name__ == "__main__":
    from mlx_lm import load
    from mlx_embeddings.utils import load as emb_load

    tickers = ["wmt","walmart","nvda","nvidia"]

    print("Loading generation model...")
    gen_model, gen_tokenizer = load("mlx-community/Llama-3.2-3B-Instruct-4bit")
    print("Loading embedding model...")
    embed_model, embed_tokenizer = emb_load("mlx-community/embeddinggemma-300m-bf16")

    build_dataset(
        tickers=tickers,
        gen_model=gen_model,
        gen_tokenizer=gen_tokenizer,
        embed_model=embed_model,
        embed_tokenizer=embed_tokenizer,
    )
