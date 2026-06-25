#%%
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
import pandas as pd

from search_engine import search_with_payload, search_bm25, rrf_fuse, generate_llm_answer
from prompts import META_EXTRACT_PROMPT, QUERY_ENHANCEMENT_PROMPT
from FinDER import run_finder
from db.database import get_qdrant_client
from mlx_lm import generate, load
import mlx.core
from mlx_embeddings.utils import load as emb_load
from utils import parse_metadata_response

client = get_qdrant_client()
#%%
# COLLECTIONS_1 = {
#     "doc":      {"coll_name": "--level 0", "use_parent_fetch": False},
#     "header":   {"coll_name": "--level 1", "use_parent_fetch": False},
#     "child":    {"coll_name": "--level 2", "use_parent_fetch": True},
#     "enriched": {"coll_name": "--level 3", "use_parent_fetch": True},
# }

COLLECTIONS_2 = {
    "header":   {"coll_name": "--level 1 DENSE", "use_parent_fetch": False},
}
PARENT_COLL  = "--level 1 DENSE"
COLL_BM25    = "--level 1 BM25"
VALID_MODES  = {"dense", "sparse", "hybrid"}


def extract_metadata(query: str, model, tokenizer) -> dict:
    messages = [
        {"role": "system", "content": META_EXTRACT_PROMPT},
        {"role": "user",   "content": query},
    ]
    prompt   = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    response = generate(model, tokenizer, prompt=prompt, verbose=False)
    return parse_metadata_response(response, fallback_query=query)


def enhance_query(query: str, model, tokenizer) -> str:
    prompt = QUERY_ENHANCEMENT_PROMPT.format(query=query)
    messages = [{"role": "user", "content": prompt}]
    formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    return generate(model, tokenizer, prompt=formatted, verbose=False).strip()


def run_evaluation(
    finder_df:          pd.DataFrame,
    gen_model           = None,
    gen_tokenizer       = None,
    embed_model         = None,
    embed_tokenizer     = None,
    output_dir: str     = "/Users/nadavsmacbookair/Desktop/Thesis/data/eval_results",
    top_k:      int     = 5,
    levels:     list    = None,
    modes:      list    = None,   # any subset of ["dense", "sparse", "hybrid"]
    enhance_query_flag: bool = False,
) -> pd.DataFrame:
    if levels is None:
        levels = list(COLLECTIONS_2.keys())
    if isinstance(levels, str):
        levels = [levels]
    if modes is None:
        modes = ["dense"]
    if isinstance(modes, str):
        modes = [modes]
    unknown = set(modes) - VALID_MODES
    if unknown:
        raise ValueError(f"Unknown modes: {unknown}. Valid: {VALID_MODES}")

    if finder_df.empty:
        print("No FinDER questions found.")
        return pd.DataFrame()

    need_dense  = any(m in {"dense",  "hybrid"} for m in modes)
    need_sparse = any(m in {"sparse", "hybrid"} for m in modes)
    # For hybrid, retrieve wider candidate sets before RRF fusion.
    prefetch_k  = top_k * 3 if "hybrid" in modes else top_k ###################

    total_runs = len(finder_df) * len(levels) * len(modes)
    print(f"Questions: {len(finder_df)} | Levels: {levels} | Modes: {modes} | Total runs: {total_runs}")

    results_list = []

    for q_idx, (_, row) in enumerate(finder_df.iterrows()):
        query        = row.get("query", "")
        truth_answer = row.get("truth_answer", "")
        truth_ref    = row.get("truth_ref", "")
        print(f"\n[Q {q_idx+1}/{len(finder_df)}] {query[:80]}...")

        if enhance_query_flag:
            query = enhance_query(query, gen_model, gen_tokenizer)
            print(f"  [enhance] → {query[:120]}...")

        meta = extract_metadata(query, gen_model, gen_tokenizer)
        print(f"  ticker={meta['ticker']} year={meta['year']} form_type={meta['form_type']}")

        filters = {k: meta[k] for k in ("ticker", "year", "form_type") if meta.get(k)}

        # ── Dense embedding — computed ONCE per query, reused across levels and modes ──
        query_vec = None
        if need_dense:
            formatted_query = f"task: search result | query: {query}"
            query_tokens    = embed_tokenizer.encode(formatted_query, return_tensors="mlx")
            query_vec       = embed_model(query_tokens).text_embeds.tolist()[0]

        for level in levels:
            level_cfg        = COLLECTIONS_2[level] ###########
            coll_name_dense  = level_cfg["coll_name"]
            use_parent_fetch = level_cfg["use_parent_fetch"]
            print(f"\n  ── level: {level} ──")

            # ── Dense retrieval — once per level, reused across modes ──
            dense_results = None
            if need_dense:
                print(f"  [dense] collection='{coll_name_dense}' filters={filters} top_k={prefetch_k}")
                dense_results = search_with_payload(coll_name_dense, query_vec, payload_must=filters, top_k=prefetch_k)
                dense_pts = dense_results.points if hasattr(dense_results, "points") else []
                print(f"  [dense] {len(dense_pts)} hits | scores={[round(p.score, 3) for p in dense_pts]}")

            # ── BM25 sparse retrieval — once per level, reused across modes ──
            # Uses the same enhanced query as dense, but without the Gemma-specific prefix.
            sparse_results = None
            if need_sparse:
                print(f"  [bm25]  collection='{COLL_BM25}' query='{query[:60]}...'")
                sparse_results = search_bm25(COLL_BM25, query, payload_must=filters, top_k=prefetch_k)
                sparse_pts = sparse_results.points if hasattr(sparse_results, "points") else []
                print(f"  [bm25]  {len(sparse_pts)} hits")

            for mode in modes:
                run_id = f"{meta.get('ticker','?')}_{meta.get('year','?')}_{level}_{mode}_{q_idx}"
                print(f"\n    ── mode: {mode} ──")

                context_points = []
                rag_answer     = ""

                try:
                    # ── Build context_points for this mode ──
                    # context points are the points sent to generator
                    if mode == "dense":
                        context_points = (dense_results.points if hasattr(dense_results, "points") else [])[:top_k]
                    elif mode == "sparse":
                        context_points = (sparse_results.points if hasattr(sparse_results, "points") else [])[:top_k]
                    elif mode == "hybrid":
                        context_points = rrf_fuse(dense_results, sparse_results, top_k=top_k)

                    # ── Parent fetch (dense/hybrid only, when configured) ──
                    if use_parent_fetch and mode in {"dense", "hybrid"} and context_points:
                        parent_ids = list({
                            p.payload.get("parent_id")
                            for p in context_points
                            if p.payload.get("parent_id")
                        })
                        if parent_ids:
                            parent_records = client.retrieve(PARENT_COLL, ids=parent_ids, with_payload=True)
                            word_count     = sum(len(r.payload.get("text", "").split()) for r in parent_records)
                            print(f"    [parent_fetch] {len(parent_ids)} child → {len(parent_records)} header (~{word_count} words)")
                            context_points = parent_records
                        else:
                            print("    [parent_fetch] WARNING: no parent_ids found; using retrieved chunks")

                    # ── Generate ──
                    print(f"    [generate] {len(context_points)} chunk(s) → LLM")
                    if not context_points:
                        rag_answer = "No relevant context retrieved."
                    else:
                        wrapped = SimpleNamespace(points=context_points)
                        rag_answer, _ = generate_llm_answer(meta["optimized_query"], wrapped, gen_model, gen_tokenizer)
                        print(f"    [generate] {len(rag_answer)} chars: {rag_answer[:120].strip()}{'...' if len(rag_answer) > 120 else ''}")

                except Exception as e:
                    print(f"    [ERROR] {mode}: {e}")

                rag_ret = "".join(
                    f"======================\nSource Number {p_n}\n {p}"
                    for p_n, p in enumerate(context_points)
                )

                results_list.append({
                    "run_id":        run_id,
                    "level":         level,
                    "mode":          mode,
                    "query":         query,
                    "truth_answer":  truth_answer,
                    "truth_ref":     truth_ref,
                    "rag_answer":    rag_answer,
                    "rag_retrieved": rag_ret,
                })

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    ts         = datetime.now().strftime("%Y%m%d_%H%M")
    results_df = pd.DataFrame(results_list)

    n_empty = (results_df["rag_answer"] == "No relevant context retrieved.").sum()
    print(f"\n── Evaluation complete ──")
    print(f"  Total: {len(results_df)} | Empty answers: {n_empty}")

    results_df.to_csv(f"{output_dir}/eval_{ts}.csv",   index=False)
    results_df.to_pickle(f"{output_dir}/eval_{ts}.pkl")
    results_df.to_json(f"{output_dir}/eval_{ts}.json")

    print(f"  Saved → {output_dir}/eval_{ts}.{{csv,pkl,json}}")
    return results_df


#%%
if __name__ == "__main__":
    tickers   = ["nvda"]#,"wmt","tsla","pypl"]
    finder_df = run_finder(tickers=tickers)
    print("Loading generation model...")
    gen_model, gen_tokenizer = load("mlx-community/Llama-3.2-3B-Instruct-4bit")

    print("Loading embedding model...")
    embed_model, embed_tokenizer = emb_load("mlx-community/embeddinggemma-300m-bf16")
    results = run_evaluation(
        finder_df=finder_df,
        gen_model=gen_model,
        gen_tokenizer=gen_tokenizer,
        embed_model=embed_model,
        embed_tokenizer=embed_tokenizer,
        top_k=6,
        enhance_query_flag=True,
        levels=["header"],
        modes=["hybrid"],  # dense embedding computed once per query
    )
    print(results)

# %%
