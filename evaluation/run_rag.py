#%%
import gc
import re
import subprocess
from datetime import datetime
from types import SimpleNamespace
import pandas as pd
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))
from search_engine import (
    search_with_payload, search_bm25, rrf_fuse, generate_llm_answer,
    search_dense_tiered, search_bm25_tiered,
)
from prompts import META_EXTRACT_PROMPT, QUERY_ENHANCEMENT_PROMPT
from FinDER import run_finder
from db.database import get_qdrant_client
from mlx_lm import generate, load
import mlx.core as mx
from mlx_embeddings.utils import load as emb_load
from utils import parse_metadata_response, sanitize_year_extraction, extract_ticker_hint

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
    ticker_hint = extract_ticker_hint(query)

    user_content = query
    if ticker_hint:
        user_content = f"{query}\n[Ticker hint: {ticker_hint.upper()}]"

    messages = [
        {"role": "system", "content": META_EXTRACT_PROMPT},
        {"role": "user",   "content": user_content},
    ]
    prompt   = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    response = generate(model, tokenizer, prompt=prompt, verbose=False)
    gc.collect() # garbage collector takes all processes that are taking memory but arent in use any more. and clears them
    mx.clear_cache()
    meta     = parse_metadata_response(response, fallback_query=query)

    # Sanitize year: convert strings/shorthands to integer(s)
    clean_year = sanitize_year_extraction(meta.get("year"), query=query)
    meta["year"] = clean_year  # int, list[int], or None

    # Fallback: use regex-extracted ticker if LLM returned nothing
    if not meta.get("ticker") and ticker_hint: # if any is falsey
        meta["ticker"] = ticker_hint

    return meta


_REFUSAL_RE = re.compile(
    r"^(i (can'?t|cannot|am unable|won'?t)|sorry[,. ]|i'?m sorry)",
    re.IGNORECASE,
)

def _refusal_fallback(raw: str, original: str) -> str:
    """Return the usable rewrite; fall back to original if the model refused."""
    # זה יעזור לי להבין למה תמיד קורה הפולבאק.
    if not _REFUSAL_RE.match(raw.strip()):
        return raw.strip()
    # Salvage: take whatever follows a transition phrase like "as requested" / "however"
    parts = re.split(r"(?:as requested[,.]?|however[,.]?)\s*", raw, flags=re.IGNORECASE)
    candidate = parts[-1].strip() if len(parts) > 1 else ""
    orig_tokens = set(original.lower().split())
    if candidate and any(tok in candidate.lower() for tok in orig_tokens):
        return candidate
    print(f"  [enhance] refusal detected — passing original query through")
    return original


MAX_ENHANCE_RETRIES = 3

def enhance_query(query: str, model, tokenizer) -> str:
    prompt    = QUERY_ENHANCEMENT_PROMPT.format(query=query)
    messages  = [{"role": "user", "content": prompt}]
    formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    for attempt in range(1, MAX_ENHANCE_RETRIES + 1):
        raw = generate(model, tokenizer, prompt=formatted, verbose=False, max_tokens=300)
        gc.collect()
        mx.clear_cache()
        print(f"  [enhance attempt {attempt}] {len(raw)} chars: {repr(raw[:80])}")
        if raw.strip():
            return _refusal_fallback(raw, query)
        print(f"  [enhance attempt {attempt}] empty output — retrying")

    print(f"  [ERROR] enhance_query: empty output after {MAX_ENHANCE_RETRIES} attempts — falling back to original query")
    return query


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
    enhance_query_flag:  bool = False,
    use_tiered_years:    bool = True,
    tickers:    list    = None,   # used for output filename
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
        finder_id    = row.get("_id", "")
        query        = row.get("query", "")
        truth_answer = row.get("truth_answer", "")
        truth_ref    = row.get("truth_ref", "")
        print(f"\n[Q {q_idx+1}/{len(finder_df)}] {query[:80]}...")
        if enhance_query_flag:
            query = enhance_query(query, gen_model, gen_tokenizer)
            print(f"  [enhance] → {query[:120]}...")

        meta = extract_metadata(query, gen_model, gen_tokenizer)
        print(f"  ticker={meta['ticker']} year={meta['year']} form_type={meta['form_type']}")

        year_val   = meta.get("year")
        # base_filters: ticker + form_type only — year handled separately for tiered retrieval
        base_filters = {k: meta[k] for k in ("ticker", "form_type") if meta.get(k) is not None}
        filters      = {**base_filters, "year": year_val} if year_val is not None else base_filters
        # tiered = multi-year list AND the flag is on
        is_tiered  = use_tiered_years and isinstance(year_val, list) and len(year_val) > 1
        if is_tiered:
            print(f"  [tiered] years={year_val} — using proportional top_k per year")

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
                if is_tiered:
                    print(f"  [dense-tiered] collection='{coll_name_dense}' years={year_val} top_k={prefetch_k}")
                    dense_results = search_dense_tiered(coll_name_dense, query_vec, base_filters, year_val, prefetch_k)
                else:
                    print(f"  [dense] collection='{coll_name_dense}' filters={filters} top_k={prefetch_k}")
                    dense_results = search_with_payload(coll_name_dense, query_vec, payload_must=filters, top_k=prefetch_k)
                dense_pts = dense_results.points if hasattr(dense_results, "points") else []
                print(f"  [dense] {len(dense_pts)} hits | scores={[round(p.score, 3) for p in dense_pts]}")

            # ── BM25 sparse retrieval — once per level, reused across modes ──
            # Uses the same enhanced query as dense, but without the Gemma-specific prefix.
            sparse_results = None
            if need_sparse:
                if is_tiered:
                    print(f"  [bm25-tiered]  collection='{COLL_BM25}' years={year_val} query='{query[:50]}...'")
                    sparse_results = search_bm25_tiered(COLL_BM25, query, base_filters, year_val, prefetch_k)
                else:
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
                    "finder_id":     finder_id,
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

    tickers_tag = "-".join(t.upper() for t in sorted(tickers)) if tickers else "ALL"
    levels_tag  = "-".join(levels).upper()
    modes_tag   = "-".join(modes).upper()
    enh_tag     = "ENHANCED" if enhance_query_flag else "PLAIN"
    yr_tag      = "TIERED" if use_tiered_years else "FLAT"
    tag = f"{tickers_tag}_{levels_tag}_{modes_tag}_{enh_tag}_{yr_tag}"

    n_empty = (results_df["rag_answer"] == "No relevant context retrieved.").sum()
    print(f"\n── Evaluation complete ──")
    print(f"  Total: {len(results_df)} | Empty answers: {n_empty}")

    results_df.to_csv(f"{output_dir}/{tag}_eval_{ts}.csv",   index=False)
    results_df.to_pickle(f"{output_dir}/{tag}_eval_{ts}.pkl")
    results_df.to_json(f"{output_dir}/{tag}_eval_{ts}.json")

    print(f"  Saved → {output_dir}/{tag}_eval_{ts}.{{csv,pkl,json}}")
    return results_df


def run_multi_evaluation(
    configs:         list[dict],
    gen_model,
    gen_tokenizer,
    embed_model,
    embed_tokenizer,
    output_dir: str = "/Users/nadavsmacbookair/Desktop/Thesis/data/eval_results",
    top_k:      int = 6,
) -> None:
    """
    Run multiple evaluation configs, reusing expensive LLM/embedding/retrieval calls
    wherever the results would be identical across configs.

    Caching hierarchy (per question, within a ticker group):
      1. enhanced_query  — one LLM call per question, shared by all enhance=True configs.
      2. metadata + embedding — one call per unique query_text (original or enhanced).
      3. dense + BM25 retrieval — one search per (query_text, level, use_tiered_years),
                                   using the max prefetch_k needed by any sharing config.
      4. generation — not cached (context varies by mode/top_k).

    Configs are grouped by ticker set so finder_df is also loaded only once per group.
    Each config produces its own output file with an auto-generated filename.
    """
    from collections import defaultdict

    # ── Group configs by sorted ticker tuple ──────────────────────────────────
    ticker_groups: dict[tuple, list[tuple[int, dict]]] = defaultdict(list)
    for i, cfg in enumerate(configs):
        key = tuple(sorted(cfg["tickers"]))
        ticker_groups[key].append((i, cfg))

    for ticker_tuple, cfg_group in ticker_groups.items():
        tickers    = list(ticker_tuple)
        finder_df  = run_finder(tickers=tickers)
        if finder_df.empty:
            print(f"  [skip] no FinDER questions for {tickers}")
            continue

        n_q = len(finder_df)
        print(f"\n{'='*60}\nTicker group: {tickers}  |  {n_q} questions  |  {len(cfg_group)} config(s)\n{'='*60}")

        # Determine the union of needs across all configs in this group
        any_enhanced  = any(cfg["enhance_query_flag"] for _, cfg in cfg_group)
        union_levels  = sorted({l for _, cfg in cfg_group for l in cfg["levels"]})
        union_modes   = {m for _, cfg in cfg_group for m in cfg["modes"]}
        union_tiered  = {cfg["use_tiered_years"] for _, cfg in cfg_group}
        need_dense    = any(m in {"dense", "hybrid"} for m in union_modes)
        need_sparse   = any(m in {"sparse", "hybrid"} for m in union_modes)

        # Pre-compute max prefetch_k per (query_text_kind, level, use_tiered) key.
        # "query_text_kind" is "enhanced" or "plain" — proxy for actual query text.
        def _prefetch_k_for(enhance_flag, level, use_tiered):
            relevant = [
                cfg for _, cfg in cfg_group
                if cfg["enhance_query_flag"] == enhance_flag
                and level in cfg["levels"]
                and cfg["use_tiered_years"] == use_tiered
            ]
            return max(
                (top_k * 3 if "hybrid" in cfg["modes"] else top_k for cfg in relevant),
                default=top_k,
            )

        # Initialise per-config result accumulators
        result_lists = {i: [] for i, _ in cfg_group}

        for q_idx, (_, row) in enumerate(finder_df.iterrows()):
            finder_id    = row.get("_id", "")
            orig_query   = row.get("query", "")
            truth_answer = row.get("truth_answer", "")
            truth_ref    = row.get("truth_ref", "")
            print(f"\n[Q {q_idx+1}/{n_q}] {orig_query[:80]}...")

            # ── Cache 1: enhanced query (one LLM call if any config needs it) ──
            enh_query = None
            if any_enhanced:
                enh_query = enhance_query(orig_query, gen_model, gen_tokenizer)
                print(f"  [enhance] → {enh_query[:100]}...")

            # ── Cache 2: metadata + embedding per query_text ──────────────────
            # query_texts we actually need depend on which enhance flags are in use
            needed_texts: set[str] = {orig_query}
            if enh_query:
                needed_texts.add(enh_query)

            query_cache: dict[str, dict] = {}
            for qt in needed_texts:
                meta       = extract_metadata(qt, gen_model, gen_tokenizer)
                query_vec  = None
                if need_dense:
                    tokens    = embed_tokenizer.encode(
                        f"task: search result | query: {qt}", return_tensors="mlx"
                    )
                    query_vec = embed_model(tokens).text_embeds.tolist()[0]
                query_cache[qt] = {"meta": meta, "vec": query_vec}
                print(f"  [meta/{('enh' if qt == enh_query else 'orig')}] "
                      f"ticker={meta.get('ticker')} year={meta.get('year')}")

            # ── Cache 3: retrieval per (query_text, level, use_tiered) ─────────
            retrieval_cache: dict[tuple, dict] = {}

            for qt, qdata in query_cache.items():
                meta        = qdata["meta"]
                query_vec   = qdata["vec"]
                year_val    = meta.get("year")
                base_filters = {k: meta[k] for k in ("ticker", "form_type") if meta.get(k)}
                filters      = {**base_filters, "year": year_val} if year_val else base_filters
                enh_flag     = (qt == enh_query) if enh_query else False

                for level in union_levels:
                    for use_tiered in union_tiered:
                        rkey = (qt, level, use_tiered)
                        if rkey in retrieval_cache:
                            continue

                        is_tiered  = use_tiered and isinstance(year_val, list) and len(year_val) > 1
                        pk         = _prefetch_k_for(enh_flag, level, use_tiered)
                        level_cfg  = COLLECTIONS_2[level]
                        coll_dense = level_cfg["coll_name"]

                        dense_results = None
                        if need_dense:
                            if is_tiered:
                                dense_results = search_dense_tiered(coll_dense, query_vec, base_filters, year_val, pk)
                            else:
                                dense_results = search_with_payload(coll_dense, query_vec, payload_must=filters, top_k=pk)
                            pts = dense_results.points if hasattr(dense_results, "points") else []
                            print(f"  [dense/{level}/tiered={is_tiered}] {len(pts)} hits")

                        sparse_results = None
                        if need_sparse:
                            if is_tiered:
                                sparse_results = search_bm25_tiered(COLL_BM25, qt, base_filters, year_val, pk)
                            else:
                                sparse_results = search_bm25(COLL_BM25, qt, payload_must=filters, top_k=pk)
                            pts = sparse_results.points if hasattr(sparse_results, "points") else []
                            print(f"  [bm25/{level}/tiered={is_tiered}]  {len(pts)} hits")

                        retrieval_cache[rkey] = {
                            "dense":            dense_results,
                            "sparse":           sparse_results,
                            "is_tiered":        is_tiered,
                            "use_parent_fetch": level_cfg["use_parent_fetch"],
                            "meta":             meta,
                        }

            # ── Run each config against cached results ────────────────────────
            for cfg_idx, cfg in cfg_group:
                qt      = enh_query if (cfg["enhance_query_flag"] and enh_query) else orig_query
                qdata   = query_cache[qt]
                meta    = qdata["meta"]

                for level in cfg["levels"]:
                    rkey  = (qt, level, cfg["use_tiered_years"])
                    rc    = retrieval_cache[rkey]
                    dense_results     = rc["dense"]
                    sparse_results    = rc["sparse"]
                    use_parent_fetch  = rc["use_parent_fetch"]

                    for mode in cfg["modes"]:
                        run_id = f"{meta.get('ticker','?')}_{meta.get('year','?')}_{level}_{mode}_{q_idx}"
                        print(f"    [cfg {cfg_idx}] mode={mode} enh={cfg['enhance_query_flag']} tiered={cfg['use_tiered_years']}")

                        context_points = []
                        rag_answer     = ""
                        try:
                            if mode == "dense":
                                context_points = (dense_results.points if hasattr(dense_results, "points") else [])[:top_k]
                            elif mode == "sparse":
                                context_points = (sparse_results.points if hasattr(sparse_results, "points") else [])[:top_k]
                            elif mode == "hybrid":
                                context_points = rrf_fuse(dense_results, sparse_results, top_k=top_k)

                            if use_parent_fetch and mode in {"dense", "hybrid"} and context_points:
                                parent_ids = list({
                                    p.payload.get("parent_id")
                                    for p in context_points
                                    if p.payload.get("parent_id")
                                })
                                if parent_ids:
                                    parent_records = client.retrieve(PARENT_COLL, ids=parent_ids, with_payload=True)
                                    context_points = parent_records

                            if not context_points:
                                rag_answer = "No relevant context retrieved."
                            else:
                                wrapped = SimpleNamespace(points=context_points)
                                rag_answer, _ = generate_llm_answer(
                                    meta["optimized_query"], wrapped, gen_model, gen_tokenizer
                                )
                                print(f"      [gen] {len(rag_answer)} chars: {rag_answer[:80].strip()}...")

                        except Exception as e:
                            print(f"      [ERROR] {e}")

                        rag_ret = "".join(
                            f"======================\nSource Number {p_n}\n {p}"
                            for p_n, p in enumerate(context_points)
                        )
                        result_lists[cfg_idx].append({
                            "finder_id":     finder_id,
                            "run_id":        run_id,
                            "level":         level,
                            "mode":          mode,
                            "query":         qt,
                            "truth_answer":  truth_answer,
                            "truth_ref":     truth_ref,
                            "rag_answer":    rag_answer,
                            "rag_retrieved": rag_ret,
                        })

        # ── Save one file per config ──────────────────────────────────────────
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M")
        for cfg_idx, cfg in cfg_group:
            results_df  = pd.DataFrame(result_lists[cfg_idx])
            tickers_tag = "-".join(t.upper() for t in sorted(tickers))
            levels_tag  = "-".join(cfg["levels"]).upper()
            modes_tag   = "-".join(cfg["modes"]).upper()
            enh_tag     = "ENHANCED" if cfg["enhance_query_flag"] else "PLAIN"
            yr_tag      = "TIERED" if cfg["use_tiered_years"] else "FLAT"
            tag         = f"{tickers_tag}_{levels_tag}_{modes_tag}_{enh_tag}_{yr_tag}"
            n_empty     = (results_df["rag_answer"] == "No relevant context retrieved.").sum()
            print(f"\n  [cfg {cfg_idx}] {tag}: {len(results_df)} rows | {n_empty} empty")
            results_df.to_csv(f"{output_dir}/{tag}_eval_{ts}.csv",   index=False)
            results_df.to_pickle(f"{output_dir}/{tag}_eval_{ts}.pkl")
            results_df.to_json(f"{output_dir}/{tag}_eval_{ts}.json")
            print(f"  Saved → {output_dir}/{tag}_eval_{ts}.{{csv,pkl,json}}")


def main(
        tickers,
        enhance_query_flag=True,
        levels=["header"],
        modes=["hybrid"],
        use_tiered_years=True,
):
    tickers   = tickers
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
        enhance_query_flag=enhance_query_flag,
        levels=levels,
        modes=modes,
        use_tiered_years=use_tiered_years,
        tickers=tickers,
    )
    print(results)

#%%
if __name__ == "__main__":
    configs = [
        dict(tickers=["tsla"], enhance_query_flag=True, levels=["header"], modes=["hybrid"], use_tiered_years=False),
        dict(tickers=["tsla"], enhance_query_flag=True, levels=["header"], modes=["hybrid"], use_tiered_years=True),
        dict(tickers=["tsla"], enhance_query_flag=False, levels=["header"], modes=["hybrid"], use_tiered_years=True),
        dict(tickers=["tsla"], enhance_query_flag=False, levels=["header"], modes=["hybrid"], use_tiered_years=False),
        # dict(tickers=["pypl"], enhance_query_flag=True,  levels=["header"], modes=["hybrid"], use_tiered_years=True),
        # dict(tickers=["nvda"], enhance_query_flag=True,  levels=["header"], modes=["hybrid"], use_tiered_years=True),
    ]

    # Load models once — reloading per run would add ~2 min overhead each iteration
    print("Loading generation model...")
    gen_model, gen_tokenizer = load("mlx-community/Llama-3.2-3B-Instruct-4bit")
    print("Loading embedding model...")
    embed_model, embed_tokenizer = emb_load("mlx-community/embeddinggemma-300m-bf16")

    run_multi_evaluation(
        configs=configs,
        gen_model=gen_model,
        gen_tokenizer=gen_tokenizer,
        embed_model=embed_model,
        embed_tokenizer=embed_tokenizer,
        top_k=6,
    )
