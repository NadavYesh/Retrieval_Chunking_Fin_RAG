#%%
import subprocess
from datetime import datetime
from types import SimpleNamespace
import pandas as pd
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))
from search_engine import (
    search_with_payload, search_bm25, rrf_fuse, rrf_fuse_multi, generate_llm_answer,
    search_dense_tiered, search_bm25_tiered,
)
from FinDER import run_finder
from evaluation.section_routing import make_section_filter, route, routing_stats
from evaluation.rag_functions import enhance_query, extract_metadata, embed_query
from db.database import get_qdrant_client
from mlx_lm import load
from mlx_embeddings.utils import load as emb_load

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



def run_evaluation(
    finder_df:          pd.DataFrame,
    gen_model           = None,
    gen_tokenizer       = None,
    embed_model         = None,
    embed_tokenizer     = None,
    output_dir: str     = "/Users/nadavsmacbookair/Desktop/Thesis/data/eval_results",
    top_k:      int     = 5,
    levels:     list    = None,
    retrieval_modes:      list    = None,   # any subset of ["dense", "sparse", "hybrid"]
    enhance_query_flag:  bool = False,
    use_tiered_years:    bool = True,
    use_section_routing: bool = False,
    tickers:    list    = None,   # used for output filename
) -> pd.DataFrame:
    if levels is None:
        levels = list(COLLECTIONS_2.keys())
    if isinstance(levels, str):
        levels = [levels]
    if retrieval_modes is None:
        retrieval_modes = ["dense"]
    if isinstance(retrieval_modes, str):
        retrieval_modes = [retrieval_modes]
    unknown = set(retrieval_modes) - VALID_MODES
    if unknown:
        raise ValueError(f"Unknown retrieval_modes: {unknown}. Valid: {VALID_MODES}")

    if finder_df.empty:
        print("No FinDER questions found.")
        return pd.DataFrame()

    need_dense  = any(m in {"dense",  "hybrid"} for m in retrieval_modes)
    need_sparse = any(m in {"sparse", "hybrid"} for m in retrieval_modes)
    # For hybrid, retrieve wider candidate sets before RRF fusion.
    prefetch_k  = top_k * 3 if "hybrid" in retrieval_modes else top_k ###################

    total_runs = len(finder_df) * len(levels) * len(retrieval_modes)
    print(f"Questions: {len(finder_df)} | Levels: {levels} | Modes: {retrieval_modes} | Total runs: {total_runs}")

    results_list = []

    for q_idx, (_, row) in enumerate(finder_df.iterrows()):
        finder_id    = row.get("_id", "")
        orig_query   = row.get("query", "")
        truth_answer = row.get("truth_answer", "")
        truth_ref    = row.get("truth_ref", "")
        category     = row.get("category", "")
        query_type   = row.get("type", "")
        print(f"\n[Q {q_idx+1}/{len(finder_df)}] {orig_query[:80]}...")
        query         = orig_query
        query_enhanced = None
        if enhance_query_flag:
            query_enhanced = enhance_query(orig_query, gen_model, gen_tokenizer)
            query          = query_enhanced
            print(f"  [enhance] → {query_enhanced[:120]}...")

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

        query_vec = embed_query(query, embed_model, embed_tokenizer) if need_dense else None

        for level in levels:
            level_cfg        = COLLECTIONS_2[level] ###########
            coll_name_dense  = level_cfg["coll_name"]
            use_parent_fetch = level_cfg["use_parent_fetch"]
            print(f"\n  ── level: {level} ──")

            # ── Track B: global dense retrieval (baseline, reused across retrieval_modes) ──
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

            # ── Track B: global BM25 retrieval ──
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

            # ── Track A: section-targeted retrieval (only when routing is on) ──
            sec_dense_results  = None
            sec_sparse_results = None
            if use_section_routing:
                sec_filter = make_section_filter(category)
                if sec_filter:
                    sec_k = max(prefetch_k // 2, top_k)
                    if need_dense:
                        sec_dense_results = search_with_payload(
                            coll_name_dense, query_vec,
                            payload_must=filters, top_k=sec_k,
                            extra_filter=sec_filter,
                        )
                        sec_pts = sec_dense_results.points if hasattr(sec_dense_results, "points") else []
                        print(f"  [dense-section] cat={category!r} {len(sec_pts)} hits")
                    if need_sparse:
                        sec_sparse_results = search_bm25(
                            COLL_BM25, query,
                            payload_must=filters, top_k=sec_k,
                            extra_filter=sec_filter,
                        )
                        sec_spts = sec_sparse_results.points if hasattr(sec_sparse_results, "points") else []
                        print(f"  [bm25-section]  cat={category!r} {len(sec_spts)} hits")

            for retrieval_mode in retrieval_modes:
                run_id = f"{meta.get('ticker','?')}_{meta.get('year','?')}_{level}_{retrieval_mode}_{q_idx}"
                print(f"\n    ── retrieval_mode: {retrieval_mode} ──")

                context_points = []
                rag_answer     = ""

                try:
                    # ── Build candidate pool for this retrieval_mode ──
                    if retrieval_mode == "dense":
                        track_b = (dense_results.points if hasattr(dense_results, "points") else [])
                        if use_section_routing and sec_dense_results is not None:
                            track_a = (sec_dense_results.points if hasattr(sec_dense_results, "points") else [])
                            candidates = rrf_fuse_multi([SimpleNamespace(points=track_a), SimpleNamespace(points=track_b)], top_k=prefetch_k)
                        else:
                            candidates = track_b
                    elif retrieval_mode == "sparse":
                        track_b = (sparse_results.points if hasattr(sparse_results, "points") else [])
                        if use_section_routing and sec_sparse_results is not None:
                            track_a = (sec_sparse_results.points if hasattr(sec_sparse_results, "points") else [])
                            # fuse track A and track B if section routing is used. 
                            # track A is the actual new thing that mapps keyword to sections; track B is the simpler
                            # unrouted approach.
                            # it relates the section specific rounting

                            candidates = rrf_fuse_multi([SimpleNamespace(points=track_a), SimpleNamespace(points=track_b)], top_k=prefetch_k)
                        else:
                            candidates = track_b
                    elif retrieval_mode == "hybrid":
                        if use_section_routing and (sec_dense_results is not None or sec_sparse_results is not None):
                            result_sets = [dense_results, sparse_results]
                            if sec_dense_results is not None:
                                result_sets.append(sec_dense_results)
                            if sec_sparse_results is not None:
                                result_sets.append(sec_sparse_results)
                            # now fuse.
                            candidates = rrf_fuse_multi(result_sets, top_k=prefetch_k)
                        else:
                            candidates = rrf_fuse(dense_results, sparse_results, top_k=prefetch_k)

                    # ── Section routing: re-rank candidate pool by section tier ──
                    if use_section_routing:
                        stats = routing_stats(candidates, category)
                        print(f"    [route] {stats} → taking top {top_k}")
                        context_points = route(candidates, category, top_k)
                    else:
                        context_points = candidates[:top_k]

                    # ── Parent fetch (dense/hybrid only, when configured) ──
                    if use_parent_fetch and retrieval_mode in {"dense", "hybrid"} and context_points:
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
                    print(f"    [ERROR] {retrieval_mode}: {e}")

                rag_ret = "".join(
                    f"======================\nSource Number {p_n}\n {p}"
                    for p_n, p in enumerate(context_points)
                )

                results_list.append({
                    "finder_id":      finder_id,
                    "run_id":         run_id,
                    "level":          level,
                    "retrieval_mode":           retrieval_mode,
                    "query":          orig_query,
                    "query_enhanced": query_enhanced,
                    "category":       category,
                    "query_type":     query_type,
                    "truth_answer":   truth_answer,
                    "truth_ref":      truth_ref,
                    "rag_answer":     rag_answer,
                    "rag_retrieved":  rag_ret,
                })

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    ts         = datetime.now().strftime("%Y%m%d_%H%M")
    results_df = pd.DataFrame(results_list)

    tickers_tag = "-".join(t.upper() for t in sorted(tickers)) if tickers else "ALL"
    levels_tag  = "-".join(levels).upper()
    modes_tag   = "-".join(retrieval_modes).upper()
    enh_tag     = "ENHANCED" if enhance_query_flag else "PLAIN"
    yr_tag      = "TIERED" if use_tiered_years else "FLAT"
    sec_tag     = "ROUTED" if use_section_routing else "UNROUTED"
    tag = f"{tickers_tag}_{levels_tag}_{modes_tag}_{enh_tag}_{yr_tag}_{sec_tag}"

    n_empty = (results_df["rag_answer"] == "No relevant context retrieved.").sum()
    print(f"\n── Evaluation complete ──")
    print(f"  Total: {len(results_df)} | Empty answers: {n_empty}")

    results_df.to_json(f"{output_dir}/{tag}_eval_{ts}.json")

    print(f"  Saved → {output_dir}/{tag}_eval_{ts}.json")
    client.close()
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
      4. generation — not cached (context varies by retrieval_mode/top_k).

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
        any_enhanced        = any(cfg["enhance_query_flag"] for _, cfg in cfg_group)
        any_section_routing = any(cfg.get("use_section_routing", False) for _, cfg in cfg_group)
        union_levels  = sorted({l for _, cfg in cfg_group for l in cfg["levels"]})
        union_modes   = {m for _, cfg in cfg_group for m in cfg["retrieval_modes"]}
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
                (top_k * 3 if "hybrid" in cfg["retrieval_modes"] else top_k for cfg in relevant),
                default=top_k,
            )

        # Initialise per-config result accumulators
        result_lists = {i: [] for i, _ in cfg_group}

        for q_idx, (_, row) in enumerate(finder_df.iterrows()):
            finder_id    = row.get("_id", "")
            orig_query   = row.get("query", "")
            truth_answer = row.get("truth_answer", "")
            truth_ref    = row.get("truth_ref", "")
            category     = row.get("category", "")
            query_type   = row.get("type", "")
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
                query_vec = embed_query(qt, embed_model, embed_tokenizer) if need_dense else None
                query_cache[qt] = {"meta": meta, "vec": query_vec}
                print(f"  [meta/{('enh' if qt == enh_query else 'orig')}] "
                      f"ticker={meta.get('ticker')} year={meta.get('year')}")

            # ── Cache 3: Track B global retrieval per (query_text, level, use_tiered) ──
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
                            "filters":          filters,
                            "query_vec":        query_vec,
                            "pk":               pk,
                        }

            # ── Cache 4: Track A section-targeted retrieval per (query_text, level) ──
            # Category varies per question, so this cache is rebuilt each question.
            # Shared across configs that request section routing.
            section_cache: dict[tuple, dict] = {}
            if any_section_routing:
                sec_filter = make_section_filter(category)
                if sec_filter:
                    for qt, qdata in query_cache.items():
                        meta      = qdata["meta"]
                        query_vec = qdata["vec"]
                        year_val  = meta.get("year")
                        filters   = {k: meta[k] for k in ("ticker", "form_type") if meta.get(k)}
                        if year_val:
                            filters = {**filters, "year": year_val}
                        for level in union_levels:
                            skey = (qt, level)
                            if skey in section_cache:
                                continue
                            level_cfg  = COLLECTIONS_2[level]
                            coll_dense = level_cfg["coll_name"]
                            rkey_ref   = (qt, level, next(iter(union_tiered)))
                            pk         = retrieval_cache.get(rkey_ref, {}).get("pk", top_k)
                            sec_k      = max(pk // 2, top_k)

                            sec_dense  = None
                            sec_sparse = None
                            if need_dense and query_vec is not None:
                                sec_dense = search_with_payload(
                                    coll_dense, query_vec, payload_must=filters,
                                    top_k=sec_k, extra_filter=sec_filter,
                                )
                                pts = sec_dense.points if hasattr(sec_dense, "points") else []
                                print(f"  [dense-section/{level}] cat={category!r} {len(pts)} hits")
                            if need_sparse:
                                sec_sparse = search_bm25(
                                    COLL_BM25, qt, payload_must=filters,
                                    top_k=sec_k, extra_filter=sec_filter,
                                )
                                pts = sec_sparse.points if hasattr(sec_sparse, "points") else []
                                print(f"  [bm25-section/{level}]  cat={category!r} {len(pts)} hits")
                            section_cache[skey] = {"dense": sec_dense, "sparse": sec_sparse}

            # ── Run each config against cached results ────────────────────────
            for cfg_idx, cfg in cfg_group:
                qt      = enh_query if (cfg["enhance_query_flag"] and enh_query) else orig_query
                qdata   = query_cache[qt]
                meta    = qdata["meta"]
                use_sec = cfg.get("use_section_routing", False)

                for level in cfg["levels"]:
                    rkey  = (qt, level, cfg["use_tiered_years"])
                    rc    = retrieval_cache[rkey]
                    dense_results     = rc["dense"]
                    sparse_results    = rc["sparse"]
                    use_parent_fetch  = rc["use_parent_fetch"]
                    pk_cached         = rc["pk"]

                    # Track A section results (if available and requested)
                    skey = (qt, level)
                    sec_dense_r  = section_cache.get(skey, {}).get("dense")  if use_sec else None
                    sec_sparse_r = section_cache.get(skey, {}).get("sparse") if use_sec else None

                    for retrieval_mode in cfg["retrieval_modes"]:
                        run_id = f"{meta.get('ticker','?')}_{meta.get('year','?')}_{level}_{retrieval_mode}_{q_idx}"
                        print(f"    [cfg {cfg_idx}] retrieval_mode={retrieval_mode} enh={cfg['enhance_query_flag']} tiered={cfg['use_tiered_years']} routed={use_sec}")

                        context_points = []
                        rag_answer     = ""
                        try:
                            # Build candidate pool fusing Track A + Track B
                            if retrieval_mode == "dense":
                                track_b = (dense_results.points if hasattr(dense_results, "points") else [])
                                if use_sec and sec_dense_r is not None:
                                    track_a = (sec_dense_r.points if hasattr(sec_dense_r, "points") else [])
                                    candidates = rrf_fuse_multi([SimpleNamespace(points=track_a), SimpleNamespace(points=track_b)], top_k=pk_cached)
                                else:
                                    candidates = track_b
                            elif retrieval_mode == "sparse":
                                track_b = (sparse_results.points if hasattr(sparse_results, "points") else [])
                                if use_sec and sec_sparse_r is not None:
                                    track_a = (sec_sparse_r.points if hasattr(sec_sparse_r, "points") else [])
                                    candidates = rrf_fuse_multi([SimpleNamespace(points=track_a), SimpleNamespace(points=track_b)], top_k=pk_cached)
                                else:
                                    candidates = track_b
                            elif retrieval_mode == "hybrid":
                                if use_sec and (sec_dense_r is not None or sec_sparse_r is not None):
                                    result_sets = [r for r in [dense_results, sparse_results, sec_dense_r, sec_sparse_r] if r is not None]
                                    candidates = rrf_fuse_multi(result_sets, top_k=pk_cached)
                                else:
                                    candidates = rrf_fuse(dense_results, sparse_results, top_k=pk_cached)

                            if use_sec:
                                stats = routing_stats(candidates, category)
                                print(f"    [route] {stats} → taking top {top_k}")
                                context_points = route(candidates, category, top_k)
                            else:
                                context_points = candidates[:top_k]

                            if use_parent_fetch and retrieval_mode in {"dense", "hybrid"} and context_points:
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
                            "finder_id":      finder_id,
                            "run_id":         run_id,
                            "level":          level,
                            "retrieval_mode":           retrieval_mode,
                            "query":          orig_query,
                            "query_enhanced": enh_query if cfg["enhance_query_flag"] else None,
                            "category":       category,
                            "query_type":     query_type,
                            "truth_answer":   truth_answer,
                            "truth_ref":      truth_ref,
                            "rag_answer":     rag_answer,
                            "rag_retrieved":  rag_ret,
                        })

        # ── Save one file per config ──────────────────────────────────────────
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        ts = datetime.now().strftime("%Y%m%d_%H%M")
        for cfg_idx, cfg in cfg_group:
            results_df  = pd.DataFrame(result_lists[cfg_idx])
            tickers_tag = "-".join(t.upper() for t in sorted(tickers))
            levels_tag  = "-".join(cfg["levels"]).upper()
            modes_tag   = "-".join(cfg["retrieval_modes"]).upper()
            enh_tag     = "ENHANCED" if cfg["enhance_query_flag"] else "PLAIN"
            yr_tag      = "TIERED" if cfg["use_tiered_years"] else "FLAT"
            sec_tag     = "ROUTED" if cfg.get("use_section_routing", False) else "UNROUTED"
            tag         = f"{tickers_tag}_{levels_tag}_{modes_tag}_{enh_tag}_{yr_tag}_{sec_tag}"
            n_empty     = (results_df["rag_answer"] == "No relevant context retrieved.").sum()
            print(f"\n  [cfg {cfg_idx}] {tag}: {len(results_df)} rows | {n_empty} empty")
            results_df.to_json(f"{output_dir}/{tag}_eval_{ts}.json")
            print(f"  Saved → {output_dir}/{tag}_eval_{ts}.json")


# def main(
#         tickers,
#         enhance_query_flag=True,
#         levels=["header"],
#         retrieval_modes=["hybrid"],
#         use_tiered_years=True,
# ):
#     tickers   = tickers
#     finder_df = run_finder(tickers=tickers)
#     print("Loading generation model...")
#     gen_model, gen_tokenizer = load("mlx-community/Llama-3.2-3B-Instruct-4bit")

#     print("Loading embedding model...")
#     embed_model, embed_tokenizer = emb_load("mlx-community/embeddinggemma-300m-bf16")
#     results = run_evaluation(
#         finder_df=finder_df,
#         gen_model=gen_model,
#         gen_tokenizer=gen_tokenizer,
#         embed_model=embed_model,
#         embed_tokenizer=embed_tokenizer,
#         top_k=6,
#         enhance_query_flag=enhance_query_flag,
#         levels=levels,
#         retrieval_modes=retrieval_modes,
#         use_tiered_years=use_tiered_years,
#         tickers=tickers,
#     )
#     print(results)

#%%
if __name__ == "__main__":
    # ── Section routing experiment ─────────────────────────────────────────────
    # Each ticker runs two configs back-to-back (UNROUTED vs ROUTED) so results
    # are directly comparable.  Add more tickers by duplicating the pair below.
    configs = [
        # Baseline (best prior config: enhanced + tiered)
        # dict(tickers=["tsla"], enhance_query_flag=True, levels=["header"], retrieval_modes=["hybrid"], use_tiered_years=True,  use_section_routing=False),
        # Section-routed counterpart — same everything, routing on
        dict(tickers=["tsla"], enhance_query_flag=True, levels=["header"], retrieval_modes=["hybrid"], use_tiered_years=True,  use_section_routing=True),
        
        
        # Extend to other tickers (uncomment as needed)
        # dict(tickers=["nvda"], enhance_query_flag=True, levels=["header"], retrieval_modes=["hybrid"], use_tiered_years=True, use_section_routing=False),
        # dict(tickers=["nvda"], enhance_query_flag=True, levels=["header"], retrieval_modes=["hybrid"], use_tiered_years=True, use_section_routing=True),
        # dict(tickers=["wmt"],  enhance_query_flag=True, levels=["header"], retrieval_modes=["hybrid"], use_tiered_years=True, use_section_routing=False),
        # dict(tickers=["wmt"],  enhance_query_flag=True, levels=["header"], retrieval_modes=["hybrid"], use_tiered_years=True, use_section_routing=True),
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
