"""
Lazy evaluation runner: uses precomputed query enhancements, metadata, and embeddings
from the dataset built by build_evaluation_dataset.py.

Only retrieval and LLM answer generation run live — all upstream compute is skipped.
Interface mirrors run_rag.py so configs are interchangeable.
"""

import json
from collections import defaultdict
from itertools import product
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
import sys

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.parent))

from search_engine import (
    search_with_payload, search_bm25, rrf_fuse, rrf_fuse_multi, generate_llm_answer,
    search_dense_tiered, search_bm25_tiered,
)
from FinDER import run_finder
from evaluation.section_routing import make_section_filter, route, routing_stats
from evaluation.build_evaluation_dataset import DATASET_PATH, load_dataset
from db.database import get_qdrant_client

COLLECTIONS_2 = {
    "header": {"coll_name": "--level 1 DENSE", "use_parent_fetch": False},
}
PARENT_COLL = "--level 1 DENSE"
COLL_BM25   = "--level 1 BM25"
VALID_MODES = {"dense", "sparse", "hybrid"}


def _lookup(dataset: pd.DataFrame, finder_id: str, ticker: str, enhance_flag: bool) -> dict:
    """Return query text, parsed meta dict, and embedding vector from the dataset."""
    mask = (dataset["finder_id"] == finder_id) & (dataset["ticker"] == ticker)
    hits = dataset[mask]
    if hits.empty:
        raise KeyError(f"Not found in dataset: finder_id={finder_id!r}, ticker={ticker!r}. "
                       "Run build_evaluation_dataset.py first.")
    row = hits.iloc[0]

    if enhance_flag:
        query = row["enhanced_query"] or row["query"]
        meta  = json.loads(row["meta_enhanced"])
        vec   = row["vec_enhanced"]
    else:
        query = row["query"]
        meta  = json.loads(row["meta_orig"])
        vec   = row["vec_orig"]

    return {
        "query":          query,
        "meta":           meta,
        "vec":            vec,
        "enhanced_query": row["enhanced_query"],
    }


def run_evaluation_lazy(
    finder_df:           pd.DataFrame,
    dataset:             pd.DataFrame,
    gen_model,
    gen_tokenizer,
    output_dir:  str     = "/Users/nadavsmacbookair/Desktop/Thesis/data/eval_results",
    top_k:       int     = 5,
    levels:      list    = None,
    retrieval_modes:       list = None,
    enhance_query_flag:  bool  = False,
    use_tiered_years:    bool  = True,
    use_section_routing: bool  = False,
    tickers:     list    = None,
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
        raise ValueError(f"Unknown retrieval_modes: {unknown}")
    if finder_df.empty:
        print("No FinDER questions found.")
        return pd.DataFrame()

    client      = get_qdrant_client()
    need_dense  = any(m in {"dense", "hybrid"} for m in retrieval_modes)
    need_sparse = any(m in {"sparse", "hybrid"} for m in retrieval_modes)
    prefetch_k  = top_k * 3 if "hybrid" in retrieval_modes else top_k

    ticker_str = (tickers[0] if tickers and len(tickers) == 1 else "multi")
    results_list = []

    for q_idx, (_, row) in enumerate(finder_df.iterrows()):
        finder_id    = row.get("_id", "")
        orig_query   = row.get("query", "")
        truth_answer = row.get("truth_answer", "")
        truth_ref    = row.get("truth_ref", "")
        category     = row.get("category", "")
        query_type   = row.get("type", "")
        print(f"\n[Original Query {q_idx+1}/{len(finder_df)}] {orig_query[:80]}...")

        precomp = _lookup(dataset, finder_id, ticker_str, enhance_query_flag)
        query       = precomp["query"]
        meta        = precomp["meta"]
        query_vec   = precomp["vec"]
        enh_display = precomp["enhanced_query"]
        if enhance_query_flag:
            print(f"  [precomp enhance] → {query[:80]}...")
        print(f"  [precomp meta] ticker={meta.get('ticker')} year={meta.get('year')}")

        year_val     = meta.get("year")
        base_filters = {k: meta[k] for k in ("ticker", "form_type") if meta.get(k) is not None}
        filters      = {**base_filters, "year": year_val} if year_val is not None else base_filters
        is_tiered    = use_tiered_years and isinstance(year_val, list) and len(year_val) > 1

        for level in levels:
            level_cfg        = COLLECTIONS_2[level]
            coll_name_dense  = level_cfg["coll_name"]
            use_parent_fetch = level_cfg["use_parent_fetch"]

            dense_results  = None
            sparse_results = None

            if need_dense:
                if is_tiered:
                    dense_results = search_dense_tiered(coll_name_dense, query_vec, base_filters, year_val, prefetch_k)
                else:
                    dense_results = search_with_payload(coll_name_dense, query_vec, payload_must=filters, top_k=prefetch_k)
                pts = dense_results.points if hasattr(dense_results, "points") else []
                print(f"  [dense/{level}] {len(pts)} hits")

            if need_sparse:
                if is_tiered:
                    sparse_results = search_bm25_tiered(COLL_BM25, query, base_filters, year_val, prefetch_k)
                else:
                    sparse_results = search_bm25(COLL_BM25, query, payload_must=filters, top_k=prefetch_k)
                pts = sparse_results.points if hasattr(sparse_results, "points") else []
                print(f"  [bm25/{level}]  {len(pts)} hits")

            sec_dense_results  = None
            sec_sparse_results = None
            if use_section_routing:
                sec_filter = make_section_filter(category)
                if sec_filter:
                    sec_k = max(prefetch_k // 2, top_k)
                    if need_dense:
                        sec_dense_results = search_with_payload(
                            coll_name_dense, query_vec, payload_must=filters,
                            top_k=sec_k, extra_filter=sec_filter,
                        )
                    if need_sparse:
                        sec_sparse_results = search_bm25(
                            COLL_BM25, query, payload_must=filters,
                            top_k=sec_k, extra_filter=sec_filter,
                        )

            for retrieval_mode in retrieval_modes:
                run_id         = f"{meta.get('ticker','?')}_{meta.get('year','?')}_{level}_{retrieval_mode}_{q_idx}"
                context_points = []
                rag_answer     = ""
                try:
                    if retrieval_mode == "dense":
                        track_b = dense_results.points if hasattr(dense_results, "points") else []
                        if use_section_routing and sec_dense_results is not None:
                            track_a    = sec_dense_results.points if hasattr(sec_dense_results, "points") else []
                            candidates = rrf_fuse_multi([SimpleNamespace(points=track_a), SimpleNamespace(points=track_b)], top_k=prefetch_k)
                        else:
                            candidates = track_b
                    elif retrieval_mode == "sparse":
                        track_b = sparse_results.points if hasattr(sparse_results, "points") else []
                        if use_section_routing and sec_sparse_results is not None:
                            track_a    = sec_sparse_results.points if hasattr(sec_sparse_results, "points") else []
                            candidates = rrf_fuse_multi([SimpleNamespace(points=track_a), SimpleNamespace(points=track_b)], top_k=prefetch_k)
                        else:
                            candidates = track_b
                    elif retrieval_mode == "hybrid":
                        if use_section_routing and (sec_dense_results is not None or sec_sparse_results is not None):
                            result_sets = [r for r in [dense_results, sparse_results, sec_dense_results, sec_sparse_results] if r is not None]
                            candidates  = rrf_fuse_multi(result_sets, top_k=prefetch_k)
                        else:
                            candidates = rrf_fuse(dense_results, sparse_results, top_k=prefetch_k)

                    if use_section_routing:
                        context_points = route(candidates, category, top_k)
                    else:
                        context_points = candidates[:top_k]

                    if use_parent_fetch and retrieval_mode in {"dense", "hybrid"} and context_points:
                        parent_ids = list({p.payload.get("parent_id") for p in context_points if p.payload.get("parent_id")})
                        if parent_ids:
                            context_points = client.retrieve(PARENT_COLL, ids=parent_ids, with_payload=True)

                    if not context_points:
                        rag_answer = "No relevant context retrieved."
                    else:
                        wrapped    = SimpleNamespace(points=context_points)
                        rag_answer, _ = generate_llm_answer(meta["optimized_query"], wrapped, gen_model, gen_tokenizer)
                        print(f"    [gen] {len(rag_answer)} chars: {rag_answer[:80].strip()}...")

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
                    "retrieval_mode": retrieval_mode,
                    "query":          orig_query,
                    "query_enhanced": enh_display if enhance_query_flag else None,
                    "category":       category,
                    "query_type":     query_type,
                    "truth_answer":   truth_answer,
                    "truth_ref":      truth_ref,
                    "rag_answer":     rag_answer,
                    "rag_retrieved":  rag_ret,
                })

    client.close()
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    ts         = datetime.now().strftime("%Y%m%d_%H%M")
    results_df = pd.DataFrame(results_list)

    tickers_tag = "-".join(t.upper() for t in sorted(tickers)) if tickers else "ALL"
    levels_tag  = "-".join(levels).upper()
    modes_tag   = "-".join(retrieval_modes).upper()
    enh_tag     = "ENHANCED" if enhance_query_flag else "PLAIN"
    yr_tag      = "TIERED" if use_tiered_years else "FLAT"
    sec_tag     = "ROUTED" if use_section_routing else "UNROUTED"
    tag         = f"LAZY_{tickers_tag}_{levels_tag}_{modes_tag}_{enh_tag}_{yr_tag}_{sec_tag}"

    n_empty = (results_df["rag_answer"] == "No relevant context retrieved.").sum()
    print(f"\n── Lazy evaluation complete ──")
    print(f"  Total: {len(results_df)} | Empty: {n_empty}")
    results_df.to_json(f"{output_dir}/{tag}_eval_{ts}.json")
    print(f"  Saved → {output_dir}/{tag}_eval_{ts}.json")
    return results_df


def run_multi_evaluation_lazy(
    configs:         list[dict],
    dataset:         pd.DataFrame,
    gen_model,
    gen_tokenizer,
    output_dir: str = "/Users/nadavsmacbookair/Desktop/Thesis/data/eval_results",
    top_k:      int = 6,
) -> None:
    """
    Multi-config lazy evaluation. Retrieval is still cached across configs per question
    (identical to run_multi_evaluation), but precompute steps come from the dataset.

    All configs write into a single output JSON. Each row carries explicit config columns
    (enhance_query_flag, use_tiered_years, use_section_routing, config_key) so the
    analysis script can group and compare configs without parsing filenames.
    """
    client  = get_qdrant_client()
    all_rows: list[dict] = []
    ts = datetime.now().strftime("%Y%m%d_%H%M")

    ticker_groups: dict[tuple, list[tuple[int, dict]]] = defaultdict(list)
    for i, cfg in enumerate(configs):
        key = tuple(sorted(cfg["tickers"]))
        ticker_groups[key].append((i, cfg))

    for ticker_tuple, cfg_group in ticker_groups.items():
        tickers   = list(ticker_tuple)
        finder_df = run_finder(tickers=tickers)
        if finder_df.empty:
            print(f"  [skip] no FinDER questions for {tickers}")
            continue

        n_q = len(finder_df)
        print(f"\n{'='*60}\nTicker group: {tickers}  |  {n_q} questions  |  {len(cfg_group)} config(s)\n{'='*60}")

        union_modes  = {m for _, cfg in cfg_group for m in cfg["retrieval_modes"]}
        union_levels = sorted({l for _, cfg in cfg_group for l in cfg["levels"]})
        union_tiered = {cfg["use_tiered_years"] for _, cfg in cfg_group}
        need_dense   = any(m in {"dense", "hybrid"} for m in union_modes)
        need_sparse  = any(m in {"sparse", "hybrid"} for m in union_modes)
        any_section  = any(cfg.get("use_section_routing", False) for _, cfg in cfg_group)

        def _prefetch_k_for(enhance_flag, level, use_tiered):
            relevant = [cfg for _, cfg in cfg_group
                        if cfg["enhance_query_flag"] == enhance_flag
                        and level in cfg["levels"]
                        and cfg["use_tiered_years"] == use_tiered]
            return max(
                (top_k * 3 if "hybrid" in cfg["retrieval_modes"] else top_k for cfg in relevant),
                default=top_k,
            )

        ticker_str = tickers[0] if len(tickers) == 1 else "multi"

        for q_idx, (_, row) in enumerate(finder_df.iterrows()):
            finder_id    = row.get("_id", "")
            orig_query   = row.get("query", "")
            truth_answer = row.get("truth_answer", "")
            truth_ref    = row.get("truth_ref", "")
            category     = row.get("category", "")
            query_type   = row.get("type", "")
            print(f"\n[Q {q_idx+1}/{n_q}] {orig_query[:80]}...")

            # Build per-enhance-flag precomputed lookup (orig + enhanced variants)
            precomp_cache: dict[bool, dict] = {}
            for enh_flag in {cfg["enhance_query_flag"] for _, cfg in cfg_group}:
                # precomp_cache gathers per ticker-config pair the correct queries.
                precomp_cache[enh_flag] = _lookup(dataset, finder_id, ticker_str, enh_flag)
                if enh_flag:
                    print(f"  [precomp enhance] → {precomp_cache[enh_flag]['query'][:80]}...")

            # Retrieval cache: (query_text, level, use_tiered) → {dense, sparse}
            retrieval_cache: dict[tuple, dict] = {}
            for enh_flag, pc in precomp_cache.items():
                qt        = pc["query"]
                meta      = pc["meta"]
                query_vec = pc["vec"]
                year_val  = meta.get("year")
                base_filt = {k: meta[k] for k in ("ticker", "form_type") if meta.get(k)}
                filters   = {**base_filt, "year": year_val} if year_val else base_filt

                for level in union_levels:
                    for use_tiered in union_tiered:
                        rkey = (qt, level, use_tiered)
                        if rkey in retrieval_cache:
                            continue
                        is_tiered  = use_tiered and isinstance(year_val, list) and len(year_val) > 1
                        pk         = _prefetch_k_for(enh_flag, level, use_tiered)
                        coll_dense = COLLECTIONS_2[level]["coll_name"]

                        dense_r  = None
                        sparse_r = None
                        if need_dense:
                            dense_r = (search_dense_tiered(coll_dense, query_vec, base_filt, year_val, pk)
                                       if is_tiered else
                                       search_with_payload(coll_dense, query_vec, payload_must=filters, top_k=pk))
                            print(f"  [dense/{level}] {len(dense_r.points if hasattr(dense_r,'points') else [])} hits")
                        if need_sparse:
                            sparse_r = (search_bm25_tiered(COLL_BM25, qt, base_filt, year_val, pk)
                                        if is_tiered else
                                        search_bm25(COLL_BM25, qt, payload_must=filters, top_k=pk))
                            print(f"  [bm25/{level}]  {len(sparse_r.points if hasattr(sparse_r,'points') else [])} hits")

                        retrieval_cache[rkey] = {
                            "dense": dense_r, "sparse": sparse_r,
                            "is_tiered": is_tiered, "pk": pk,
                            "use_parent_fetch": COLLECTIONS_2[level]["use_parent_fetch"],
                        }

            # Section cache: (query_text, level) → {dense, sparse}
            section_cache: dict[tuple, dict] = {}
            if any_section:
                sec_filter = make_section_filter(category)
                if sec_filter:
                    for enh_flag, pc in precomp_cache.items():
                        qt        = pc["query"]
                        meta      = pc["meta"]
                        query_vec = pc["vec"]
                        year_val  = meta.get("year")
                        filters   = {k: meta[k] for k in ("ticker", "form_type") if meta.get(k)}
                        if year_val:
                            filters = {**filters, "year": year_val}
                        for level in union_levels:
                            skey = (qt, level)
                            if skey in section_cache:
                                continue
                            rkey_ref = (qt, level, next(iter(union_tiered)))
                            pk       = retrieval_cache.get(rkey_ref, {}).get("pk", top_k)
                            sec_k    = max(pk // 2, top_k)
                            coll_dn  = COLLECTIONS_2[level]["coll_name"]
                            sec_d    = search_with_payload(coll_dn, query_vec, payload_must=filters, top_k=sec_k, extra_filter=sec_filter) if need_dense and query_vec is not None else None
                            sec_s    = search_bm25(COLL_BM25, qt, payload_must=filters, top_k=sec_k, extra_filter=sec_filter) if need_sparse else None
                            section_cache[skey] = {"dense": sec_d, "sparse": sec_s}

            # Per-config generation
            for cfg_idx, cfg in cfg_group:
                enh_flag = cfg["enhance_query_flag"]
                pc       = precomp_cache[enh_flag]
                qt       = pc["query"]
                meta     = pc["meta"]
                use_sec  = cfg.get("use_section_routing", False)

                for level in cfg["levels"]:
                    rkey             = (qt, level, cfg["use_tiered_years"])
                    rc               = retrieval_cache[rkey]
                    dense_results    = rc["dense"]
                    sparse_results   = rc["sparse"]
                    use_parent_fetch = rc["use_parent_fetch"]
                    pk_cached        = rc["pk"]
                    skey             = (qt, level)
                    sec_dense_r      = section_cache.get(skey, {}).get("dense")  if use_sec else None
                    sec_sparse_r     = section_cache.get(skey, {}).get("sparse") if use_sec else None

                    for retrieval_mode in cfg["retrieval_modes"]:
                        run_id         = f"{meta.get('ticker','?')}_{meta.get('year','?')}_{level}_{retrieval_mode}_{q_idx}"
                        context_points = []
                        rag_answer     = ""
                        try:
                            if retrieval_mode == "dense":
                                track_b = dense_results.points if hasattr(dense_results, "points") else []
                                candidates = rrf_fuse_multi([SimpleNamespace(points=(sec_dense_r.points if hasattr(sec_dense_r,"points") else [])), SimpleNamespace(points=track_b)], top_k=pk_cached) if use_sec and sec_dense_r else track_b
                            elif retrieval_mode == "sparse":
                                track_b = sparse_results.points if hasattr(sparse_results, "points") else []
                                candidates = rrf_fuse_multi([SimpleNamespace(points=(sec_sparse_r.points if hasattr(sec_sparse_r,"points") else [])), SimpleNamespace(points=track_b)], top_k=pk_cached) if use_sec and sec_sparse_r else track_b
                            elif retrieval_mode == "hybrid":
                                result_sets = [r for r in [dense_results, sparse_results, sec_dense_r, sec_sparse_r] if r is not None] if use_sec and (sec_dense_r or sec_sparse_r) else None
                                candidates  = rrf_fuse_multi(result_sets, top_k=pk_cached) if result_sets else rrf_fuse(dense_results, sparse_results, top_k=pk_cached)

                            context_points = route(candidates, category, top_k) if use_sec else candidates[:top_k]

                            if use_parent_fetch and retrieval_mode in {"dense", "hybrid"} and context_points:
                                parent_ids = list({p.payload.get("parent_id") for p in context_points if p.payload.get("parent_id")})
                                if parent_ids:
                                    context_points = client.retrieve(PARENT_COLL, ids=parent_ids, with_payload=True)

                            if not context_points:
                                rag_answer = "No relevant context retrieved."
                            else:
                                wrapped    = SimpleNamespace(points=context_points)
                                rag_answer, _ = generate_llm_answer(meta["optimized_query"], wrapped, gen_model, gen_tokenizer)
                                print(f"      [gen] {len(rag_answer)} chars: {rag_answer[:80].strip()}...")

                        except Exception as e:
                            print(f"      [ERROR] {e}")

                        rag_ret = "".join(f"======================\nSource Number {p_n}\n {p}" for p_n, p in enumerate(context_points))
                        all_rows.append({
                            "finder_id":           finder_id,
                            "run_id":              run_id,
                            "level":               level,
                            "retrieval_mode":      retrieval_mode,
                            "ticker_filter":       ticker_str,
                            "enhance_query_flag":  enh_flag,
                            "use_tiered_years":    cfg["use_tiered_years"],
                            "use_section_routing": use_sec,
                            "config_key":          f"{level}_{retrieval_mode}_{'ENH' if enh_flag else 'PLAIN'}_{'TIERED' if cfg['use_tiered_years'] else 'FLAT'}_{'ROUTED' if use_sec else 'UNROUTED'}",
                            "query":               orig_query,
                            "query_enhanced":      pc["enhanced_query"] if enh_flag else None,
                            "category":            category,
                            "query_type":          query_type,
                            "truth_answer":        truth_answer,
                            "truth_ref":           truth_ref,
                            "rag_answer":          rag_answer,
                            "rag_retrieved":       rag_ret,
                        })

    client.close()

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    tickers_all = sorted({cfg["tickers"][0] for cfg in configs})
    tag         = "-".join(t.upper() for t in tickers_all)
    out_path    = f"{output_dir}/eval_multi_{tag}_{ts}.json"
    results_df  = pd.DataFrame(all_rows)
    results_df.to_json(out_path)
    n_empty = (results_df["rag_answer"] == "No relevant context retrieved.").sum()
    print(f"\n── Multi eval complete ──")
    print(f"  Total rows: {len(all_rows)} | Empty: {n_empty} | Configs: {len(configs)}")
    print(f"  Saved → {out_path}")


def write_config(
    tickers:         list[str],
    levels:          list[str] = ["header"],
    retrieval_modes: list[str] = ["hybrid","dense","sparse"],
) -> list[dict]:
    """
    Generate all permutations of evaluation configs for the given tickers.

    Boolean flags (enhance_query_flag, use_tiered_years, use_section_routing) are
    always fully permuted (2^3 = 8 combinations). Each value in `levels` and
    `retrieval_modes` is treated as its own dimension, so the total number of
    configs is: len(tickers) × len(levels) × len(retrieval_modes) × 8.
    """
    configs = []
    for ticker, level, mode, enhance, tiered, routed in product(
        tickers,
        levels,
        retrieval_modes,
        [True, False],   # enhance_query_flag
        [True, False],   # use_tiered_years
        [True, False],   # use_section_routing
    ):
        configs.append({
            "tickers":            [ticker],
            "levels":             [level],
            "retrieval_modes":    [mode],
            "enhance_query_flag": enhance,
            "use_tiered_years":   tiered,
            "use_section_routing": routed,
        })
    return configs






if __name__ == "__main__":
    from mlx_lm import load

    dataset, _ = load_dataset(DATASET_PATH)
    if dataset.empty:
        raise SystemExit(f"Dataset not found at {DATASET_PATH}. Run build_evaluation_dataset.py first.")

    configs = [
        write_config(
        tickers=["pypl"],
        levels=["header","child","enriched"],
        retrieval_modes=["hybrid","dense","sparse"],
    ),
        write_config(
        tickers=["nvda"],
        levels=["header","child","enriched"],
        retrieval_modes=["hybrid","dense","sparse"],        
    )   
    ]
    print(f"Running {len(configs)} configs...")

    print("Loading generation model...")
    gen_model, gen_tokenizer = load("mlx-community/Qwen3.5-9B-OptiQ-4bit") #changed to Qwen 9B

    for cfg in configs:
        run_multi_evaluation_lazy(
            configs=cfg,
            dataset=dataset,
            gen_model=gen_model,
            gen_tokenizer=gen_tokenizer,
            top_k=5,
        )
