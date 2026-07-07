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
    search_dense_for_year, search_bm25_for_year,
)
from FinDER import run_finder
from evaluation.section_routing import make_section_filter, section_fuse
from evaluation.build_evaluation_dataset import load_dataset

DATASET_PATH = "/Users/nadavsmacbookair/Desktop/Thesis/data/eval_dataset/limited-new-precomputed-qwen.parquet"
from db.database import get_qdrant_client

# Keyed by the meaningful choice (chunking level 1/2/3), not by the verbose
# Qdrant collection name -- the collection names live only as values here.
COLLECTIONS = {
    1: {
        "coll_name":      "--limited --level 1 DENSE",
        "coll_name_bm25": "--limited --level 1 BM25",
        "use_parent_fetch": False,
    },
    2: {
        "coll_name":      "--limited --level 2 DENSE",
        "coll_name_bm25": "--limited --level 2 BM25",
        "use_parent_fetch": True,
    },
    3: {
        "coll_name":      "--limited --level 3 DENSE",
        "coll_name_bm25": "--limited --level 3 BM25",
        "use_parent_fetch": True,
    },
}
PARENT_COLL_DENSE = COLLECTIONS[1]["coll_name"]
PARENT_COLL_BM25  = COLLECTIONS[1]["coll_name_bm25"]
VALID_MODES = {"dense", "sparse", "hybrid"}

# Full-word retrieval-mode labels for config_key -- spelled out rather than
# abbreviated to a single letter (e.g. "L2_BM25", not "L2B").
_MODE_LABEL = {"dense": "DENSE", "sparse": "BM25", "hybrid": "HYBRID"}


def _short_level(level: int) -> str:
    """1 -> 'L1', 2 -> 'L2', etc."""
    return f"L{level}"


def _lookup(dataset: pd.DataFrame, finder_id: str, ticker: str, enhance_flag: bool) -> dict:
    """
    Return query text, parsed meta dict, and embedding vector from the dataset.

    meta always comes from meta_orig, regardless of enhance_flag: ticker/year/
    form_type must never depend on query enhancement (see extract_metadata in
    rag_functions.py), so enhancement only changes the query text used for
    BM25 and the embedding vector used for dense search, never the metadata
    filters.
    """
    mask = (dataset["finder_id"] == finder_id) & (dataset["ticker"] == ticker)
    hits = dataset[mask]
    if hits.empty:
        raise KeyError(f"Not found in dataset: finder_id={finder_id!r}, ticker={ticker!r}. "
                       "Run build_evaluation_dataset.py first.")
    row = hits.iloc[0]

    meta = json.loads(row["meta_orig"])
    if enhance_flag:
        query = row["enhanced_query"] or row["query"]
        vec   = row["vec_enhanced"]
    else:
        query = row["query"]
        vec   = row["vec_orig"]

    return {
        "query":          query,
        "meta":           meta,
        "vec":            vec,
        "enhanced_query": row["enhanced_query"],
    }



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
    (enhance_query_flag, section_alpha, config_key) so the analysis script can
    group and compare configs without parsing filenames.

    Year filtering has no config toggle: it's a deterministic function of the
    query's own extracted year (see extract_year_deterministic in utils.py
    and search_dense_for_year/search_bm25_for_year in search_engine.py). An
    explicit year (or explicit multi-year list) from the query is applied as
    a flat, unweighted filter; a query with no year signal falls back to the
    default 5-year window with exponential recency decay.
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

        # the union variables contain things that can run together. 
        union_modes  = {m for _, cfg in cfg_group for m in cfg["retrieval_modes"]}
        union_levels = sorted({l for _, cfg in cfg_group for l in cfg["levels"]})
        need_dense   = any(m in {"dense", "hybrid"} for m in union_modes)
        need_sparse  = any(m in {"sparse", "hybrid"} for m in union_modes)
        any_section  = any(cfg.get("section_alpha", 0.0) > 0 for _, cfg in cfg_group) # for section routing

        def _prefetch_k_for(enhance_flag, level):
            relevant = [cfg for _, cfg in cfg_group
                        if cfg["enhance_query_flag"] == enhance_flag
                        and level in cfg["levels"]]
            # if dense of sparse, we use top k. otherwise, multiply by 3
            # why?
            return max(
                (top_k * 3 if "hybrid" in cfg["retrieval_modes"] else top_k for cfg in relevant), 
                default=top_k,
            )

        ticker_str = tickers[0] if len(tickers) == 1 else "multi" # all thesis cases are single.

        for q_idx, (_, row) in enumerate(finder_df.iterrows()):
            finder_id    = row.get("_id", "")
            orig_query   = row.get("query", "")
            truth_answer = row.get("truth_answer", "")
            truth_ref    = row.get("truth_ref", "")
            category     = row.get("category", "")
            query_type   = row.get("type", "")
            print(f"\n[Q {q_idx+1}/{n_q}] {orig_query}")

            # Build per-enhance-flag precomputed lookup (orig + enhanced variants)
            precomp_cache: dict[bool, dict] = {}
            for enh_flag in {cfg["enhance_query_flag"] for _, cfg in cfg_group}:
                precomp_cache[enh_flag] = _lookup(dataset, finder_id, ticker_str, enh_flag)
                pc  = precomp_cache[enh_flag]
                tag = "ENH" if enh_flag else "PLAIN"
                if enh_flag:
                    print(f"  [pre computed enhanced query] → {pc['query'][:160]}...")
                print(f"  [meta/{tag}] ticker={pc['meta'].get('ticker')} year={pc['meta'].get('year')}")

            # Retrieval cache: (query_text, level) → {dense, sparse}
            # section_alpha is intentionally excluded — retrieval is shared across alpha variants
            retrieval_cache: dict[tuple, dict] = {}
            for enh_flag, pc in precomp_cache.items():
                qt        = pc["query"]
                meta      = pc["meta"]
                query_vec = pc["vec"]
                year_val  = meta.get("year")
                base_filt = {k: meta[k] for k in ("ticker", "form_type") if meta.get(k)}

                tag = "ENH" if enh_flag else "PLAIN"
                for level in union_levels:
                    rkey = (qt, level)
                    if rkey in retrieval_cache:
                        continue
                    pk         = _prefetch_k_for(enh_flag, level)
                    coll_dense = COLLECTIONS[level]["coll_name"]
                    coll_bm25  = COLLECTIONS[level]["coll_name_bm25"]

                    dense_r  = None
                    sparse_r = None
                    if need_dense:
                        dense_r = search_dense_for_year(coll_dense, query_vec, base_filt, year_val, pk)
                        print(f"  [dense/{level}] {tag} {len(dense_r.points if hasattr(dense_r,'points') else [])} hits")
                    if need_sparse:
                        sparse_r = search_bm25_for_year(coll_bm25, qt, base_filt, year_val, pk)
                        print(f"  [bm25/{level}]  {tag} {len(sparse_r.points if hasattr(sparse_r,'points') else [])} hits")

                    retrieval_cache[rkey] = {
                        "dense": dense_r, "sparse": sparse_r, "pk": pk,
                        "use_parent_fetch": COLLECTIONS[level]["use_parent_fetch"],
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
                            pk       = retrieval_cache.get((qt, level), {}).get("pk", top_k)
                            sec_k    = max(pk // 2, top_k)
                            coll_dn  = COLLECTIONS[level]["coll_name"]
                            coll_bm  = COLLECTIONS[level]["coll_name_bm25"]
                            sec_d    = search_with_payload(coll_dn, query_vec, payload_must=filters, top_k=sec_k, extra_filter=sec_filter) if need_dense and query_vec is not None else None
                            sec_s    = search_bm25(coll_bm, qt, payload_must=filters, top_k=sec_k, extra_filter=sec_filter) if need_sparse else None
                            section_cache[skey] = {"dense": sec_d, "sparse": sec_s}

            # Per-config generation
            for cfg_idx, cfg in cfg_group:
                enh_flag      = cfg["enhance_query_flag"]
                pc            = precomp_cache[enh_flag]
                qt            = pc["query"]
                meta          = pc["meta"]
                section_alpha = cfg.get("section_alpha", 0.0)
                use_sec       = section_alpha > 0

                for level in cfg["levels"]:
                    rkey             = (qt, level)
                    rc               = retrieval_cache[rkey]
                    # compute results per level
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
                                if use_sec and sec_dense_r is not None:
                                    track_a        = sec_dense_r.points if hasattr(sec_dense_r, "points") else []
                                    context_points = section_fuse(track_a, track_b, section_alpha, top_k)
                                else:
                                    context_points = track_b[:top_k]
                            elif retrieval_mode == "sparse":
                                track_b = sparse_results.points if hasattr(sparse_results, "points") else []
                                if use_sec and sec_sparse_r is not None:
                                    track_a        = sec_sparse_r.points if hasattr(sec_sparse_r, "points") else []
                                    context_points = section_fuse(track_a, track_b, section_alpha, top_k)
                                else:
                                    context_points = track_b[:top_k]
                            elif retrieval_mode == "hybrid":
                                track_b = rrf_fuse(dense_results, sparse_results, top_k=pk_cached)
                                if use_sec:
                                    sec_parts = [r for r in [sec_dense_r, sec_sparse_r] if r is not None]
                                    if sec_parts:
                                        track_a = rrf_fuse_multi(sec_parts, top_k=pk_cached) if len(sec_parts) > 1 else (
                                            sec_parts[0].points if hasattr(sec_parts[0], "points") else []
                                        )
                                        context_points = section_fuse(track_a, track_b, section_alpha, top_k)
                                    else:
                                        context_points = track_b[:top_k]
                                else:
                                    context_points = track_b[:top_k]

                            if use_parent_fetch and context_points:
                                parent_ids = list({p.payload.get("parent_id") for p in context_points if p.payload.get("parent_id")})
                                if parent_ids:
                                    fetch_coll = PARENT_COLL_BM25 if retrieval_mode == "sparse" else PARENT_COLL_DENSE
                                    context_points = client.retrieve(fetch_coll, ids=parent_ids, with_payload=True)

                            if not context_points:
                                rag_answer = "No relevant context retrieved."
                            else:
                                wrapped    = SimpleNamespace(points=context_points)
                                if gen_model:
                                    rag_answer, _ = generate_llm_answer(orig_query, wrapped, gen_model, gen_tokenizer)
                                    print(f"      [gen] {len(rag_answer)} chars: {rag_answer[:160].strip()}...")
                                else:
                                    rag_answer = " NO GENERATED ANSWER "

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
                            "section_alpha":       section_alpha,
                            "config_key":          f"{_short_level(level)}_{_MODE_LABEL[retrieval_mode]}_{'ENH' if enh_flag else 'PLAIN'}_A{section_alpha}",
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
    return out_path


def write_config(
    tickers:         list[str],
    levels:          list[str] = None,
    retrieval_modes: list[str] = None,
    enhance_query_flag = [True, False],
    section_alpha = [0, 0.5 ,1], 
) -> list[dict]:
    """
    Generate all permutations of evaluation configs for the given tickers.

    The boolean flag (enhance_query_flag) and section_alpha [0.0, 0.5, 1.0]
    are fully permuted (2 × 3 = 6 combinations). Each value in `levels` and
    `retrieval_modes` is its own dimension, so the total is:
    len(tickers) × len(levels) × len(retrieval_modes) × 6.

    There is no year-filtering toggle: year handling is a deterministic
    function of the query itself (see extract_year_deterministic in
    utils.py) -- an explicit year in the query is always respected as a flat
    filter, and a query with no year signal always falls back to the
    default 5-year decayed window. Neither is a config choice.

    section_alpha semantics:
      0.0 → no section routing (Track B only)
      0.5 → soft routing (equal-weight blend of Track A and Track B)
      1.0 → hard routing (Track A only)
    """
    if levels is None:
        levels = list(COLLECTIONS.keys())
    if retrieval_modes is None:
        retrieval_modes = ["hybrid", "dense", "sparse"]

    configs = []
    for ticker, level, mode, enhance, alpha in product(
        tickers,
        levels,
        retrieval_modes,
        enhance_query_flag,
        section_alpha,   
    ):
        configs.append({
            "tickers":            [ticker],
            "levels":             [level],
            "retrieval_modes":    [mode],
            "enhance_query_flag": enhance,
            "section_alpha":      alpha,
        })
    return configs


if __name__ == "__main__":
    from mlx_lm import load

    # load parquet FinDER with enhanced queries.
    dataset, _ = load_dataset(DATASET_PATH)
    if dataset.empty:
        raise SystemExit(f"Dataset not found at {DATASET_PATH}. Run build_evaluation_dataset.py first.")

    LEVELS = [1, 2, 3]
    configs = [
        #write_config(tickers=["pypl"], levels=LEVELS, retrieval_modes=["hybrid","sparse"], enhance_query_flag = [True, False], section_alpha = [0 ,1]),
        write_config(tickers=["tsla"], levels=LEVELS, retrieval_modes=["hybrid","sparse"], enhance_query_flag = [True, False], section_alpha = [0 ,1]),
    ]
    print(f"Running {len(configs)} configs...")

    print("Loading generation model...")
    #gen_model, gen_tokenizer = load("mlx-community/Qwen3.5-9B-OptiQ-4bit")
    gen_model, gen_tokenizer = None,None
    for cfg in configs:
        run_multi_evaluation_lazy(
            configs=cfg,
            dataset=dataset,
            gen_model=gen_model,
            gen_tokenizer=gen_tokenizer,
            top_k=5,
        )
