import pandas as pd
from qdrant_client import models
from qdrant_client.models import PointStruct
import uuid
import time
from datetime import datetime
from mlx_lm import load, generate
from db.database import get_qdrant_client
from prompts import RAG_ANSWER_PROMPT
from utils import year_weights, _default_year_window



#%%
client = get_qdrant_client()

from qdrant_client import models

def _build_filter(payload_must=None, payload_must_not=None, payload_should=None):
    """Build a qdrant Filter from payload dicts. Returns None if no conditions."""
    def _conditions(mapping):
        conds = []
        if not mapping:
            return conds
        items = mapping.items() if isinstance(mapping, dict) else mapping
        for k, v in items:
            if v is None:
                continue
            if k in ["fiscal_year_end", "year"]:
                year_vals = v if isinstance(v, list) else [v]
                year_conds = []
                for val in year_vals:
                    if val is None:
                        continue
                    try:
                        if hasattr(val, "year"):
                            yr = val.year
                        elif isinstance(val, str):
                            yr = int(val[:4])
                        else:
                            yr = int(val)
                        year_conds.append(
                            models.FieldCondition(
                                key="fiscal_year_end",
                                range=models.DatetimeRange(
                                    gte=f"{yr}-01-01T00:00:00Z",
                                    lte=f"{yr}-12-31T23:59:59Z",
                                ),
                            )
                        )
                    except (ValueError, TypeError) as e:
                        print(f"Warning: Could not parse year from {val}: {e}")
                if len(year_conds) == 1:
                    conds.append(year_conds[0])
                elif len(year_conds) > 1:
                    conds.append(models.Filter(should=year_conds))
            else:
                if isinstance(v, list):
                    conds.append(models.FieldCondition(key=k, match=models.MatchAny(any=v)))
                else:
                    conds.append(models.FieldCondition(key=k, match=models.MatchValue(value=v)))
        return conds

    must     = _conditions(payload_must)
    must_not = _conditions(payload_must_not)
    should   = _conditions(payload_should)

    if must or must_not or should:
        return models.Filter(
            must=must or None,
            must_not=must_not or None,
            should=should or None,
        )
    return None


def search_with_payload(coll_name, query_vec, payload_must=None, payload_must_not=None, payload_should=None, top_k=5, extra_filter=None):
    """Dense vector search with optional payload filtering.

    extra_filter : a pre-built qdrant_models.Filter that is AND-ed with the
                   payload filter (used for section-targeted Track A retrieval).
    """
    query_filter = _build_filter(payload_must, payload_must_not, payload_should)
    if extra_filter is not None:
        if query_filter is not None:
            query_filter = models.Filter(must=[query_filter, extra_filter])
        else:
            query_filter = extra_filter
    results_ = client.query_points(
        collection_name=coll_name,
        query=query_vec,
        limit=top_k,
        with_payload=True,
        query_filter=query_filter,
        search_params=models.SearchParams(hnsw_ef=128, exact=False),
    )
    return results_


def search_bm25(coll_name_sparse, query_text, payload_must=None, top_k=5, extra_filter=None):
    """BM25 sparse search against a collection ingested with qdrant/bm25 model.

    extra_filter : a pre-built qdrant_models.Filter AND-ed with the payload filter.
    """
    query_filter = _build_filter(payload_must)
    if extra_filter is not None:
        if query_filter is not None:
            query_filter = models.Filter(must=[query_filter, extra_filter])
        else:
            query_filter = extra_filter
    results_ = client.query_points(
        collection_name=coll_name_sparse,
        query=models.Document(text=query_text.lower(), model="qdrant/bm25"),
        using="bm25",
        limit=top_k,
        with_payload=True,
        query_filter=query_filter,
    )
    return results_


def rrf_fuse(dense_results, sparse_results, k: int = 60, top_k: int = 6) -> list:
    """
    Reciprocal Rank Fusion of dense and sparse result sets.
    Returns a plain list of ScoredPoints (up to top_k), sorted by RRF score descending.
    Points present in only one result set still get partial credit.
    """
    scores: dict = {}
    point_map: dict = {}

    dense_pts  = dense_results.points  if hasattr(dense_results,  "points") else (dense_results  or [])
    sparse_pts = sparse_results.points if hasattr(sparse_results, "points") else (sparse_results or [])

    for rank, p in enumerate(dense_pts):
        scores[p.id]    = scores.get(p.id, 0.0) + 1.0 / (k + rank + 1)
        point_map[p.id] = p
    for rank, p in enumerate(sparse_pts):
        scores[p.id] = scores.get(p.id, 0.0) + 1.0 / (k + rank + 1)
        point_map.setdefault(p.id, p)

    sorted_ids = sorted(scores, key=scores.__getitem__, reverse=True)[:top_k]
    result = []
    for pid in sorted_ids:
        if pid in point_map:
            p = point_map[pid]
            p.score = scores[pid]
            result.append(p)
    return result


def rrf_fuse_multi(result_sets: list, k: int = 60, top_k: int = 6) -> list:
    """
    RRF fusion over N result sets (generalisation of rrf_fuse for multi-year retrieval).
    Each element may be a QueryResponse (has .points) or a plain list of ScoredPoints.
    """
    scores: dict = {}
    point_map: dict = {}
    for result_set in result_sets:
        pts = result_set.points if hasattr(result_set, "points") else (result_set or [])
        for rank, p in enumerate(pts):
            scores[p.id] = scores.get(p.id, 0.0) + 1.0 / (k + rank + 1)
            point_map.setdefault(p.id, p)
    sorted_ids = sorted(scores, key=scores.__getitem__, reverse=True)[:top_k]
    result = []
    for pid in sorted_ids:
        if pid in point_map:
            p = point_map[pid]
            p.score = scores[pid]
            result.append(p)
    return result


def search_dense_multi_year(coll_name, query_vec, base_filters, years, prefetch_k):
    """
    Dense retrieval across multiple years with recency-weighted top_k budgets.
    Runs one search per year, RRF-fuses all results into a single QueryResponse-like object.

    base_filters : payload_must dict WITHOUT a year key
    years        : list of ints (all years to query across)
    prefetch_k   : total candidate budget; distributed proportionally across years
    """
    from types import SimpleNamespace
    year_wts = year_weights(years)
    result_sets = []
    for yr, w in year_wts:
        k_i = max(1, round(prefetch_k * w))
        yr_filters = {**base_filters, "year": yr}
        res = search_with_payload(coll_name, query_vec, payload_must=yr_filters, top_k=k_i)
        result_sets.append(res)
    fused = rrf_fuse_multi(result_sets, top_k=prefetch_k)
    return SimpleNamespace(points=fused)


def search_bm25_multi_year(coll_name_sparse, query_text, base_filters, years, prefetch_k):
    """
    BM25 retrieval across multiple years with recency-weighted top_k budgets.
    Mirrors search_dense_multi_year for the sparse collection.
    """
    from types import SimpleNamespace
    year_wts = year_weights(years)
    result_sets = []
    for yr, w in year_wts:
        k_i = max(1, round(prefetch_k * w))
        yr_filters = {**base_filters, "year": yr}
        res = search_bm25(coll_name_sparse, query_text, payload_must=yr_filters, top_k=k_i)
        result_sets.append(res)
    fused = rrf_fuse_multi(result_sets, top_k=prefetch_k)
    return SimpleNamespace(points=fused)


def search_dense_for_year(coll_name, query_vec, base_filters, year_val, prefetch_k):
    """
    Resolve a query's year filter and run dense retrieval.

    - year_val is None (query gave no year signal): fall back to the
      default 5-year window (current anchor year + 4 previous, see
      utils._default_year_window) with exponential recency decay -- the
      system's own guess when the user wasn't specific, weighted so the
      most recent year dominates.
    - year_val is an int or list (explicit, deterministically extracted
      from the query text): flat, unweighted OR filter across exactly
      those years. We trust what the user asked for -- overriding it with
      a recency prior would contradict an explicit request.
    """
    if year_val is None:
        return search_dense_multi_year(coll_name, query_vec, base_filters, _default_year_window(), prefetch_k)
    filters = {**base_filters, "year": year_val}
    return search_with_payload(coll_name, query_vec, payload_must=filters, top_k=prefetch_k)


def search_bm25_for_year(coll_name_sparse, query_text, base_filters, year_val, prefetch_k):
    """BM25 counterpart of search_dense_for_year -- see its docstring."""
    if year_val is None:
        return search_bm25_multi_year(coll_name_sparse, query_text, base_filters, _default_year_window(), prefetch_k)
    filters = {**base_filters, "year": year_val}
    return search_bm25(coll_name_sparse, query_text, payload_must=filters, top_k=prefetch_k)


def get_all_chunks_for_payload(coll_name, payload_must=None):
    """
    Retrieves all chunks matching the payload filter using scroll.
    """
    must = []
    if payload_must:
        items = payload_must.items() if isinstance(payload_must, dict) else payload_must
        for k, v in items:
            if k in ["fiscal_year_end", "year"]:
                year_vals = v if isinstance(v, list) else [v]
                year_conditions = []
                for val in year_vals:
                    try:
                        if hasattr(val, 'year'):
                            yr = val.year
                        elif isinstance(val, str):
                            yr = int(val[:4])
                        else:
                            yr = int(val)
                        
                        year_conditions.append(
                            models.FieldCondition(
                                key="fiscal_year_end",
                                range=models.DatetimeRange(
                                    gte=f"{yr}-01-01T00:00:00Z",
                                    lte=f"{yr}-12-31T23:59:59Z"
                                )
                            )
                        )
                    except (ValueError, TypeError):
                        pass
                
                if len(year_conditions) == 1:
                    must.append(year_conditions[0])
                elif len(year_conditions) > 1:
                    must.append(models.Filter(should=year_conditions))
            else:
                must.append(models.FieldCondition(key=k, match=models.MatchValue(value=v)))
    
    all_points = []
    next_page = None
    while True:
        res, next_page = client.scroll(
            collection_name=coll_name,
            scroll_filter=models.Filter(must=must) if must else None,
            limit=100,
            with_payload=True,
            with_vectors=False,
            offset=next_page
        )
        all_points.extend(res)
        if not next_page:
            break
    return all_points

# def search_agent(user_query, model, tokenizer, embed_model, coll_name, ENAHNCE_QUERY=True, BOTH = True):
#     """
#     1. Enhances the user query for vector search using an LLM. [depends on the boolean]
#     2. Extracts payload filters (ticker, year, form_type).
#     3. Embeds the optimized prompt.
#     4. Calls search_with_payload to get results.
#     """
#     messages = [
#         {"role": "system", "content": META_EXTRACT_PROMPT},
#         {"role": "user", "content": user_query} # instead of embdding the query within the META_EXTRACT_PROMPT
#     ]
    
#     prompt = tokenizer.apply_chat_template(
#         messages, tokenize=False, add_generation_prompt=True
#     )
    
#     # Generate LLM response
#     response_text = generate(model, tokenizer, prompt=prompt, verbose=False)
#     try:
#         meta           = parse_metadata_response(response_text, fallback_query=user_query)
#         opt_retr_query = meta["optimized_query"]
#         payload_filters = {k: meta[k] for k in ("ticker", "year", "form_type") if meta.get(k)}
        
#         # Embed the optimized prompt
        
#         query_vec_enhanced = embed_model.encode(opt_retr_query, 
#                                                 prompt_name="Retrieval-query").tolist()
        
#         query_vec_raw = embed_model.encode(user_query,
#                                            prompt_name="Retrieval-query").tolist()
        
#         # Call the search function
#         if ENAHNCE_QUERY and BOTH:
#             results_enhanced = search_with_payload(coll_name, query_vec_enhanced, payload_must=payload_filters)
#             results_raw = search_with_payload(coll_name, query_vec_raw, payload_must=payload_filters)
#             return opt_retr_query, results_enhanced, results_raw
#         elif ENAHNCE_QUERY:
#             results_enhanced = search_with_payload(coll_name, query_vec_enhanced, payload_must=payload_filters)
#             return opt_retr_query, results_enhanced, None
#         else:
#             results_raw = search_with_payload(coll_name, query_vec_raw, payload_must=payload_filters)
#             return None,None,results_raw
            
#     except Exception as e:
#         print(f"Error in search_agent: {e}")
#         return None, None

def _build_rag_prompt(user_query, search_results, tokenizer):
    """
    Build the chat-templated prompt text and context_chunks for one RAG query, or
    (None, []) if there's nothing retrieved to answer from. Shared by
    generate_llm_answer and generate_llm_answers_batch so both build the exact same
    prompt for the same inputs -- batching must not change what gets asked.
    """
    if not search_results or not search_results.points:
        return None, []

    # Extract and format context from search results
    context_chunks = []
    for i, point in enumerate(search_results.points):
        text = point.payload.get("text", "No text content available.")
        ticker = point.payload.get("ticker", "n/a")
        year = point.payload.get("fiscal_year_end", "n/a")

        # Robust year extraction for display
        display_year = "n/a"
        try:
            if hasattr(year, 'year'):
                display_year = str(year.year)
            elif isinstance(year, str):
                display_year = year[:4]
            else:
                display_year = str(year)
        except:
            pass

        context_chunks.append(f"--- Source {i+1} (Ticker: {ticker.upper()}, Year: {display_year}) ---\n{text}")

    context_text = "\n\n".join(context_chunks)

    messages = [
        {"role": "system", "content": RAG_ANSWER_PROMPT},
        {"role": "user", "content": f"Context:\n{context_text}\n\n Question: {user_query}"}
    ]

    # enable_thinking=False: with it on, this model burns thousands of tokens on a
    # plain-text reasoning preamble (no <think> tag to strip it by) before ever
    # emitting an answer - ~15x slower for an equivalent final answer.
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )
    return prompt, context_chunks


def generate_llm_answer(user_query, search_results, model, tokenizer):
    """
    Given a user query and search results, this function returns an LLM generated answer.
    """
    prompt, context_chunks = _build_rag_prompt(user_query, search_results, tokenizer)
    if prompt is None:
        return "No relevant information found in the database to answer your query.", []

    generated_text = generate(model, tokenizer, prompt=prompt, verbose=False, max_tokens=2048)

    return generated_text.strip(), context_chunks


def _encode_for_batch(tokenizer, prompt_text: str) -> list:
    """Tokenize a chat-templated prompt string the same way mlx_lm.generate does
    internally (stream_generate), so batched and single-item generation see
    identical token inputs for the identical prompt text."""
    add_special = tokenizer.bos_token is None or not prompt_text.startswith(tokenizer.bos_token)
    return tokenizer.encode(prompt_text, add_special_tokens=add_special)


def generate_llm_answers_batch(
    items, model, tokenizer, max_tokens=2048,
    completion_batch_size=3, prefill_batch_size=1,
):
    """
    Batched version of generate_llm_answer: generates answers for a list of
    (user_query, search_results) pairs in ONE batched forward pass, instead of one
    generate() call per item -- lets a single GPU (e.g. Apple Silicon/Metal) actually
    process multiple prompts at once instead of serializing separate calls.

    Items with no retrieved points get the same "no relevant information" sentinel
    as generate_llm_answer, without being sent to the model at all -- only items
    that actually need generation are included in the batch call. Returns a list of
    (answer_text, context_chunks) tuples, one per item, in the same order as `items`.

    completion_batch_size/prefill_batch_size are forwarded to mlx_lm's
    BatchGenerator, which otherwise defaults to 32/8 -- that many prompts'
    KV caches held concurrently is what was OOM-ing. Lowering these caps how
    many sequences run at once (trading throughput for a lower memory
    ceiling) rather than how many prompts are logically in the batch.
    """
    from mlx_lm import batch_generate

    results = [None] * len(items)
    batch_prompts  = []
    batch_chunks   = []
    batch_positions = []

    for i, (user_query, search_results) in enumerate(items):
        prompt, context_chunks = _build_rag_prompt(user_query, search_results, tokenizer)
        if prompt is None:
            results[i] = ("No relevant information found in the database to answer your query.", [])
        else:
            batch_prompts.append(prompt)
            batch_chunks.append(context_chunks)
            batch_positions.append(i)

    if batch_prompts:
        token_prompts = [_encode_for_batch(tokenizer, p) for p in batch_prompts]
        batch = batch_generate(
            model, tokenizer, token_prompts, max_tokens=max_tokens, verbose=False,
            completion_batch_size=completion_batch_size, prefill_batch_size=prefill_batch_size,
        )
        for pos, text, chunks in zip(batch_positions, batch.texts, batch_chunks):
            results[pos] = (text.strip(), chunks)

    return results
