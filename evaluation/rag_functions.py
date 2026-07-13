import gc
import re
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import mlx.core as mx
from mlx_lm import generate
from prompts import QUERY_ENHANCEMENT_PROMPT
from utils import extract_ticker_deterministic, extract_year_deterministic

MAX_ENHANCE_RETRIES = 3

_REFUSAL_RE = re.compile(
    r"i (can'?t|cannot|am unable|won'?t|must decline|'?m not able)|"
    r"i'?m sorry|^sorry[,. ]|"
    r"this (request|query|question) (is|involves|relates to)|"
    r"as an? (ai|language model|assistant)[,. ]",
    re.IGNORECASE,
)


def _refusal_fallback(raw: str, original: str) -> str:
    m = _REFUSAL_RE.search(raw)
    if m is None:
        return raw.strip()
    before = raw[:m.start()].strip()
    if len(before) > 20:
        return before
    parts = re.split(r"(?:as requested[,.]?|however[,.]?|here is[,:]?)\s*", raw, flags=re.IGNORECASE)
    candidate = parts[-1].strip() if len(parts) > 1 else ""
    orig_tokens = set(original.lower().split())
    if candidate and any(tok in candidate.lower() for tok in orig_tokens):
        return candidate
    print("  [enhance] refusal detected — passing original query through")
    return original


def enhance_query(query: str, model, tokenizer) -> str:
    messages = [
        {"role": "system", "content": QUERY_ENHANCEMENT_PROMPT},
        {"role": "user",   "content": query},
    ]
    # enable_thinking=False: some models (e.g. Qwen3.5) burn thousands of tokens
    # on a plain-text reasoning preamble otherwise -- see the identical fix in
    # search_engine.py:generate_llm_answer. Harmless no-op for chat templates
    # that don't reference this variable (e.g. Llama's).
    formatted = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )

    for attempt in range(1, MAX_ENHANCE_RETRIES + 1):
        print(f"*****************************\n\nThe raw user query is {query}")
        enhanced_raw = generate(model, tokenizer, prompt=formatted, verbose=False, max_tokens=300)
        gc.collect()
        mx.clear_cache()
        print(f"  [enhance attempt {attempt}] {len(enhanced_raw)} chars: {repr(enhanced_raw[:80])}")
        if enhanced_raw.strip():
            return _refusal_fallback(enhanced_raw, query)
        print(f"  [enhance attempt {attempt}] empty output — retrying")

    print(f"  [ERROR] enhance_query: all {MAX_ENHANCE_RETRIES} attempts empty — falling back to original")
    return query


def _encode_for_batch(tokenizer, prompt_text: str) -> list:
    """Tokenize a chat-templated prompt string the same way mlx_lm.generate does
    internally (stream_generate), so batched and single-item generation see
    identical token inputs for the identical prompt text."""
    add_special = tokenizer.bos_token is None or not prompt_text.startswith(tokenizer.bos_token)
    return tokenizer.encode(prompt_text, add_special_tokens=add_special)


def _chat_prompt(tokenizer, system: str, user: str) -> str:
    messages = [
        {"role": "system", "content": system},
        {"role": "user",   "content": user},
    ]
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )


def _batch_generate_texts(
    model, tokenizer, prompts_text: list[str], max_tokens: int,
    completion_batch_size: int, prefill_batch_size: int,
) -> list[str]:
    """
    Run one batched forward pass over already chat-templated prompt strings.

    completion_batch_size/prefill_batch_size are forwarded to mlx_lm's
    BatchGenerator, which otherwise defaults to 32/8 -- that many prompts'
    KV caches held concurrently is what OOMs on a single Metal GPU. Lowering
    them caps how many sequences run at once, not how many prompts are
    logically in the batch.
    """
    from mlx_lm import batch_generate

    token_prompts = [_encode_for_batch(tokenizer, p) for p in prompts_text]
    batch = batch_generate(
        model, tokenizer, token_prompts, max_tokens=max_tokens, verbose=False,
        completion_batch_size=completion_batch_size, prefill_batch_size=prefill_batch_size,
    )
    gc.collect()
    mx.clear_cache()
    return list(batch.texts)


def enhance_query_batch(
    queries: list[str], model, tokenizer,
    completion_batch_size: int = 3, prefill_batch_size: int = 1,
) -> list[str]:
    """
    Batched version of enhance_query: enhances a list of queries in ONE batched
    forward pass instead of one generate() call per query.

    Retry semantics match enhance_query: an empty generation is retried up to
    MAX_ENHANCE_RETRIES times, but only the still-empty queries are re-sent, as
    a smaller batch. Queries still empty after the last attempt fall back to the
    original text. Returns one string per input query, in input order.
    """
    if not queries:
        return []

    prompts = [_chat_prompt(tokenizer, QUERY_ENHANCEMENT_PROMPT, q) for q in queries]
    results: list[str | None] = [None] * len(queries)
    pending = list(range(len(queries)))

    for attempt in range(1, MAX_ENHANCE_RETRIES + 1):
        texts = _batch_generate_texts(
            model, tokenizer, [prompts[i] for i in pending], 300,
            completion_batch_size, prefill_batch_size,
        )
        still_pending = []
        for i, raw in zip(pending, texts):
            if raw.strip():
                results[i] = _refusal_fallback(raw, queries[i])
            else:
                still_pending.append(i)
        print(f"  [enhance attempt {attempt}] {len(pending) - len(still_pending)}/{len(pending)} produced output")
        pending = still_pending
        if not pending:
            break

    for i in pending:
        print(f"  [ERROR] enhance_query: all {MAX_ENHANCE_RETRIES} attempts empty for "
              f"{queries[i][:60]!r} — falling back to original")
        results[i] = queries[i]

    return results


def extract_metadata_batch(
    queries: list[str]
    #model, tokenizer,
    #completion_batch_size: int = 3, prefill_batch_size: int = 1,
) -> list[dict]:
    """
    Batched version of extract_metadata. As in the single-query path, only the
    optimized_query rewrite comes from the LLM; ticker/year/form_type are
    resolved deterministically per query. Returns one meta dict per input query,
    in input order.
    """
    if not queries:
         return []
    metas = []
    for query in queries:
        meta = {}
        meta["ticker"]    = extract_ticker_deterministic(query)
        meta["year"]      = extract_year_deterministic(query)
        meta["form_type"] = "10-k"
        metas.append(meta)

    return metas


def embed_query_batch(queries: list[str], embed_model, embed_tokenizer) -> list[list[float]]:
    """
    Batched version of embed_query: embeds all queries in one padded forward pass.
    Returns one vector per input query, in input order.
    """
    if not queries:
        return []

    inputs = embed_tokenizer.batch_encode_plus(
        [f"task: search result | query: {q}" for q in queries],
        return_tensors="mlx",
        padding=True,
        truncation=True,
    )
    outputs = embed_model(inputs["input_ids"], attention_mask=inputs["attention_mask"])
    return outputs.text_embeds.tolist()


# def extract_metadata(query: str, model, tokenizer) -> dict:
#     """
#     ticker/year/form_type are resolved deterministically (no LLM, no
#     hallucination risk); the LLM call below produces only the
#     optimized_query rewrite used for vector search.
#     """
#     messages = [
#         {"role": "system", "content": META_EXTRACT_PROMPT},
#         {"role": "user",   "content": query},
#     ]
#     prompt   = tokenizer.apply_chat_template(
#         messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
#     )
#     response = generate(model, tokenizer, prompt=prompt, verbose=False)
#     gc.collect()
#     mx.clear_cache()

#     meta = parse_metadata_response(response, fallback_query=query)
#     meta["ticker"]    = extract_ticker_deterministic(query)
#     meta["year"]      = extract_year_deterministic(query)
#     meta["form_type"] = "10-k"

#     return meta


def embed_query(query: str, embed_model, embed_tokenizer) -> list[float]:
    inputs = embed_tokenizer.batch_encode_plus(
        [f"task: search result | query: {query}"],
        return_tensors="mlx",
        padding=True,
        truncation=True,
    )
    outputs = embed_model(inputs["input_ids"], attention_mask=inputs["attention_mask"])
    return outputs.text_embeds.tolist()[0]
