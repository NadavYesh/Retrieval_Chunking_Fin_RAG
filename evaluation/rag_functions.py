import gc
import re
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

import mlx.core as mx
from mlx_lm import generate
from prompts import META_EXTRACT_PROMPT, QUERY_ENHANCEMENT_PROMPT
from utils import parse_metadata_response, extract_ticker_deterministic, extract_year_deterministic

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
    formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

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


def extract_metadata(query: str, model, tokenizer) -> dict:
    """
    ticker/year/form_type are resolved deterministically (no LLM, no
    hallucination risk); the LLM call below produces only the
    optimized_query rewrite used for vector search.
    """
    messages = [
        {"role": "system", "content": META_EXTRACT_PROMPT},
        {"role": "user",   "content": query},
    ]
    prompt   = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    response = generate(model, tokenizer, prompt=prompt, verbose=False)
    gc.collect()
    mx.clear_cache()

    meta = parse_metadata_response(response, fallback_query=query)
    meta["ticker"]    = extract_ticker_deterministic(query)
    meta["year"]      = extract_year_deterministic(query)
    meta["form_type"] = "10-k"

    return meta


def embed_query(query: str, embed_model, embed_tokenizer) -> list[float]:
    inputs = embed_tokenizer.batch_encode_plus(
        [f"task: search result | query: {query}"],
        return_tensors="mlx",
        padding=True,
        truncation=True,
    )
    outputs = embed_model(inputs["input_ids"], attention_mask=inputs["attention_mask"])
    return outputs.text_embeds.tolist()[0]
