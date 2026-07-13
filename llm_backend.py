"""
LLM backend switch: local MLX (Apple Silicon) or a remote OpenAI-compatible server.

Only the LLM calls move off-box. Qdrant, embeddings, chunking, retrieval and fusion
all stay local and are untouched by this module.

Reproduction caveat (read before quoting numbers)
    The local run uses MLX quantizations that have no NVIDIA equivalent:
      - mlx-community/Qwen3.5-9B-OptiQ-4bit  -- MLX *mixed-precision affine* quant,
        group_size 64, with 250 layers individually pinned (134 @ 8-bit, 116 @ 4-bit).
      - mlx-community/phi-4-4bit             -- MLX affine 4-bit, group_size 64.
    MLX's affine format is Metal-only; no CUDA runtime loads it, and there is no
    lossless converter. The server models below are the SAME BASE MODELS at a
    COMPARABLE but DIFFERENT 4-bit quantization (AWQ / bnb). Expect answers to differ
    from the local run in wording and occasionally in content. This is a
    re-run, not a bit-exact reproduction.

Serving (vLLM, OpenAI-compatible) -- ON THE GPU BOX, never on the Mac:
    # Qwen3.5 needs vLLM from main, not a stable release.
    # Do NOT pass --quantization awq: despite the repo name this checkpoint ships as
    # compressed-tensors, and forcing awq makes vLLM refuse to start. Both models let
    # vLLM read the scheme straight from the checkpoint.
    vllm serve cyankiwi/Qwen3.5-9B-AWQ-4bit \
        --max-model-len 32768 --host 0.0.0.0 --port 8000
    vllm serve RedHatAI/phi-4-quantized.w4a16 \
        --max-model-len 16384 --host 0.0.0.0 --port 8000

Configure via environment:
    LLM_BACKEND=api            # "mlx" (default) or "api"
    LLM_API_BASE=http://host:8000/v1
    LLM_API_KEY=...            # vLLM accepts any string unless --api-key is set
    LLM_GEN_MODEL=cyankiwi/Qwen3.5-9B-AWQ-4bit
    LLM_JUDGE_MODEL=<phi-4 checkpoint served>
    LLM_JUDGE_API_BASE=...     # optional; defaults to LLM_API_BASE
    LLM_CONCURRENCY=32
"""

import os
import time
from concurrent.futures import ThreadPoolExecutor

import requests

# ── Config ───────────────────────────────────────────────────────────────────

USE_API = os.getenv("LLM_BACKEND", "mlx").lower() == "api"

API_BASE       = os.getenv("LLM_API_BASE", "http://localhost:8000/v1").rstrip("/")
API_KEY        = os.getenv("LLM_API_KEY", "EMPTY")
GEN_MODEL      = os.getenv("LLM_GEN_MODEL", "cyankiwi/Qwen3.5-9B-AWQ-4bit")
JUDGE_MODEL    = os.getenv("LLM_JUDGE_MODEL", "RedHatAI/phi-4-quantized.w4a16")
JUDGE_API_BASE = os.getenv("LLM_JUDGE_API_BASE", API_BASE).rstrip("/")

# vLLM batches server-side; this only caps how many requests are in flight at once.
# run_rag_lazy generates once per question, sending that question's unique retrievals as
# one chat_batch call -- measured at mean 26.6, max 36 (the 36 configs de-duplicate down).
# 40 fits the largest question in a single wave. At 32, ~15% of questions spilled 2-4
# stragglers into a second wave and the whole question blocked on them with the GPU idle.
CONCURRENCY = int(os.getenv("LLM_CONCURRENCY", "40"))
TIMEOUT     = int(os.getenv("LLM_TIMEOUT", "600"))
MAX_RETRIES = 5


class LLMBackendError(RuntimeError):
    pass


class _RemoteModel:
    """
    Stand-in for a loaded local model when the real one lives on the server.

    The pipeline gates work on `model is not None` in several places (e.g. "LLM judging
    only runs when judge_model is given"). On the API backend there is no local model
    object, but the work must still run -- so callers pass this instead of None. Nothing
    ever calls into it: judge_llm_batch / generate_llm_answers_batch check
    llm_backend.USE_API first and route to chat_batch.
    """
    def __repr__(self):
        return f"<remote model via {API_BASE}>"


REMOTE_MODEL = _RemoteModel()


def _post_chat(messages, model, max_tokens, base_url, extra_body=None):
    """One chat completion, retried on transport errors and 429/5xx."""
    payload = {
        "model":       model,
        "messages":    messages,
        "max_tokens":  max_tokens,
        # Greedy, to match mlx_lm.batch_generate's default sampler. Without this the
        # server's own default (often temperature=0.7) would add sampling noise on top
        # of the quantization difference, making the re-run harder to interpret.
        "temperature": 0.0,
    }
    if extra_body:
        payload.update(extra_body)

    headers = {"Authorization": f"Bearer {API_KEY}"}
    delay   = 2.0
    last    = None
    for attempt in range(MAX_RETRIES):
        try:
            r = requests.post(f"{base_url}/chat/completions", json=payload,
                              headers=headers, timeout=TIMEOUT)
            if r.status_code == 200:
                return r.json()["choices"][0]["message"]["content"] or ""
            # 4xx other than 429 is a request bug -- retrying will not help.
            if r.status_code != 429 and r.status_code < 500:
                raise LLMBackendError(f"{r.status_code} from {base_url}: {r.text[:300]}")
            last = f"{r.status_code}: {r.text[:200]}"
        except requests.RequestException as e:
            last = f"{type(e).__name__}: {e}"
        if attempt < MAX_RETRIES - 1:
            time.sleep(delay)
            delay *= 2
    raise LLMBackendError(f"failed after {MAX_RETRIES} attempts -- {last}")


def chat_batch(messages_list, model, max_tokens, base_url=None, extra_body=None):
    """
    Run many chat completions concurrently; returns texts in the input order.

    Mirrors the contract of mlx_lm.batch_generate as this codebase uses it: a list of
    prompts in, a list of completed texts out, same order. Concurrency replaces MLX's
    single batched forward pass -- vLLM does the actual batching server-side.

    Per-prompt failures are isolated. One prompt that the server refuses -- typically a
    context longer than --max-model-len, which is a permanent property of that prompt and
    not of the server -- used to raise out of the whole batch. Since callers batch a
    question's configs together, a single oversized context wiped out every config for that
    question. Such a prompt now yields "" while its siblings still return.

    A batch in which EVERY prompt fails still raises: that is the signature of a dead
    backend, and the caller's consecutive-failure guard needs to see it.
    """
    if not messages_list:
        return []
    base_url = base_url or API_BASE
    with ThreadPoolExecutor(max_workers=min(CONCURRENCY, len(messages_list))) as pool:
        futures = [
            pool.submit(_post_chat, m, model, max_tokens, base_url, extra_body)
            for m in messages_list
        ]
        texts, errors = [], []
        for i, f in enumerate(futures):
            try:
                texts.append(f.result())
            except Exception as e:
                errors.append(f"[{i}] {e}")
                texts.append("")

    if errors and len(errors) == len(messages_list):
        raise LLMBackendError(
            f"all {len(errors)} request(s) in the batch failed -- backend is down.\n  "
            + "\n  ".join(errors[:3])
        )
    if errors:
        print(f"      [gen] {len(errors)}/{len(messages_list)} prompt(s) rejected by the "
              f"server (answer left empty):")
        for e in errors[:3]:
            print(f"        {e[:160]}")
    return texts


# Qwen3.5 emits a long plain-text reasoning preamble unless thinking is disabled, the
# same problem _build_rag_prompt works around locally with enable_thinking=False. vLLM
# forwards chat_template_kwargs into the model's chat template, so the server applies
# the identical switch.
QWEN_NO_THINK = {"chat_template_kwargs": {"enable_thinking": False}}


def health_check():
    """Verify both endpoints are up and serving the expected model ids."""
    out = []
    for label, base, model in (("generation", API_BASE, GEN_MODEL),
                               ("judge", JUDGE_API_BASE, JUDGE_MODEL)):
        try:
            r = requests.get(f"{base}/models", headers={"Authorization": f"Bearer {API_KEY}"},
                             timeout=15)
            served = [m["id"] for m in r.json().get("data", [])]
            ok = model in served
            out.append(f"  [{'OK ' if ok else 'BAD'}] {label:10s} {base}  "
                       f"wants={model}  serving={served}")
        except Exception as e:
            out.append(f"  [DOWN] {label:10s} {base}  ({type(e).__name__}: {e})")
    return "\n".join(out)


if __name__ == "__main__":
    print(f"LLM_BACKEND={'api' if USE_API else 'mlx'}")
    print(health_check())
