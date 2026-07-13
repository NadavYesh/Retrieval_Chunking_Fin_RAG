# Running the LLM calls on a server

Only generation and judging move off-box. Qdrant, embeddings, chunking, retrieval and
fusion stay local and are unchanged. `LLM_BACKEND` is unset by default, so nothing here
changes the local MLX behaviour.

## What you cannot reproduce, and why

The local models are MLX quantizations with **no NVIDIA equivalent**:

| local (MLX, Metal only) | quantization | server equivalent | quantization |
|---|---|---|---|
| `mlx-community/Qwen3.5-9B-OptiQ-4bit` | affine, group 64, **mixed** — 250 layers pinned (134 @ 8-bit, 116 @ 4-bit) | `cyankiwi/Qwen3.5-9B-AWQ-4bit` | AWQ 4-bit |
| `mlx-community/phi-4-4bit` | affine 4-bit, group 64 | a Phi-4 checkpoint (see below) | AWQ / GPTQ / bf16 |

MLX's affine format is Metal-only; no CUDA runtime loads it and there is no lossless
converter. The server runs the **same base models** at a **different 4-bit quantization**.
Answers will differ from the local run in wording and occasionally in content. Treat this
as a re-run, not a bit-exact reproduction, and say so in the methods section.

A note on the judge: `unsloth/phi-4-bnb-4bit` works, but bitsandbytes is the slowest
quantization path in vLLM (it exists mainly for QLoRA, not throughput serving), and
Unsloth converted Phi-4 to the Llama architecture rather than the `Phi3ForCausalLM` used
locally. For ~4,300 judge calls, an AWQ/GPTQ Phi-4 — or plain bf16 (14B ≈ 28GB, fits an
80GB card) — will be materially faster.

## Serve

Qwen3.5 needs vLLM from **main**, not a stable release.

```bash
vllm serve cyankiwi/Qwen3.5-9B-AWQ-4bit --quantization awq \
    --max-model-len 32768 --port 8000        # contexts reach ~28k tokens
vllm serve <phi-4 checkpoint> --max-model-len 16384 --port 8001
```

`--max-model-len 32768` is not padding: the p95 retrieved context is ~10k tokens and the
longest observed is ~28k (a Level-2 parent-fetch case).

## Run

```bash
export LLM_BACKEND=api
export LLM_API_BASE=http://<host>:8000/v1
export LLM_JUDGE_API_BASE=http://<host>:8001/v1
export LLM_GEN_MODEL=cyankiwi/Qwen3.5-9B-AWQ-4bit
export LLM_JUDGE_MODEL=<phi-4 checkpoint>
export LLM_CONCURRENCY=32

python -m llm_backend                  # health check: both endpoints, expected model ids
python evaluation/run_rag_lazy.py      # 121 questions x 36 configs = 4,356 runs
python evaluation/evaluation_run.py    # set eval_files to the JSON the above wrote
```

Both scripts health-check the endpoints before doing any work, so a wrong model id or a
down server fails immediately rather than 4,000 requests in.

`temperature=0` is forced on every request to match `mlx_lm.batch_generate`'s greedy
default — otherwise the server's own default (often 0.7) would stack sampling noise on
top of the quantization difference. Qwen additionally gets
`chat_template_kwargs={"enable_thinking": false}`, the API equivalent of the
`enable_thinking=False` the local path passes to `apply_chat_template`; without it Qwen
emits a long plain-text reasoning preamble with no `<think>` tag to strip.

## Expected volume and cost

~4,150 generation calls (16.3M in / 2.6M out) and ~4,270 judge calls (6.1M in / 1.3M out).
On one H100 with vLLM that is roughly 40 minutes of GPU work; budget an hour, so about
$2–3 at typical rental rates. Bandwidth is negligible (~65MB of prompt text total).
