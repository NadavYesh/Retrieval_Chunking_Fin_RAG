# Running the LLM calls on a server

Only generation and judging move off-box. Qdrant, embeddings, chunking, retrieval and
fusion stay local and are unchanged. `LLM_BACKEND` is unset by default, so nothing here
changes the local MLX behaviour.

## What you cannot reproduce, and why

The local models are MLX quantizations with **no NVIDIA equivalent**:

| role | local (MLX, Metal only) | quantization | server | quantization |
|---|---|---|---|---|
| generation | `mlx-community/Qwen3.5-9B-OptiQ-4bit` | affine, group 64, **mixed** — 250 layers pinned (134 @ 8-bit, 116 @ 4-bit) | `cyankiwi/Qwen3.5-9B-AWQ-4bit` | AWQ 4-bit |
| judge | `mlx-community/phi-4-4bit` | affine 4-bit, group 64 | `RedHatAI/phi-4-quantized.w4a16` | GPTQ INT4, group 128 |

MLX's affine format is Metal-only; no CUDA runtime loads it and there is no lossless
converter. The server runs the **same base models** at a **different 4-bit quantization**.
Answers will differ from the local run in wording and occasionally in content. Treat this
as a re-run, not a bit-exact reproduction, and say so in the methods section.

### Why `RedHatAI/phi-4-quantized.w4a16` for the judge

It is `Phi3ForCausalLM` — the same architecture the local `mlx-community/phi-4-4bit` runs
— so quantization is the *only* thing that changes. It is built by RedHat/Neural Magic,
who maintain vLLM, and `compressed-tensors` w4a16 is a first-class fast kernel there. And
it publishes accuracy recovery against unquantized Phi-4: **99.3% average** over six
benchmarks (MMLU 99.5%, GSM-8K 99.6%, ARC-C 97.6%, HellaSwag 98.9%, Winogrande 100.2%,
TruthfulQA 99.7%) — citable evidence that the judge is not materially degraded.

Rejected alternatives:
- `unsloth/phi-4-bnb-4bit` — Unsloth converts Phi-4 to the **Llama** architecture (a second
  confound on top of the quantization change), and **bitsandbytes is the slowest quant path
  in vLLM**: it exists mainly to serve QLoRA adapters, not for throughput. Wrong trade for
  ~4,300 judge calls.
- Community GPTQ repos (`jakiAJK/...`, `fhamborg/...`) — fine, but no published accuracy
  recovery and not maintained by the vLLM team.
- Plain bf16 `microsoft/phi-4` — no quantization error at all, but 14B ≈ 28GB of weights.
  A legitimate choice if you have an 80GB card and would rather remove the judge's
  quantization as a variable entirely.

## Serve — on the GPU box, never on the Mac

vLLM is a CUDA server. Installing it on the Mac gets you a CPU-only build that fails with
`Failed to import from vllm._C` / `cpu_fused_moe`. The Mac needs nothing beyond `requests`;
it only makes HTTP calls.

Qwen3.5 needs vLLM from **main**, not a stable release:

```bash
pip install -U "vllm @ git+https://github.com/vllm-project/vllm.git"
```

Serve one model at a time — `run_rag_lazy.py` only uses Qwen and `evaluation_run.py` only
uses Phi-4, so they never need to be up together:

```bash
vllm serve cyankiwi/Qwen3.5-9B-AWQ-4bit \
    --max-model-len 32768 --host 0.0.0.0 --port 8000   # contexts reach ~28k tokens

vllm serve RedHatAI/phi-4-quantized.w4a16 \
    --max-model-len 16384 --host 0.0.0.0 --port 8000
```

Two flags that are not optional:

- **No `--quantization` flag.** Despite its name, `cyankiwi/Qwen3.5-9B-AWQ-4bit` ships as
  *compressed-tensors*, not classic AWQ; passing `--quantization awq` makes vLLM refuse to
  start ("Quantization method specified in the model config (compressed-tensors) does not
  match ... (awq)"). Let vLLM read the scheme from the checkpoint, for both models.
- **`--host 0.0.0.0`.** vLLM defaults to binding localhost, which a cloud provider's HTTP
  proxy cannot reach.

`--max-model-len 32768` is not padding: the p95 retrieved context is ~10k tokens and the
longest observed is ~28k (a Level-2 parent-fetch case).

## Run

```bash
export LLM_BACKEND=api
export LLM_API_BASE=http://<host>:8000/v1
export LLM_JUDGE_API_BASE=http://<host>:8001/v1
export LLM_GEN_MODEL=cyankiwi/Qwen3.5-9B-AWQ-4bit
export LLM_JUDGE_MODEL=RedHatAI/phi-4-quantized.w4a16
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
