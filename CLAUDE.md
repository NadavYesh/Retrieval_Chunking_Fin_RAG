# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A thesis project implementing a RAG (retrieval-augmented generation) pipeline over SEC 10-K
filings, evaluated against the **FinDER** benchmark (`/Users/nadavsmacbookair/Desktop/Thesis/data/FinDER/train.parquet`).
Everything runs locally on Apple Silicon via **MLX** — no external LLM APIs.

Data lives outside this repo, under `/Users/nadavsmacbookair/Desktop/Thesis/data/` (raw HTML
filings, markdown, pickled chunk DataFrames, eval datasets/results, SP500 reference CSV). Code
paths frequently hardcode absolute paths into that data directory — check the CONFIG block at
the top of a script before running it.

## Running things

There's no `requirements.txt`/`pyproject.toml`; dependencies are managed in an external
env — commands from `.claude/settings.local.json` suggest a couple of interpreters have been
used across the project's life (`/opt/venv/thesis_venv_02/bin/python`, and conda envs
`mps_env`/`vision_env`). Confirm which interpreter has `qdrant_client`, `mlx`, `mlx_lm`,
`mlx_embeddings`, `sentence_transformers`, `langchain_text_splitters`, `tiktoken`, `pandas`,
`openpyxl` installed before running scripts.

Qdrant must be running locally at `http://localhost:6333` (see `db/database.py`) before any
ingestion or retrieval code will work.

Most pipeline scripts are written to be run directly (`python <script>.py`), driven by a
`CONFIG` block near the top or an `if __name__ == "__main__":` block at the bottom — edit that
block rather than adding CLI args. There are no automated tests in this repo.

Key entry points:
- `run_pipeline.py` — end-to-end: load precomputed FinDER dataset → run retrieval/generation
  sweep per ticker → (optionally) analysis + LLM judging. Edit the CONFIG block (tickers,
  output dir, `USE_LLM_JUDGE`, levels, retrieval modes).
- `chunking/sec-pipeline.py` — hierarchical chunking pipeline for raw SEC HTML filings.
- `db/injest_orchestration.py` — drives dense + BM25 ingestion into Qdrant for a batch of
  chunk pickles, across all three chunking levels.
- `evaluation/build_evaluation_dataset.py` — precomputes query enhancement, metadata
  extraction, and embeddings for FinDER questions once, saved to a Parquet dataset so later
  evaluation runs ("lazy" runs) don't repeat expensive LLM/embedding calls.
- `evaluation/run_rag_lazy.py` — the actual multi-config retrieval + generation sweep, reading
  from the precomputed dataset.
- `evaluation/evaluation_run.py` — post-hoc analysis: retrieval metrics, truth-chunk
  resolution, and the pairwise LLM judge (Phi-4) comparing each config's answer against a
  fixed baseline config (`BASELINE_CONFIG_KEY = "L1_BM25_PLAIN_A0"`).
- `evaluation/statistical_tests/` — one-off analysis scripts over saved eval outputs (judge
  tie breakdowns, overlap analysis, section-routing-by-category, year-filter stats).
- `drafts/` — superseded/experimental versions of pipeline stages; not part of the current
  pipeline, kept for reference.

## Architecture

### Chunking (`chunking/`)

SEC 10-K HTML filings are converted to enriched markdown, then chunked hierarchically into
four **levels**, each saved as a separate pickle and linked by ID (`chunking/sec-pipeline.py`,
`chunking/sec_processing.py`):

- **Level 0 (doc)** — one chunk per filing, full markdown text.
- **Level 1 (header)** — header-split chunks; carries `doc_id` → Level 0. This is also the
  **parent-fetch target**: Level 2/3 retrieval hits get expanded back up to their Level 1
  parent chunk before being handed to generation (`use_parent_fetch` in
  `evaluation/run_rag_lazy.py`'s `COLLECTIONS`).
- **Level 2 (child)** — dense ~400-token chunks; carries `parent_id` → Level 1, `doc_id` → Level 0.
- **Level 3 (enriched)** — Level 2 chunks augmented with an LLM-generated (Llama-3.2-3B)
  conceptual `description`; at ingest time `description + text` is embedded instead of `text`
  alone, and enrichment is skipped for chunks below `MIN_TOKENS_FOR_ENRICHMENT`.

`metadata_extractor.py` regex-parses filing-level metadata (form type, fiscal year end,
company name, ticker) directly out of the markdown, independent of any LLM.

### Storage (`db/`, `models.py`)

Each chunking level has a **dense** Qdrant collection (embeddinggemma-300m vectors) and a
**BM25 sparse** collection, both keyed by the same chunk IDs. `models.doc_payload` (pydantic)
is the canonical payload schema — all string fields are lowercased on validation, and most
fields are `Optional` because not every chunking level populates every field (e.g. `section`/
`subsection`/`item`/`subitem`/`run_header` form a header-hierarchy that gets progressively more
specific; `description` only exists at Level 3).

`ingest_dense.py`/`ingest_bm25.py` embed a **formatted title + text** string (see `fmt_title`)
rather than raw text alone — `section`/`subsection` are boilerplate SEC item titles repeated
verbatim across every filing and are deliberately excluded from the embedded text; `item`/
`subitem`/`run_header` are the parts of the header hierarchy that actually carry
discriminative signal. `injest_orchestration.py` is the top-level driver that runs both dense
and BM25 ingestion across all three levels for a batch of new chunk files.

### Retrieval and generation (`search_engine.py`, `evaluation/`)

Retrieval always runs two tracks that get fused:

- **Dense + sparse fusion**: `rrf_fuse`/`rrf_fuse_multi` combine dense (Qdrant vector search)
  and BM25 hits via Reciprocal Rank Fusion for `"hybrid"` mode.
- **Section routing** (`evaluation/section_routing.py`): a query's FinDER `category` (e.g.
  "risk", "financials") maps to expected 10-K Item numbers (`CATEGORY_ITEMS`). Track A is a
  section-filtered retrieval, Track B is the unfiltered baseline; `section_alpha` (0/1 in
  current configs) controls the RRF blend between them (`section_fuse`).
- **Year filtering** is *not* a config toggle — it's fully deterministic
  (`utils.extract_year_deterministic`): explicit years/ranges in the query text always win as
  a flat filter across those years; otherwise retrieval falls back to a recency-weighted
  5-year window anchored at `FINDER_ANCHOR_YEAR = 2024` (`utils.year_weights`,
  `_default_year_window`). Ticker extraction (`extract_ticker_deterministic`) is similarly
  regex/S&P500-alias-table driven rather than LLM-driven — see the long comment there
  explaining why company-name aliasing is hand-curated rather than auto-derived.
- Query rewriting (`evaluation/rag_functions.enhance_query`) is a *separate, optional* LLM step
  that only changes the retrieval query text/embedding — it must never influence the
  ticker/year/form_type metadata filters, which are always extracted from the *original* query.

`evaluation/build_evaluation_dataset.py` precomputes the expensive per-question steps (query
enhancement, metadata extraction, embeddings) once into a Parquet file; `run_rag_lazy.py` then
sweeps many `(level, retrieval_mode, enhance_query_flag, section_alpha)` configs cheaply by
reading from that dataset. Within one sweep, retrieval results are cached and de-duplicated
across configs that turn out to fetch identical chunks (`overlap_groups` in
`run_multi_evaluation_lazy`), and generation is batched once per unique retrieval rather than
once per config — this is a meaningful efficiency pattern to preserve when touching that loop.

`evaluation/evaluation_run.py` (stage 2, post-hoc, run separately from retrieval/generation)
computes retrieval quality metrics (soft MRR/Recall/NDCG via `chunk_relevance`, lexical
word/number recall, hard truth-chunk alignment via `find_truth_chunks`) and runs an LLM judge
(Phi-4, a different model family from the Llama/Qwen generation models, to avoid self-serving
bias) that pairwise-compares each config's answer against the fixed baseline config's answer.
A/B slot order is randomized per comparison (hash-based, deterministic given the same inputs —
`_should_swap`) to mitigate positional bias, then unswapped before being recorded. Both the
eval-generation loop and the judging loop checkpoint after every chunk of work via
write-then-`os.replace` (atomic) so a crash only loses the most recent partial chunk, and are
resumable (`RESUME_FROM` dict, "unjudged rows have null relevance" checks).

### Batched MLX generation

Several modules (`search_engine.generate_llm_answers_batch`,
`evaluation/rag_functions._batch_generate_texts`, `evaluation/evaluation_run.judge_llm_batch`,
`chunking/sec-pipeline.build_enriched_level`) use `mlx_lm.batch_generate` rather than calling
`generate()` once per prompt, so a single Metal GPU processes multiple prompts per forward
pass. `completion_batch_size`/`prefill_batch_size` cap how many sequences' KV caches are held
concurrently (mlx_lm defaults to 32/8, which OOMs) — when generation exhausts GPU memory here,
lower these two, not the logical batch size.

### FinDER dataset access (`FinDER.py`)

`run_finder(tickers)` loads the FinDER parquet, renames columns to `query`/`truth_answer`/
`truth_ref`, filters out rows with no real reference answer, and filters to the given ticker(s)
by substring match on the query text.
