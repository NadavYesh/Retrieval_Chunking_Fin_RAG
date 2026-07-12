# Retrieval and Chunking Taxonomy for an Open-Weight Financial RAG System

Master's thesis project (University of Amsterdam) implementing a retrieval-augmented
generation (RAG) pipeline over SEC 10-K filings, evaluated against the
[FinDER](https://huggingface.co/datasets/Linq-AI-Research/FinDER) benchmark. Everything runs
locally on Apple Silicon via [MLX](https://github.com/ml-explore/mlx) — no external LLM APIs.

See [`CITATION.bib`](CITATION.bib) if you use this work.

## Requirements

No `requirements.txt` is checked in — dependencies live in an external env. You'll need at
least:

- `pandas`
- `mlx`, `mlx_lm`, `mlx_embeddings`
- `qdrant_client`
- `langchain_text_splitters`, `tiktoken`
- `sentence_transformers`
- `openpyxl`
- `scipy`
- `edgartools` (for downloading filings)
- `kagglehub` (one-time, to fetch the S&P 500 reference CSV — see below)

`pickle` is used throughout but is part of the Python standard library.

Qdrant must be running locally before any ingestion or retrieval code will work:

```bash
docker run -p 6333:6333 qdrant/qdrant
```

### Reference data

Ticker/company-name resolution (`utils.py`) depends on a static S&P 500 constituent snapshot
that is *not* fetched at runtime. Pull it once via `kagglehub` and place it at
`data/reference/sp500_companies.csv`:

```python
# one-time setup, not run by any pipeline script
import kagglehub
path = kagglehub.dataset_download("andrewmvd/sp-500-stocks")
# copy the CSV out of `path` to data/reference/sp500_companies.csv
```

### Data layout

Data lives outside this repo (`data/` is gitignored) under a sibling `data/` directory. Most
scripts hardcode absolute paths into it and are driven by editing a `CONFIG`-style block near
the top of the file or in `if __name__ == "__main__":`, rather than CLI args. Check those
blocks before running anything — the paths below assume `data/` sits next to this repo's `code/`
directory, matching the paths currently hardcoded in the scripts.

## Pipeline: HTML filings → evaluation results

### 1. Download 10-K filings

```bash
python download_reports/download-10_k-script.py
```

Pulls 10-K HTML filings from EDGAR (via `edgartools`) for S&P 500 tickers listed in
`data/reference/sp500_companies.csv`, saving to `data/html/`. Edit `FILINGS_PER_TICKER` /
`exclude_names` at the top of the script to change scope. This script is a research utility,
not hardened for unattended full-universe runs — expect to babysit it.

### 2. Chunk filings

```bash
python chunking/sec-pipeline.py
```

Converts raw HTML into enriched markdown, then chunks hierarchically into four levels, each
saved as a separate pickle:

- **Level 0 (doc)** — one chunk per filing.
- **Level 1 (header)** — header-split chunks; parent-fetch target for Level 2/3 hits.
- **Level 2 (child)** — dense ~400-token chunks.
- **Level 3 (enriched)** — Level 2 chunks + an LLM-generated conceptual description
  (Llama-3.2-3B via MLX).

Edit `RAW_DIR`, `BASE_CHUNKS`, `BASE_MD`, and `RUN_ENRICHED` (set `False` to skip the
Level 3 LLM enrichment pass) in the `if __name__ == "__main__":` block.

### 3. Ingest into Qdrant

With Qdrant running:

```bash
python db/injest_orchestration.py
```

Reads chunk pickles per level from the hardcoded `paths` list and calls `db/ingest_dense.py`
(embeddinggemma-300m dense vectors) and `db/ingest_bm25.py` (sparse) to populate one dense +
one BM25 collection per level. Edit `paths` and the `COLL_NAME_DENSE`/`COLL_NAME_SPARSE`
collection names before each new ingestion batch.

### 4. Precompute the FinDER evaluation dataset

```bash
python evaluation/build_evaluation_dataset.py
```

Loads FinDER questions via `FinDER.run_finder(tickers)` (reads `data/FinDER/train.parquet`) and
precomputes the expensive per-question steps once — query enhancement, metadata extraction,
embeddings — saving the result to a Parquet dataset (`DATASET_PATH` in
`evaluation/build_evaluation_dataset.py`). Loads a Qwen3.5-9B generation model and
embeddinggemma via MLX. Edit `tickers` in the `if __name__ == "__main__":` block.

### 5. Run the retrieval + generation sweep

```bash
python evaluation/run_rag_lazy.py
```

Sweeps configs (chunk level × retrieval mode × query-enhancement flag × section-routing alpha,
generated via `write_config`) against the precomputed dataset from step 4, reading from
`DATASET_PATH`. Retrieval is cached and de-duplicated across configs that fetch identical
chunks, and generation is batched once per unique retrieval. Outputs one merged JSON per ticker
to `data/eval_results/`. Edit the `configs` list and `RESUME_FROM` (per-ticker resume point for
interrupted runs) in the `if __name__ == "__main__":` block.

Year filtering is *not* a config toggle — it's deterministic per query
(`utils.extract_year_deterministic`): an explicit year/range in the query text always wins as a
flat filter; otherwise retrieval falls back to a recency-weighted 5-year window anchored at
2024.

### 6. Evaluate results

```bash
python evaluation/evaluation_run.py
```

Post-hoc analysis over the JSON output(s) from step 5, listed in `eval_files` in `main()`.
Computes:

- **Retrieval**: `soft_MRR`, `soft_Recall@3` — per-chunk soft relevance (graded overlap vs.
  truth passages, thresholded to binary) scored against every row, independent of corpus
  coverage.
- **Generation**: an LLM judge (Phi-4 — a different model family from the Llama/Qwen
  generation models, to avoid self-serving bias) pairwise-compares each config's answer
  against a fixed baseline config's answer (`BASELINE_CONFIG_KEY = "L1_BM25_PLAIN_A0"`),
  producing relevance/completeness win rates. A/B slot order is randomized per comparison to
  mitigate positional bias.

Set `USE_LLM_JUDGE = True` in `main()` to run the judge inline (loads Phi-4 via MLX); leave it
`False` for retrieval-only analysis. Both the eval-generation loop (step 5) and the judging
loop here checkpoint after every chunk of work and are resumable.

`evaluation/statistical_tests/` holds one-off analysis scripts over saved eval outputs (judge
tie breakdowns, retrieval overlap analysis, section-routing-by-category, year-filter stats).

## Repository layout

```
chunking/           HTML → markdown → hierarchical chunk pickles
db/                 Qdrant ingestion (dense + BM25), per chunk level
evaluation/         FinDER dataset prep, retrieval/generation sweep, post-hoc analysis
  statistical_tests/  one-off analyses over saved eval outputs
download_reports/   EDGAR 10-K download script
drafts/             superseded/experimental pipeline versions (gitignored, not in repo)
models.py           canonical Qdrant payload schema (pydantic)
search_engine.py    retrieval fusion (RRF) + batched MLX generation
utils.py            deterministic ticker/year extraction
FinDER.py           FinDER benchmark loader/filter
prompts.py          all LLM prompts (query enhancement, generation, judging)
```
