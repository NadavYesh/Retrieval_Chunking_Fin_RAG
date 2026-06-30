#!/usr/bin/env python3
"""
RAG evaluation analysis: retrieval statistics, truth-chunk corpus lookup, generation judging.

Retrieval metrics — three tiers:

  evidence_hit   — soft content coverage over the full merged context (all chunks together).
                   word_recall ≥ 0.50 OR num_recall ≥ 0.70. Available for every row.

  soft_*         — per-chunk soft relevance against truth passages using number overlap
                   and text similarity. Gives credit for "right neighbourhood" retrievals
                   where the exact chunk ID was missed. Soft Recall@k, MRR, NDCG@5.
                   Available for every row (no corpus-lookup dependency).

  hard_*         — exact chunk-ID matching. Only defined when the truth chunk exists in
                   the indexed PKL corpus (n_truth_in_corpus > 0). Hard Recall@k, MRR,
                   NDCG@5. None otherwise.

Generation metrics:
  LLM judge (Phi-4) — relevance YES/NO, completeness YES/NO. Primary signal.
  Lexical recall  — supplementary only (word_recall, num_recall).

Usage:
    python evaluation_run.py path/to/eval_multi_TSLA_YYYYMMDD_HHMM.json
"""

import contextlib
import io
import json
import math
import pickle
import re
import sys
from difflib import SequenceMatcher
from pathlib import Path
import openpyxl

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from evaluation_functions import score_row

# ── Config ────────────────────────────────────────────────────────────────────
# disabled because we run in batches.
# EVAL_FILE  = Path("/Users/nadavsmacbookair/Desktop/Thesis/data/eval_results/WMT_eval_20260626_1603.json")

CHUNKS_DIR = Path("/Users/nadavsmacbookair/Desktop/Thesis/data/financial_corpora/chunks/hierarchical/indexed-at-26-06-26/header")

LOW_SCORE_THRESH      = 0.60   # cosine similarity threshold — only meaningful for dense mode
SCORE_GAP_THRESH      = 0.03   # min gap between best and worst retrieved score; small gap = undiscriminated results
FINGERPRINT_LEN       = 200    # chars of normalised text used as a fast exact-match key in fp_index
FUZZY_THRESH          = 0.80   # SequenceMatcher ratio needed to declare a fuzzy truth-chunk match
NUM_CONTAINMENT_THRESH = 0.85  # fraction of chunk's numbers that must appear in the truth ref
MIN_CHUNK_NUMS        = 4      # minimum distinct numbers a chunk must have to qualify for number-containment


# ── Corpus loading ────────────────────────────────────────────────────────────

def _norm(text: str) -> str:
    """Collapse whitespace and lowercase — used for all text comparisons."""
    return re.sub(r"\s+", " ", str(text).strip()).lower()


_NUM_RE = re.compile(r"-?[\d,]+\.?\d*")

def _numbers(text: str) -> set[str]:
    """Extract distinctive numeric tokens (integers, decimals, comma-formatted)."""
    return set(_NUM_RE.findall(text))


SOFT_RELEVANCE_THRESH = 0.30  # minimum overlap score to label a chunk as soft-relevant


def chunk_relevance(chunk_text: str, truth_passages: list[str]) -> float:
    """
    Soft relevance score [0, 1] for a single retrieved chunk vs the truth passages.

    Score = max over truth passages of max(num_overlap, text_similarity) where:
      num_overlap = |chunk_nums ∩ truth_nums| / |truth_nums|  (truth coverage direction)
      text_similarity = SequenceMatcher ratio on first 500 chars
    """
    chunk_nums = _numbers(chunk_text)
    best = 0.0
    for truth in truth_passages:
        truth_nums = _numbers(truth)
        num_overlap = len(chunk_nums & truth_nums) / len(truth_nums) if truth_nums else 0.0
        text_sim    = SequenceMatcher(None, _norm(chunk_text)[:500], _norm(truth)[:500]).ratio()
        best = max(best, num_overlap, text_sim)
    return best


def _dcg(scores: list[float]) -> float:
    return sum(s / math.log2(i + 2) for i, s in enumerate(scores))


def _ndcg_k(relevance_scores: list[float], k: int) -> float:
    top_k = relevance_scores[:k]
    dcg   = _dcg(top_k)
    idcg  = _dcg(sorted(relevance_scores, reverse=True)[:k])
    return dcg / idcg if idcg > 0 else 0.0


def soft_retrieval_metrics(
    retrieved_ids: list[str],
    corpus_text_map: dict[str, str],
    truth_refs: list[str],
    k_values: tuple = (1, 3, 5),
) -> dict:
    """
    Compute soft retrieval metrics using per-chunk relevance scoring.
    Available for ALL rows (no corpus lookup dependency).
    """
    rel_scores = [
        chunk_relevance(corpus_text_map.get(cid, ""), truth_refs)
        for cid in retrieved_ids
    ]
    binary     = [1 if r >= SOFT_RELEVANCE_THRESH else 0 for r in rel_scores]
    first_hit  = next((i + 1 for i, b in enumerate(binary) if b), None)

    result = {
        "soft_MRR":    1.0 / first_hit if first_hit else 0.0,
        "soft_NDCG@5": _ndcg_k(rel_scores, 5),
    }
    for k in k_values:
        result[f"soft_Recall@{k}"] = int(any(binary[:k]))
    return result


def hard_retrieval_metrics(
    retrieved_ids: list[str],
    truth_corpus_ids: list[list[str]],
    best_truth_rank,
    k_values: tuple = (1, 3, 5),
) -> dict:
    """
    Compute exact chunk-ID retrieval metrics.
    Returns None values when no truth chunks are in corpus.
    """
    truth_flat = {cid for ids in truth_corpus_ids for cid in ids}
    n_truth    = len(truth_flat)
    none_result = {"hard_MRR": None, "hard_NDCG@5": None}
    for k in k_values:
        none_result[f"hard_Recall@{k}"] = None
    if n_truth == 0:
        return none_result

    binary    = [1 if cid in truth_flat else 0 for cid in retrieved_ids]
    result = {
        "hard_MRR":    1.0 / best_truth_rank if best_truth_rank else 0.0,
        "hard_NDCG@5": _ndcg_k(binary, 5),
    }
    for k in k_values:
        n_hit = sum(binary[:k])
        result[f"hard_Recall@{k}"] = n_hit / n_truth
    return result


def load_corpus() -> tuple[pd.DataFrame, dict[str, str], dict[str, str]]:
    """
    Load every PKL chunk file under CHUNKS_DIR into a single DataFrame and build
    lookup structures for truth-chunk resolution and soft relevance scoring.

    Returns
    -------
    corpus : pd.DataFrame
        All chunks concatenated. Columns include at least 'id', 'text', 'metadata',
        'source_file'. One row per indexed chunk.

    fp_index : dict[str, str]
        Maps the first FINGERPRINT_LEN chars of normalised chunk text → chunk UUID.
        Used as the primary (exact) truth-chunk lookup; fuzzy search is the fallback.

    corpus_text_map : dict[str, str]
        Maps chunk UUID → raw text. Used by soft_retrieval_metrics() to compute
        per-chunk relevance against truth passages without re-loading the corpus.
    """
    frames = []
    for p in sorted(CHUNKS_DIR.glob("*.pkl")):
        with open(p, "rb") as f:
            df = pickle.load(f)
        df["source_file"] = p.name
        frames.append(df)
    corpus = pd.concat(frames, ignore_index=True)

    fp_index: dict[str, str] = {}
    corpus_text_map: dict[str, str] = {}
    for _, row in corpus.iterrows():
        fp = _norm(row["text"])[:FINGERPRINT_LEN]
        if fp and fp not in fp_index:
            fp_index[fp] = row["id"]
        corpus_text_map[row["id"]] = row["text"]
    return corpus, fp_index, corpus_text_map


def find_truth_chunks(truth_text: str, corpus: pd.DataFrame, fp_index: dict[str, str]) -> list[str]:
    """
    Locate all corpus chunks contained within a FinDER truth passage.

    FinDER truth passages and corpus chunks use different text formats (tab/newline
    vs markdown pipe tables), so substring matching is unreliable. Two complementary
    strategies are combined:

    Step 1 — Fingerprint / fuzzy (text-based, works for narrative passages):
      Exact match on first 200/100/50 normalised chars, then SequenceMatcher fallback.
      Finds chunks where the text format happens to align (non-table sections).

    Step 2 — Number-containment scan (format-agnostic, works for financial tables):
      Extracts the numeric tokens from the truth passage and from each corpus chunk.
      A chunk is "contained" if ≥ NUM_CONTAINMENT_THRESH of its numbers appear in the
      truth ref AND it has at least MIN_CHUNK_NUMS distinct numbers (to avoid spurious
      matches from chunks that only contain year values like 2023/2024).

    Returns a deduplicated list of chunk UUIDs. Empty list if nothing matches.
    """
    found: set[str] = set()
    norm_truth = _norm(truth_text)

    # ── Step 1a: exact fingerprint ──
    fp = norm_truth[:FINGERPRINT_LEN]
    if fp in fp_index:
        found.add(fp_index[fp])

    # ── Step 1b: shorter prefix fallback ──
    if not found:
        for plen in (100, 50):
            short = norm_truth[:plen]
            for idx_fp, cid in fp_index.items():
                if idx_fp[:plen] == short:
                    found.add(cid)
                    break
            if found:
                break

    # ── Step 2: number-containment scan ──
    truth_nums = _numbers(truth_text)
    if truth_nums:
        for _, row in corpus.iterrows():
            chunk_nums = _numbers(row["text"])
            if len(chunk_nums) < MIN_CHUNK_NUMS:
                continue
            containment = len(chunk_nums & truth_nums) / len(chunk_nums)
            if containment >= NUM_CONTAINMENT_THRESH:
                found.add(row["id"])

    # ── Step 3: fuzzy fallback (only when both steps above found nothing) ──
    if not found:
        for _, row in corpus.iterrows():
            ratio = SequenceMatcher(None, norm_truth[:500], _norm(row["text"])[:500]).ratio()
            if ratio >= FUZZY_THRESH:
                found.add(row["id"])
                break  # take first fuzzy match only

    return list(found)


# ── Parsing rag_retrieved string ──────────────────────────────────────────────

def _re_str(pattern: str, text: str) -> str | None:
    """Return first capturing group of pattern in text, or None."""
    m = re.search(pattern, text)
    return m.group(1).strip() if m else None


def _re_float(pattern: str, text: str) -> float | None:
    """Return first capturing group of pattern as float, or None."""
    m = re.search(pattern, text)
    return float(m.group(1)) if m else None


def parse_retrieved(retrieved_str: str) -> list[dict]:
    """
    Parse the 'rag_retrieved' field written by eval_new.py into a list of source dicts.

    The field is formatted as:
        ======================
        Source Number <n>
         <ScoredPoint repr>

    Each dict contains: chunk_id, score, ticker, fiscal_year_end, section, item.
    score is a raw float — for hybrid/sparse runs it is an RRF score (not cosine similarity),
    so LOW_SCORE comparisons are only valid when mode == 'dense'.
    """
    if not retrieved_str or not isinstance(retrieved_str, str):
        return []
    blocks = re.split(r"={5,}\nSource Number \d+\n?", retrieved_str)
    sources = []
    for block in blocks:
        if not block.strip():
            continue
        chunk_id = _re_str(r"\bid='([^']+)'", block)
        score    = _re_float(r"\bscore=([\d.]+)", block)
        ticker   = _re_str(r"'ticker':\s*'([^']*)'", block)
        fy_end   = _re_str(r"'fiscal_year_end':\s*'(\d{4}-\d{2}-\d{2})", block)
        section  = _re_str(r"'section':\s*'([^']*)'", block)
        item     = _re_str(r"'item':\s*'([^']*)'", block)
        sources.append({
            "chunk_id": chunk_id, "score": score, "ticker": ticker,
            "fiscal_year_end": fy_end, "section": section, "item": item,
        })
    return sources


# ── run_id helpers ────────────────────────────────────────────────────────────

KNOWN_MODES = {"dense", "sparse", "hybrid"}


def parse_run_id(run_id: str) -> tuple[str | None, list[int], str | None]:
    """
    Decode a run_id string into (ticker, years, mode).

    Two formats are supported:
      Old: ticker_year_header_idx          e.g. 'nvda_2024_header_3'
      New: ticker_year_header_mode_idx     e.g. 'nvda_2024_header_hybrid_3'

    years is a list because some queries span multiple fiscal years.
    mode is None for old-format run_ids (assumed dense in callers).
    """
    if "_header_" in run_id:
        prefix, suffix = run_id.split("_header_", 1)
        first_part = suffix.split("_")[0]
        mode = first_part if first_part in KNOWN_MODES else None
    else:
        prefix = run_id
        mode = None

    ticker, _, year_str = prefix.partition("_")
    ticker = None if ticker == "None" else ticker
    years  = [int(y) for y in re.findall(r"\d{4}", year_str)]
    return ticker, years, mode


def fy_to_year(fy_str: str | None) -> int | None:
    """Extract the 4-digit year from a fiscal_year_end string like '2024-01-31'."""
    if not fy_str:
        return None
    m = re.match(r"(\d{4})", fy_str)
    return int(m.group(1)) if m else None


# ── Per-query analysis ────────────────────────────────────────────────────────

def analyze_entry(idx: str, data: dict, corpus: pd.DataFrame, fp_index: dict,
                  corpus_text_map: dict[str, str] | None = None) -> dict:
    """
    Produce a flat analysis record for one eval row.

    Parameters
    ----------
    idx    : string key into the JSON dict (column index as string).
    data   : the full JSON loaded as a column-oriented dict
             (keys = column names, values = {idx: value} dicts).
    corpus : full PKL corpus DataFrame from load_corpus().
    fp_index : fingerprint-to-chunk-id lookup from load_corpus().

    Returned dict keys
    ------------------
    idx, run_id, mode, query         — identity fields
    n_retrieved                      — number of chunks actually returned by the retriever
    max_score, min_score, mean_score, score_gap
                                     — score statistics; only interpretable for mode='dense'
                                       (BM25/RRF scores are unbounded)
    n_wrong_ticker, n_wrong_year     — filter correctness: chunks that belong to the wrong
                                       company or fiscal year
    n_truth_refs                     — number of FinDER ground-truth passages for this question
    n_truth_in_corpus                — how many of those exist in the indexed PKL corpus
    n_truth_retrieved                — how many were actually in the retrieved top-k list
    best_truth_rank                  — 1-based rank of the highest-placed truth chunk (None if missed)
    truth_chunk_ids                  — space-separated corpus UUIDs (or 'NOT_IN_CORPUS')
    word_recall, num_recall          — token-level overlap between truth passages and retrieved context
    evidence_hit                     — True if word_recall ≥ 0.50 OR num_recall ≥ 0.70
    pitfalls                         — comma-separated flags: CROSS_TICKER, WRONG_YEAR,
                                       LOW_SCORE (dense only), SCORE_GAP_SMALL (dense only),
                                       TRUTH_NOT_IN_CORPUS, TRUTH_NOT_RETRIEVED
    """
    finder_id     = data.get("finder_id",    {}).get(idx, "")
    run_id        = data["run_id"].get(idx, "")
    query         = data["query"].get(idx, "")
    truth_refs    = data["truth_ref"].get(idx, [])
    retrieved_str = data["rag_retrieved"].get(idx, "")
    rag_answer    = data.get("rag_answer",   {}).get(idx, "")
    truth_answer  = data.get("truth_answer", {}).get(idx, "")
    category      = data.get("category",     {}).get(idx, "")
    query_type    = data.get("query_type",   {}).get(idx, "")

    # ── Config columns (new multi-config format; None for old single-config JSONs) ──
    config_key           = data.get("config_key",           {}).get(idx)
    enhance_query_flag   = data.get("enhance_query_flag",   {}).get(idx)
    use_tiered_years     = data.get("use_tiered_years",     {}).get(idx)
    section_alpha        = data.get("section_alpha",         {}).get(idx)
    ticker_filter        = data.get("ticker_filter",        {}).get(idx)
    level                = data.get("level",                {}).get(idx)

    if isinstance(truth_refs, str):
        try:
            parsed = json.loads(truth_refs)
            truth_refs = parsed if isinstance(parsed, list) else [truth_refs]
        except Exception:
            truth_refs = [truth_refs]
    if truth_refs is None:
        truth_refs = []

    q_ticker, q_years, mode = parse_run_id(run_id)
    # run_rag_lazy writes "retrieval_mode"; older single-config format used "mode"
    if "retrieval_mode" in data:
        mode = data["retrieval_mode"].get(idx) or mode
    elif "mode" in data:
        mode = data["mode"].get(idx, mode)

    retrieved     = parse_retrieved(retrieved_str)
    retrieved_ids = [s["chunk_id"] for s in retrieved if s["chunk_id"]]

    # ── Score stats ────────────────────────────────────────────────────────────
    # After the rrf_fuse fix, hybrid/sparse scores are true RRF scores (rank-based,
    # bounded by 1/(k+1) per list). Absolute thresholds (LOW_SCORE, SCORE_GAP_SMALL)
    # still only apply to dense (cosine ∈ [0,1]), but std/gap are meaningful for all.
    scores     = [s["score"] for s in retrieved if s["score"] is not None]
    max_score  = max(scores) if scores else None
    min_score  = min(scores) if scores else None
    score_gap  = round(max_score - min_score, 4) if scores else None
    mean_score = round(float(np.mean(scores)), 4) if scores else None
    score_std  = round(float(np.std(scores)), 4) if len(scores) > 1 else None

    # ── Ticker / year correctness ──────────────────────────────────────────────
    # Checks whether the retriever respected the payload filters
    n_wrong_ticker = sum(
        1 for s in retrieved
        if s["ticker"] and q_ticker and s["ticker"].lower() != q_ticker.lower()
    )
    n_wrong_year = 0
    wrong_year_detail = []
    if q_years:
        for s in retrieved:
            fy_year = fy_to_year(s["fiscal_year_end"])
            if fy_year and fy_year not in q_years:
                n_wrong_year += 1
                wrong_year_detail.append(f"{fy_year}∉{q_years}")

    # ── Truth chunk lookup ─────────────────────────────────────────────────────
    # truth_corpus_ids: for each FinDER reference passage, the list of PKL chunk UUIDs
    # that are contained within it (empty list = not in indexed corpus).
    # A passage can map to multiple chunks when the FinDER passage spans several
    # header-level Qdrant chunks (common for financial tables).
    truth_corpus_ids: list[list[str]] = [
        find_truth_chunks(ref, corpus, fp_index) for ref in truth_refs
    ]
    n_truth_in_corpus = sum(len(ids) > 0 for ids in truth_corpus_ids)

    retrieved_id_set  = set(retrieved_ids)
    n_truth_retrieved = sum(
        any(cid in retrieved_id_set for cid in ids)
        for ids in truth_corpus_ids
        if ids
    )

    best_truth_rank = None
    for ids in truth_corpus_ids:
        for cid in ids:
            if cid in retrieved_ids:
                rank = retrieved_ids.index(cid) + 1
                if best_truth_rank is None or rank < best_truth_rank:
                    best_truth_rank = rank

    # ── Word / number recall ───────────────────────────────────────────────────
    # score_row compares the truth_refs text against the full retrieved context string
    coverage = score_row(truth_refs, retrieved_str)

    # ── Filter precision ───────────────────────────────────────────────────────
    n_retrieved_total = len(retrieved)
    year_precision = None
    if q_years and n_retrieved_total > 0:
        n_correct_year = sum(
            1 for s in retrieved
            if fy_to_year(s["fiscal_year_end"]) in q_years
        )
        year_precision = round(n_correct_year / n_retrieved_total, 4)

    ticker_precision = None
    if q_ticker and n_retrieved_total > 0:
        n_correct_ticker = sum(
            1 for s in retrieved
            if s["ticker"] and s["ticker"].lower() == q_ticker.lower()
        )
        ticker_precision = round(n_correct_ticker / n_retrieved_total, 4)

    n_unique_sections = len({s["section"] for s in retrieved if s["section"]})

    # ── Query-side features ────────────────────────────────────────────────────
    query_length      = len(query.split())
    year_is_multi     = int(len(q_years) > 1)
    answer_length     = len(str(rag_answer).split())
    answer_has_number = int(bool(_numbers(str(rag_answer))))

    # ── Soft retrieval metrics ─────────────────────────────────────────────────
    if corpus_text_map is not None and truth_refs:
        soft_metrics = soft_retrieval_metrics(retrieved_ids, corpus_text_map, truth_refs)
    else:
        soft_metrics = {
            "soft_MRR": None, "soft_NDCG@5": None,
            "soft_Recall@1": None, "soft_Recall@3": None, "soft_Recall@5": None,
        }

    # ── Hard retrieval metrics ─────────────────────────────────────────────────
    hard_metrics = hard_retrieval_metrics(retrieved_ids, truth_corpus_ids, best_truth_rank)

    # ── Pitfall flags ──────────────────────────────────────────────────────────
    pitfalls = []
    if n_wrong_ticker > 0:
        pitfalls.append("CROSS_TICKER")
    if q_years and n_wrong_year > 0:
        pitfalls.append("WRONG_YEAR")
    # Score-based flags only make sense for dense (cosine similarity in [0,1])
    if mode in (None, "dense"):
        if max_score is not None and max_score < LOW_SCORE_THRESH:
            pitfalls.append("LOW_SCORE")
        if score_gap is not None and score_gap < SCORE_GAP_THRESH:
            pitfalls.append("SCORE_GAP_SMALL")
    if truth_refs and n_truth_in_corpus == 0:
        pitfalls.append("TRUTH_NOT_IN_CORPUS")
    elif n_truth_in_corpus > 0 and n_truth_retrieved == 0:
        pitfalls.append("TRUTH_NOT_RETRIEVED")

    return {
        # ── Identity ──
        "finder_id":           finder_id,
        "idx":                 int(idx),
        "run_id":              run_id,
        "mode":                mode or "dense",
        "query":               query,
        "category":            category,
        "query_type":          query_type,
        # ── Config columns (None for old-format JSONs) ──
        "config_key":          config_key,
        "level":               level,
        "enhance_query_flag":  enhance_query_flag,
        "use_tiered_years":    use_tiered_years,
        "section_alpha":       section_alpha,
        "ticker_filter":       ticker_filter,
        # ── Retrieval quality ──
        "n_retrieved":         n_retrieved_total,
        "max_score":           round(max_score, 4) if max_score is not None else None,
        "min_score":           round(min_score, 4) if min_score is not None else None,
        "mean_score":          mean_score,
        "score_gap":           score_gap,
        "score_std":           score_std,
        # ── Filter precision ──
        "year_precision":      year_precision,
        "ticker_precision":    ticker_precision,
        "n_unique_sections":   n_unique_sections,
        # ── Filter leakage counts ──
        "n_wrong_ticker":      n_wrong_ticker,
        "n_wrong_year":        n_wrong_year,
        "wrong_year_detail":   "; ".join(wrong_year_detail),
        # ── Truth chunk lookup ──
        "n_truth_refs":        len(truth_refs),
        "n_truth_in_corpus":   n_truth_in_corpus,
        "n_truth_retrieved":   n_truth_retrieved,
        "best_truth_rank":     best_truth_rank,
        "truth_chunk_ids":     " | ".join(
            ", ".join(ids) if ids else "NOT_IN_CORPUS"
            for ids in truth_corpus_ids
        ),
        # ── Soft retrieval metrics (all rows) ──
        **soft_metrics,
        # ── Hard retrieval metrics (None when truth not in corpus) ──
        **hard_metrics,
        # ── Lexical coverage (merged context) ──
        "word_recall":         coverage["word_recall"],
        "num_recall":          coverage["num_recall"],
        "evidence_hit":        coverage["evidence_hit"],
        # ── Query-side features ──
        "query_length":        query_length,
        "year_is_multi":       year_is_multi,
        "answer_length":       answer_length,
        "answer_has_number":   answer_has_number,
        # ── Pitfalls ──
        "pitfalls":            ", ".join(pitfalls) if pitfalls else "—",
        # ── Answers ──
        "rag_answer":          rag_answer,
        "truth_answer":        truth_answer,
    }


# ── Output ────────────────────────────────────────────────────────────────────

def print_per_query_table(rows: list[dict]) -> None:
    """
    Print a fixed-width table with one row per eval entry.

    Columns:
      #         — row index
      mode      — dense / sparse / hybrid
      run_id    — ticker_year_level_mode_idx
      query     — truncated to 35 chars
      hit       — evidence_hit (Y/N)
      wr        — word_recall (0–1)
      nr        — num_recall  (0–1)
      in_corp   — n_truth_in_corpus / n_truth_refs
      rank      — best_truth_rank (1-based; — if not retrieved)
      max_sc    — highest retrieval score (cosine for dense, RRF for hybrid)
      gap       — score_gap shown only for dense; 'n/a' otherwise
      pitfalls  — comma-separated flag codes
    """
    W = 140
    print("\n" + "═" * W)
    print("PER-QUERY RETRIEVAL ANALYSIS")
    print("═" * W)
    print(
        f"{'#':>4}  {'mode':<7}  {'run_id':<28}  {'query':<35}  "
        f"{'hit':>5}  {'wr':>5}  {'nr':>5}  "
        f"{'in_corp':>7}  {'rank':>5}  "
        f"{'max_sc':>8}  {'gap':>6}  pitfalls"
    )
    print("─" * W)

    for r in rows:
        rank_str = str(r["best_truth_rank"]) if r["best_truth_rank"] else "—"
        q_short  = r["query"][:33] + ".." if len(r["query"]) > 35 else r["query"]
        in_corp  = f"{r['n_truth_in_corpus']}/{r['n_truth_refs']}"
        max_sc_str = f"{r['max_score']:>8.4f}" if r["max_score"] is not None else f"{'n/a':>8}"
        gap_str    = f"{r['score_gap']:>6.4f}" if (r["score_gap"] is not None and r["mode"] == "dense") else f"{'n/a':>6}"
        print(
            f"{r['idx']:>4}  {r['mode']:<7}  {r['run_id']:<28}  {q_short:<35}  "
            f"{'Y' if r['evidence_hit'] else 'N':>5}  "
            f"{r['word_recall']:>5.2f}  {r['num_recall']:>5.2f}  "
            f"{in_corp:>7}  {rank_str:>5}  "
            f"{max_sc_str}  {gap_str}  {r['pitfalls']}"
        )


def print_truth_chunk_detail(rows: list[dict]) -> None:
    """
    Print the corpus lookup result for each unique question (deduplicated across modes).

    When the same question is evaluated under multiple modes, the truth-chunk identity
    is identical — only one line is printed per question to avoid repetition.
    Deduplication key: run_id with the mode segment stripped out.
    """
    seen = set()
    print("\n" + "═" * 100)
    print("TRUTH CHUNK CORPUS LOOKUP")
    print("═" * 100)
    for r in rows:
        key = r["run_id"].replace(f"_{r['mode']}_", "_")
        if key in seen:
            continue
        seen.add(key)
        ids    = r["truth_chunk_ids"]
        status = "IN_CORPUS" if "NOT_IN_CORPUS" not in ids else (
            "PARTIAL" if any(c != "NOT_IN_CORPUS" for c in ids.split(" | ")) else "NOT_IN_CORPUS"
        )
        print(f"  {r['run_id']:<35}  {status:<15} {ids[:55]}{'…' if len(ids)>55 else ''}")


def print_summary(rows: list[dict]) -> None:
    """
    Print aggregate statistics across all rows, broken down by retrieval mode.

    Key variables
    -------------
    df    : pd.DataFrame built from rows (one row per eval entry).
            Columns match the keys returned by analyze_entry().

    modes : sorted list of unique mode values present in df['mode'].
            Drives per-section breakdowns. When only one mode is present the
            pairwise win/loss comparison is skipped.

    pivot : (multi-mode only) DataFrame indexed by q_key (question, mode-agnostic)
            with one column per mode, values = evidence_hit boolean.
            Used to count per-question wins: only_dense, only_hybrid, both, neither.

    ref_rows : subset of df for modes[0], used for truth-corpus counts.
               Truth lookup is identical across modes for the same question, so
               counting once from any single mode avoids double-counting.

    Sections printed
    ----------------
    1. Evidence Hit Rate by Mode  — hit%, mean word_recall, mean num_recall per mode
    2. Per-Question Mode Comparison — win/loss matrix (only when len(modes) > 1)
    3. Score Distribution (dense only) — max_score / score_gap statistics
    4. Truth Chunk Presence — corpus coverage + per-mode retrieval rate
    5. Pitfall Breakdown — flag counts per mode
    6. Cross-Ticker / Cross-Year — global filter leakage counts
    """
    df = pd.DataFrame(rows)
    n  = len(df)
    # modes: e.g. ['dense'], ['hybrid'], or ['dense', 'hybrid', 'sparse']
    modes = sorted(df["mode"].unique())

    print("\n" + "═" * 60)
    print("AGGREGATE SUMMARY")
    print("═" * 60)
    print(f"Total rows: {n}  |  Modes present: {modes}\n")

    # ── Per-mode coverage comparison ──────────────────────────────────────────
    print("── Evidence Hit Rate by Mode ───────────────────────")
    for mode in modes:
        sub = df[df["mode"] == mode]   # sub: rows for this mode only
        hit_n = sub["evidence_hit"].sum()
        mn    = len(sub)
        wr    = sub["word_recall"].mean()
        nr    = sub["num_recall"].mean()
        print(f"  {mode:<8}: {hit_n}/{mn}  ({hit_n/mn*100:.1f}%)  "
              f"word_recall={wr:.3f}  num_recall={nr:.3f}")
    print()

    # ── Per-question win/loss matrix (only when multiple modes present) ───────
    if len(modes) > 1:
        print("── Per-Question Mode Comparison (evidence_hit) ─────")
        # q_key strips the mode segment so the same question maps to one row regardless of mode
        df["q_key"] = df["run_id"].apply(
            lambda r: re.sub(r"_(dense|sparse|hybrid)_", "_", r)
        )
        # pivot: rows = questions, columns = modes, values = evidence_hit
        pivot = df.pivot_table(index="q_key", columns="mode", values="evidence_hit", aggfunc="first")
        if len(modes) >= 2:
            for m1, m2 in [(modes[i], modes[j]) for i in range(len(modes)) for j in range(i+1, len(modes))]:
                if m1 in pivot.columns and m2 in pivot.columns:
                    sub = pivot[[m1, m2]].dropna()
                    only_m1 = ((sub[m1] == True) & (sub[m2] == False)).sum()
                    only_m2 = ((sub[m1] == False) & (sub[m2] == True)).sum()
                    both    = ((sub[m1] == True)  & (sub[m2] == True)).sum()
                    neither = ((sub[m1] == False) & (sub[m2] == False)).sum()
                    print(f"  {m1} vs {m2}:  both={both}  only_{m1}={only_m1}  only_{m2}={only_m2}  neither={neither}")
        print()

    # ── Score distribution (dense only — BM25 scores are unbounded) ──────────
    dense_df = df[df["mode"] == "dense"]
    if not dense_df.empty:
        print("── Score Distribution (dense mode only) ────────────")
        ms = dense_df["max_score"].dropna()
        gs = dense_df["score_gap"].dropna()
        if not ms.empty:
            print(f"  max_score  : mean {ms.mean():.4f}  min {ms.min():.4f}  max {ms.max():.4f}")
            print(f"  score_gap  : mean {gs.mean():.4f}  median {gs.median():.4f}")
            low_sc = (ms < LOW_SCORE_THRESH).sum()
            print(f"  queries w/ max_score < {LOW_SCORE_THRESH}: {low_sc}/{len(dense_df)}")
        print()

    # ── Truth chunk presence ──────────────────────────────────────────────────
    # ref_rows: use a single mode to count corpus coverage (truth lookup is mode-agnostic)
    print("── Truth Chunk Presence ────────────────────────────")
    ref_rows    = df[df["mode"] == modes[0]]
    total_truth = ref_rows["n_truth_refs"].sum()
    in_corp     = ref_rows["n_truth_in_corpus"].sum()
    print(f"  truth passages in corpus : {in_corp}/{total_truth}  ({in_corp/max(total_truth,1)*100:.1f}%)")
    for mode in modes:
        sub = df[df["mode"] == mode]
        ret = sub["n_truth_retrieved"].sum()
        vr  = sub["best_truth_rank"].dropna()
        rank_str = f"  mean rank {vr.mean():.1f}" if not vr.empty else ""
        print(f"  retrieved ({mode:<8}): {ret}/{in_corp}  ({ret/max(in_corp,1)*100:.1f}%){rank_str}")
    print()

    # ── Pitfall breakdown ─────────────────────────────────────────────────────
    print("── Pitfall Breakdown ───────────────────────────────")
    for mode in modes:
        sub = df[df["mode"] == mode]
        all_flags: list[str] = []
        for p in sub["pitfalls"]:
            all_flags.extend(f.strip() for f in p.split(",") if f.strip() and f.strip() != "—")
        if all_flags:
            counts = pd.Series(all_flags).value_counts()
            flag_str = "  ".join(f"{flag}={cnt}" for flag, cnt in counts.items())
            print(f"  {mode:<8}: {flag_str}")
        else:
            print(f"  {mode:<8}: none")
    print()

    # ── Cross-ticker / cross-year ──────────────────────────────────────────────
    print("── Cross-Ticker / Cross-Year ───────────────────────")
    ct = (df["n_wrong_ticker"] > 0).sum()
    cy = (df["n_wrong_year"]   > 0).sum()
    print(f"  cross-ticker retrieval : {ct}/{n}")
    print(f"  cross-year  retrieval  : {cy}/{n}")


# ── Answer judges ─────────────────────────────────────────────────────────────

def judge_lexical(rag_answer: str, truth_answer: str) -> dict:
    """
    Lexical overlap between the generated answer and the ground truth answer.

    Calls score_row with truth_answer as the reference and rag_answer as the context.
    This measures recall from the truth's perspective: of all words/numbers in the
    ground truth, how many appear in the generated answer? Does not penalise extra
    content in rag_answer (no precision term).

    Returns lex_word_recall, lex_num_recall, lex_hit (thresholds: word≥0.50 OR num≥0.70).
    Fast and unbiased — no LLM required.
    """
    result = score_row([truth_answer], rag_answer)
    return {
        "lex_word_recall": result["word_recall"],
        "lex_num_recall":  result["num_recall"],
        "lex_hit":         result["evidence_hit"],
    }


def judge_llm(question: str, truth_answer: str, rag_answer: str, model, tokenizer) -> dict:
    """
    LLM-based judge using JUDGE_PROMPT. Scores the generated answer on relevance and
    completeness (each YES/NO) relative to the ground truth.

    Uses Phi-4 (`mlx-community/phi-4-4bit`) — a different architecture and training
    from Llama-3.2-3B (the generation model) — which eliminates self-serving bias.
    Runs post-hoc from saved eval JSON; no concurrency conflict with generation.

    Returns llm_relevance, llm_completeness (YES/NO strings), and llm_response (full
    raw model output for inspection).
    """
    sys.path.append(str(Path(__file__).resolve().parent.parent))
    from prompts import JUDGE_PROMPT
    from mlx_lm import generate

    prompt_text = JUDGE_PROMPT.format(
        question=question,
        answer_ref=truth_answer,
        answer_a=rag_answer,
    )
    messages  = [{"role": "user", "content": prompt_text}]
    formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    response  = generate(model, tokenizer, prompt=formatted, verbose=False, max_tokens=300)

    try:
        m = re.search(r'\{[^}]+\}', response, re.DOTALL)
        if m:
            parsed = json.loads(m.group())
            return {
                "llm_relevance":    parsed.get("relevance",    "?"),
                "llm_completeness": parsed.get("completeness", "?"),
                "llm_response":     response.strip(),
            }
    except Exception:
        pass
    return {
        "llm_relevance":    "parse_error",
        "llm_completeness": "parse_error",
        "llm_response":     response.strip(),
    }


def print_judge_summary(rows: list[dict], use_llm_judge: bool) -> None:
    """
    Print answer-quality summary.

    Lexical section: always printed. Shows lex_hit rate, mean word/num recall per mode.
    LLM section: only printed when use_llm_judge=True. Shows YES rate for relevance
    and completeness per mode.
    """
    df    = pd.DataFrame(rows)
    modes = sorted(df["mode"].unique())

    print("\n" + "═" * 60)
    print("ANSWER QUALITY SUMMARY")
    print("═" * 60)

    print("── Lexical Judge (answer vs truth_answer) ──────────")
    for mode in modes:
        sub = df[df["mode"] == mode]
        hit_n = sub["lex_hit"].sum()
        mn    = len(sub)
        wr    = sub["lex_word_recall"].mean()
        nr    = sub["lex_num_recall"].mean()
        print(f"  {mode:<8}: lex_hit={hit_n}/{mn} ({hit_n/mn*100:.1f}%)  "
              f"word_recall={wr:.3f}  num_recall={nr:.3f}")
    print()

    if use_llm_judge:
        print("── LLM Judge / Phi-4 (relevance & completeness) ───")
        for mode in modes:
            sub      = df[df["mode"] == mode]
            mn       = len(sub)
            rel_yes  = (sub["llm_relevance"]    == "YES").sum()
            comp_yes = (sub["llm_completeness"] == "YES").sum()
            parse_err = (sub["llm_relevance"] == "parse_error").sum()
            print(f"  {mode:<8}: relevance={rel_yes}/{mn} ({rel_yes/mn*100:.1f}%)  "
                  f"completeness={comp_yes}/{mn} ({comp_yes/mn*100:.1f}%)"
                  + (f"  [parse_errors={parse_err}]" if parse_err else ""))
        print()


# ── Core analysis (callable directly for batch runs) ─────────────────────────

def run_analysis(
    eval_path:    Path,
    corpus:       pd.DataFrame,
    fp_index:     dict,
    corpus_text_map: dict[str, str] | None = None,
    use_llm_judge: bool = False,
    judge_model   = None,
    judge_tok     = None,
) -> None:
    """
    Analyse one eval JSON file and save an Excel + TXT report alongside it.

    Accepts pre-loaded corpus and judge model so batch callers can load them
    once and reuse across multiple eval files.
    """
    print(f"\nEval file : {eval_path}")
    print(f"LLM judge : {'Phi-4 (enabled)' if use_llm_judge else 'disabled'}")
    with open(eval_path) as f:
        data = json.load(f)

    indices = sorted(data["run_id"].keys(), key=int)
    print(f"  Analyzing {len(indices)} rows …")

    rows = []
    for idx in indices:
        row = analyze_entry(idx, data, corpus, fp_index, corpus_text_map)
        rows.append(row)
        sys.stdout.write(f"\r  [{int(idx)+1:>3}/{len(indices)}] {row['run_id']:<40} mode={row['mode']}")
        sys.stdout.flush()
    print()

    # ── Lexical judge — always run (free, unbiased) ──
    for row in rows:
        row.update(judge_lexical(row.get("rag_answer", ""), row.get("truth_answer", "")))

    # ── LLM judge — optional, post-hoc, uses Phi-4 ──
    if use_llm_judge and judge_model is not None:
        print(f"  Judging {len(rows)} rows with Phi-4…")
        for i, row in enumerate(rows):
            scores = judge_llm(
                row["query"],
                row.get("truth_answer", ""),
                row.get("rag_answer", ""),
                judge_model, judge_tok,
            )
            row.update(scores)
            sys.stdout.write(f"\r  [{i+1:>3}/{len(rows)}] judged")
            sys.stdout.flush()
        print()

    # ── Print full detail to stdout, capture only summary sections for txt ──
    print_per_query_table(rows)
    print_truth_chunk_detail(rows)

    summary_parts = []

    def _capture(fn, *args, **kwargs):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            fn(*args, **kwargs)
        text = buf.getvalue()
        print(text, end="")
        summary_parts.append(text)

    _capture(print_summary,      rows)
    _capture(print_judge_summary, rows, use_llm_judge)

    out_xlsx = eval_path.parent / "analysis" / eval_path.name.replace("eval_", "analysis_").replace(".json", ".xlsx")
    out_txt  = out_xlsx.with_suffix(".txt")
    out_pkl  = out_xlsx.with_suffix(".pkl")
    out_xlsx.parent.mkdir(parents=True, exist_ok=True)

    rows_df = pd.DataFrame(rows)
    rows_df.to_excel(out_xlsx, index=False)
    rows_df.to_pickle(out_pkl)
    out_txt.write_text("".join(summary_parts), encoding="utf-8")

    print(f"\nSaved → {out_xlsx}")
    print(f"Saved → {out_pkl}")
    print(f"Saved → {out_txt}")


# ── Multi-config analysis ─────────────────────────────────────────────────────

def _write_multi_excel(rows_df: pd.DataFrame, out_path: Path) -> None:
    """Write 5-sheet Excel from a multi-config result DataFrame."""
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # ── Boolean/int casts for numeric aggregation ──────────────────────────
    df = rows_df.copy()
    for col in ("enhance_query_flag", "use_tiered_years", "year_is_multi",
                "evidence_hit", "answer_has_number"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    for col in ("llm_relevance", "llm_completeness"):
        if col in df.columns:
            df[col + "_int"] = (df[col] == "YES").astype(float)

    group_col = "config_key" if "config_key" in df.columns and df["config_key"].notna().any() else "mode"

    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        # ── Sheet 1: detail ─────────────────────────────────────────────────
        df.to_excel(writer, sheet_name="detail", index=False)

        # ── Sheet 2: config_summary ─────────────────────────────────────────
        agg_dict: dict = {
            "finder_id": "count",
            "evidence_hit": "mean",
            "word_recall": "mean",
            "num_recall": "mean",
            "soft_MRR": "mean",
            "soft_Recall@3": "mean",
            "soft_NDCG@5": "mean",
            "hard_MRR": "mean",
            "hard_Recall@3": "mean",
            "hard_NDCG@5": "mean",
            "year_precision": "mean",
            "ticker_precision": "mean",
            "n_truth_in_corpus": "sum",
        }
        for col in ("llm_relevance_int", "llm_completeness_int"):
            if col in df.columns:
                agg_dict[col] = "mean"
        for col in ("level", "mode", "enhance_query_flag", "use_tiered_years", "section_alpha"):
            if col in df.columns:
                agg_dict[col] = "first"

        cfg_sum = df.groupby(group_col, dropna=False).agg(agg_dict).reset_index()
        cfg_sum.rename(columns={"finder_id": "n_rows"}, inplace=True)

        sort_col = "llm_relevance_int" if "llm_relevance_int" in cfg_sum.columns else "evidence_hit"
        cfg_sum = cfg_sum.sort_values(sort_col, ascending=False)
        cfg_sum.to_excel(writer, sheet_name="config_summary", index=False)

        # ── Sheet 3: category_breakdown ─────────────────────────────────────
        cat_cols = ["category", group_col]
        cat_agg: dict = {"finder_id": "count", "evidence_hit": "mean", "soft_MRR": "mean",
                         "soft_Recall@3": "mean"}
        for col in ("llm_relevance_int", "llm_completeness_int"):
            if col in df.columns:
                cat_agg[col] = "mean"
        if "category" in df.columns:
            cat_bd = df.groupby(cat_cols, dropna=False).agg(cat_agg).reset_index()
            cat_bd.rename(columns={"finder_id": "n_rows"}, inplace=True)
            cat_bd.to_excel(writer, sheet_name="category_breakdown", index=False)

        # ── Sheet 4: failure_analysis ────────────────────────────────────────
        if "llm_relevance" in df.columns:
            fail_mask = (df["llm_relevance"] == "NO") | (df["llm_completeness"] == "NO")
        else:
            fail_mask = df["evidence_hit"] == 0
        fail_df = df[fail_mask].copy()
        if "category" in fail_df.columns:
            fail_df = fail_df.sort_values(["category", group_col])
        fail_df.to_excel(writer, sheet_name="failure_analysis", index=False)

        # ── Sheet 5: correlations ────────────────────────────────────────────
        corr_cols = [c for c in [
            "query_length", "year_is_multi", "n_retrieved", "score_std",
            "enhance_query_flag", "use_tiered_years", "section_alpha",
            "llm_relevance_int", "llm_completeness_int",
            "evidence_hit", "soft_MRR", "soft_Recall@3", "soft_NDCG@5",
            "hard_MRR", "hard_Recall@3",
            "year_precision", "ticker_precision",
        ] if c in df.columns]
        corr_df = df[corr_cols].apply(pd.to_numeric, errors="coerce").corr()
        corr_df.to_excel(writer, sheet_name="correlations")

        # ── Sheet 6: dimension_summary ───────────────────────────────────────
        # For each config axis (level, mode, section_alpha, …), show average metrics
        # across ALL rows with that dimension value — marginalising over everything else.
        # Makes it easy to answer "does section_alpha=0.5 beat 0.0 overall?" without
        # manually filtering the detail sheet.
        dim_axes    = [c for c in ["level", "mode", "section_alpha", "enhance_query_flag", "use_tiered_years"] if c in df.columns]
        metric_cols = [c for c in [
            "evidence_hit", "word_recall", "num_recall",
            "soft_MRR", "soft_Recall@3", "soft_NDCG@5",
            "hard_MRR", "hard_Recall@3",
            "llm_relevance_int", "llm_completeness_int",
        ] if c in df.columns]
        dim_frames = []
        for dim in dim_axes:
            grp   = df.groupby(dim, dropna=False)
            agg_m = grp[metric_cols].mean().round(4)
            agg_m.insert(0, "n_rows", grp.size())
            agg_m.insert(0, "value",  agg_m.index.astype(str))
            agg_m.insert(0, "dimension", dim)
            dim_frames.append(agg_m.reset_index(drop=True))
        if dim_frames:
            pd.concat(dim_frames, ignore_index=True).to_excel(
                writer, sheet_name="dimension_summary", index=False
            )

    print(f"Saved → {out_path}")


def run_multi_analysis(
    eval_path:   Path,
    corpus:      pd.DataFrame,
    fp_index:    dict,
    corpus_text_map: dict[str, str],
    judge_model  = None,
    judge_tok    = None,
) -> pd.DataFrame:
    """
    Primary entry point for multi-config eval JSONs produced by run_rag_lazy.py.

    Runs analyze_entry for every row, applies LLM judge (if models provided),
    writes a 5-sheet Excel, and returns the rows DataFrame.
    """
    print(f"\nEval file : {eval_path}")
    with open(eval_path) as f:
        data = json.load(f)

    indices = sorted(data["run_id"].keys(), key=int)
    print(f"  Analyzing {len(indices)} rows …")

    rows: list[dict] = []
    for idx in indices:
        row = analyze_entry(idx, data, corpus, fp_index, corpus_text_map)
        rows.append(row)
        cfg = row.get("config_key") or row.get("mode", "")
        sys.stdout.write(f"\r  [{int(idx)+1:>3}/{len(indices)}] {cfg:<45}")
        sys.stdout.flush()
    print()

    for row in rows:
        row.update(judge_lexical(row.get("rag_answer", ""), row.get("truth_answer", "")))

    if judge_model is not None:
        print(f"  Judging {len(rows)} rows with Phi-4…")
        for i, row in enumerate(rows):
            row.update(judge_llm(
                row["query"], row.get("truth_answer", ""), row.get("rag_answer", ""),
                judge_model, judge_tok,
            ))
            sys.stdout.write(f"\r  [{i+1:>3}/{len(rows)}] judged")
            sys.stdout.flush()
        print()

    rows_df = pd.DataFrame(rows)

    out_dir  = eval_path.parent / "analysis"
    out_xlsx = out_dir / eval_path.name.replace("eval_", "analysis_").replace(".json", ".xlsx")
    out_pkl  = out_xlsx.with_suffix(".pkl")
    _write_multi_excel(rows_df, out_xlsx)
    rows_df.to_pickle(out_pkl)
    print(f"Saved → {out_pkl}")

    _print_multi_summary(rows_df)
    return rows_df


def _print_multi_summary(df: pd.DataFrame) -> None:
    """Print ranked config table + top failure categories."""
    group_col = "config_key" if "config_key" in df.columns and df["config_key"].notna().any() else "mode"
    n = len(df)

    print("\n" + "═" * 80)
    print("MULTI-CONFIG SUMMARY")
    print("═" * 80)
    print(f"Total rows: {n}  |  Configs: {df[group_col].nunique()}")

    for col in ("evidence_hit", "soft_MRR", "soft_Recall@3"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    for col in ("llm_relevance", "llm_completeness"):
        if col in df.columns:
            df[col + "_int"] = (df[col] == "YES").astype(float)

    agg: dict = {"finder_id": "count", "evidence_hit": "mean", "word_recall": "mean",
                 "num_recall": "mean", "soft_MRR": "mean", "soft_Recall@3": "mean"}
    for c in ("llm_relevance_int", "llm_completeness_int"):
        if c in df.columns:
            agg[c] = "mean"
    ranked = df.groupby(group_col).agg(agg).reset_index()
    ranked.rename(columns={"finder_id": "n"}, inplace=True)
    sort_col = "llm_relevance_int" if "llm_relevance_int" in ranked else "evidence_hit"
    ranked = ranked.sort_values(sort_col, ascending=False)

    print(f"\n{'config_key':<55} {'n':>4}  {'llm_rel%':>8}  {'llm_comp%':>9}  {'soft_MRR':>8}  {'Recall@3':>8}  {'evi_hit%':>8}  {'word_rec':>8}  {'num_rec':>7}")
    print("─" * 125)
    for _, r in ranked.iterrows():
        rel  = f"{r.get('llm_relevance_int', float('nan'))*100:.1f}" if "llm_relevance_int" in r and pd.notna(r.get("llm_relevance_int")) else "  n/a"
        comp = f"{r.get('llm_completeness_int', float('nan'))*100:.1f}" if "llm_completeness_int" in r and pd.notna(r.get("llm_completeness_int")) else "  n/a"
        mrr  = f"{r['soft_MRR']:.3f}" if pd.notna(r.get("soft_MRR")) else "  n/a"
        rc3  = f"{r.get('soft_Recall@3', float('nan')):.3f}" if pd.notna(r.get("soft_Recall@3")) else "  n/a"
        hit  = f"{r['evidence_hit']*100:.1f}" if pd.notna(r.get("evidence_hit")) else "  n/a"
        wr   = f"{r['word_recall']:.3f}" if pd.notna(r.get("word_recall")) else "  n/a"
        nr   = f"{r['num_recall']:.3f}" if pd.notna(r.get("num_recall")) else "  n/a"
        print(f"  {str(r[group_col]):<53} {int(r['n']):>4}  {rel:>8}  {comp:>9}  {mrr:>8}  {rc3:>8}  {hit:>8}  {wr:>8}  {nr:>7}")

    if "category" in df.columns and "llm_relevance" in df.columns:
        fail = df[df["llm_relevance"] == "NO"]
        if not fail.empty:
            top_fail = fail.groupby("category").size().sort_values(ascending=False).head(5)
            print(f"\nTop failure categories (llm_relevance=NO):")
            for cat, cnt in top_fail.items():
                total_cat = (df["category"] == cat).sum()
                print(f"  {cat:<40} {cnt}/{total_cat}")

    if "n_truth_in_corpus" in df.columns:
        hard_rows = df[df["n_truth_in_corpus"].fillna(0) > 0]
        print(f"\nHard eval coverage: {len(hard_rows)}/{n} rows have truth chunk(s) in corpus")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    """
    CLI entry point. Accepts a positional path to a multi-config eval JSON.
    Loads corpus once; optionally loads Phi-4 when --judge flag is passed.

    Usage:
        python evaluation_run.py path/to/eval_multi_TSLA_*.json [--judge]
    """
    use_llm_judge = "--judge" in sys.argv
    positional    = [a for a in sys.argv[1:] if not a.startswith("--")]
    eval_path     = Path(positional[0])

    print(f"Corpus dir: {CHUNKS_DIR}")
    corpus, fp_index, corpus_text_map = load_corpus()
    print(f"  Loaded {len(corpus)} chunks from {corpus['source_file'].nunique()} PKL files")

    judge_model, judge_tok = None, None
    if use_llm_judge:
        from mlx_lm import load as mlx_load
        print("\nLoading Phi-4 judge model…")
        judge_model, judge_tok = mlx_load("mlx-community/phi-4-4bit")

    run_multi_analysis(eval_path, corpus, fp_index, corpus_text_map, judge_model, judge_tok)


if __name__ == "__main__":
    # ── Batch mode: define multi-config eval files to analyse in sequence ─────
    # Corpus and judge model are loaded ONCE and reused across all files.
    # To run a single file from the CLI:
    #   python evaluation_run.py path/to/eval_multi_TSLA_*.json [--judge]

    USE_LLM_JUDGE = False   # set True to enable Phi-4 judging for all runs

    eval_files = [
        "/Users/nadavsmacbookair/Desktop/Thesis/data/eval_results/eval_multi_WMT_20260628_1834.json",
    ]

    if not eval_files:
        main()
    else:
        print(f"Corpus dir: {CHUNKS_DIR}")
        corpus, fp_index, corpus_text_map = load_corpus()
        print(f"  Loaded {len(corpus)} chunks from {corpus['source_file'].nunique()} PKL files")

        judge_model, judge_tok = None, None
        if USE_LLM_JUDGE:
            from mlx_lm import load as mlx_load
            print("\nLoading Phi-4 judge model…")
            judge_model, judge_tok = mlx_load("mlx-community/phi-4-4bit")

        for i, ef in enumerate(eval_files):
            print(f"\n{'='*60}\nBatch {i+1}/{len(eval_files)}: {Path(ef).name}\n{'='*60}")
            run_multi_analysis(Path(ef), corpus, fp_index, corpus_text_map, judge_model, judge_tok)
