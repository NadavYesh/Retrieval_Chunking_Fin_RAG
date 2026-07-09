#!/usr/bin/env python3
"""
RAG evaluation analysis: retrieval statistics, truth-chunk corpus lookup, generation judging.

Retrieval metrics:

  soft_MRR / soft_Recall@3 / soft_NDCG@5
      Per-chunk soft relevance: max(number_overlap, text_similarity) vs truth passages,
      threshold 0.30 for binary. Gives credit for "right neighbourhood" retrievals where
      chunk boundaries don't align with FinDER passages. Available for every row.

  evidence_hit / word_recall / num_recall
      Lexical coverage of truth passages over the full merged retrieved context.
      Coarser than soft metrics (no ranking) but robust to any chunking scheme.

  n_truth_in_corpus / n_truth_retrieved / best_truth_rank
      Raw ground-truth alignment counts. Hard exact-ID metrics are omitted: corpus coverage
      is ~30-40% and the truth→corpus resolution pipeline is itself approximate, making
      hard IR metrics noisy and biased toward a non-representative subset.

Generation metrics:
  LLM judge (Phi-4) — relevance YES/NO, completeness YES/NO. Primary signal.
  Lexical (lex_*) — answer vs truth_answer word/number overlap. Fast supplementary check.

Usage:
    python evaluation_run.py path/to/eval_multi_TSLA_YYYYMMDD_HHMM.json
"""

import contextlib
import hashlib
import io
import json
import math
import pickle
import re
import sys
from difflib import SequenceMatcher
from pathlib import Path
import openpyxl

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from evaluation_functions import score_row

CHUNKS_DIR = Path("/Users/nadavsmacbookair/Desktop/Thesis/data/financial_corpora/chunks/hierarchical/indexed-at-07-07-26/header")

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
    k_values: tuple = (3,),
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
    for p in sorted(CHUNKS_DIR.glob("*/*.pkl")):
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

    Each dict contains: chunk_id, ticker, fiscal_year_end, section, subsection, item.
    """
    if not retrieved_str or not isinstance(retrieved_str, str):
        return []
    blocks = re.split(r"={5,}\nSource Number \d+\n?", retrieved_str)
    sources = []
    for block in blocks:
        if not block.strip():
            continue
        chunk_id   = _re_str(r"\bid='([^']+)'", block)
        ticker     = _re_str(r"'ticker':\s*'([^']*)'", block)
        fy_end     = _re_str(r"'fiscal_year_end':\s*'(\d{4}-\d{2}-\d{2})", block)
        section    = _re_str(r"'section':\s*'([^']*)'", block)
        subsection = _re_str(r"'subsection':\s*'([^']*)'", block)
        item       = _re_str(r"'item':\s*'([^']*)'", block)
        sources.append({
            "chunk_id": chunk_id, "ticker": ticker,
            "fiscal_year_end": fy_end, "section": section,
            "subsection": subsection, "item": item,
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
    n_retrieved                      — number of chunks returned by the retriever
    n_unique_sections                — distinct (section, subsection) pairs in the top-k
    top_section                      — modal (section, subsection) pair
    n_wrong_year                     — chunks from the wrong fiscal year
    n_truth_refs                     — number of FinDER ground-truth passages for this question
    n_truth_in_corpus                — how many of those exist in the indexed PKL corpus
    n_truth_retrieved                — how many were actually in the retrieved top-k list
    best_truth_rank                  — 1-based rank of the highest-placed truth chunk (None if missed)
    truth_chunk_ids                  — corpus UUIDs (or 'NOT_IN_CORPUS')
    soft_MRR, soft_Recall@3, soft_NDCG@5 — per-chunk soft relevance metrics (all rows)
    word_recall, num_recall          — token-level overlap between truth passages and retrieved context
    evidence_hit                     — True if word_recall ≥ 0.50 OR num_recall ≥ 0.70
    pitfalls                         — comma-separated flags: WRONG_YEAR,
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

    # ── Clean / human-readable config labels ──────────────────────────────────
    _lm       = re.search(r"level (\d)", str(level or ""), re.IGNORECASE) or \
                re.search(r"L(\d)",      str(config_key or ""))
    level_num = int(_lm.group(1)) if _lm else None
    enhanced  = ("YES" if enhance_query_flag else "NO") if enhance_query_flag is not None else None
    tiered    = ("YES" if use_tiered_years   else "NO") if use_tiered_years   is not None else None

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

    # Modal (section, subsection) pair across retrieved chunks — used for section routing analysis.
    # A distinct "section" is defined as a unique (section, subsection) combination,
    # because the payload distributes location across section / subsection / item / subitem.
    _sec_counts: dict[tuple, int] = {}
    for s in retrieved:
        key = (s.get("section") or "", s.get("subsection") or "")
        if key != ("", ""):
            _sec_counts[key] = _sec_counts.get(key, 0) + 1
    _top_key    = max(_sec_counts, key=_sec_counts.get) if _sec_counts else None
    top_section = "/".join(filter(None, _top_key)) if _top_key else None

    # ── Year filter correctness ────────────────────────────────────────────────
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

    n_retrieved_total = len(retrieved)

    # Count distinct (section, subsection) pairs — each unique pair is one logical section.
    n_unique_sections = len({
        (s.get("section") or "", s.get("subsection") or "")
        for s in retrieved
        if s.get("section") or s.get("subsection")
    })

    # ── Query-side features ────────────────────────────────────────────────────
    query_length      = len(query.split())
    year_is_multi     = int(len(q_years) > 1)
    answer_length     = len(str(rag_answer).split())
    answer_has_number = int(bool(_numbers(str(rag_answer))))

    # ── Soft retrieval metrics ─────────────────────────────────────────────────
    if corpus_text_map is not None and truth_refs:
        soft_metrics = soft_retrieval_metrics(retrieved_ids, corpus_text_map, truth_refs)
    else:
        soft_metrics = {"soft_MRR": None, "soft_NDCG@5": None, "soft_Recall@3": None}

    # ── Pitfall flags ──────────────────────────────────────────────────────────
    pitfalls = []
    if q_years and n_wrong_year > 0:
        pitfalls.append("WRONG_YEAR")
    if truth_refs and n_truth_in_corpus == 0:
        pitfalls.append("TRUTH_NOT_IN_CORPUS")
    elif n_truth_in_corpus > 0 and n_truth_retrieved == 0:
        pitfalls.append("TRUTH_NOT_RETRIEVED")

    return {
        # ─── Identity ─────────────────────────────────────────────────────────
        "finder_id":           finder_id,
        "idx":                 int(idx),
        "run_id":              run_id,
        "mode":                mode or "dense",
        "query":               query,
        # Broad FinDER question category (e.g. "Earnings Per Share", "Risk Factors")
        "category":            category,
        # Query structure type (e.g. "comparison", "trend"); None = free-form narrative
        "query_type":          query_type,

        # ─── Config columns (None for old-format single-config JSONs) ──────────
        # Short pipeline label, e.g. "header_hybrid_ENH_FLAT_ROUTED"
        "config_key":          config_key,
        # Chunking granularity ("header" = section-level, "child" = sub-section, …)
        "level":               level,
        "level_num":           level_num,    # 1=header, 2=child, 3=enriched
        "enhanced":            enhanced,     # YES / NO — was query rewritten by Llama before embedding?
        "tiered":              tiered,       # YES / NO — was recency-weighted multi-year retrieval used?
        "alpha":               section_alpha,
        "enhance_query_flag":  enhance_query_flag,
        "use_tiered_years":    use_tiered_years,
        "section_alpha":       section_alpha,
        "ticker_filter":       ticker_filter,

        # ─── Retrieval volume ──────────────────────────────────────────────────
        # Total chunks returned by the retriever (dense / sparse / hybrid)
        "n_retrieved":         n_retrieved_total,
        # Ordered chunk-id identity of this row's retrieval, pipe-joined so it's
        # hashable/groupable and Excel-safe (used to detect overlapping retrieval
        # across configs for the same question — see _find_overlap_groups).
        "retrieved_ids":       "|".join(retrieved_ids),

        # ─── Section diversity ─────────────────────────────────────────────────
        # Count of distinct (section, subsection) pairs in the top-k.
        # High = broad coverage across document structure; low = clustered in one area.
        # Key diagnostic for section routing: does section_alpha > 0 lower this?
        "n_unique_sections":   n_unique_sections,
        # Modal (section, subsection) pair across retrieved chunks, e.g. "Item 7/MD&A"
        "top_section":         top_section,

        # ─── Filter leakage ────────────────────────────────────────────────────
        # Chunks from a fiscal year not in q_years (wrong year slipped through filter)
        "n_wrong_year":        n_wrong_year,
        "wrong_year_detail":   "; ".join(wrong_year_detail),

        # ─── Ground truth alignment ────────────────────────────────────────────
        # Number of FinDER reference passages for this question
        "n_truth_refs":        len(truth_refs),
        # How many truth passages were found in the indexed PKL corpus
        # (via fingerprint match, number-containment scan, or fuzzy fallback)
        "n_truth_in_corpus":   n_truth_in_corpus,
        # How many truth chunks appeared anywhere in the retrieved top-k
        "n_truth_retrieved":   n_truth_retrieved,
        # 1-based rank of the highest-placed truth chunk; None if no truth chunk retrieved
        "best_truth_rank":     best_truth_rank,
        "truth_chunk_ids":     " | ".join(
            ", ".join(ids) if ids else "NOT_IN_CORPUS"
            for ids in truth_corpus_ids
        ),

        # ─── Soft retrieval metrics (Tier 1 — all rows) ───────────────────────
        # Per-chunk relevance = max(num_overlap, text_sim) vs truth passages, threshold 0.30.
        # Gives credit for "right neighbourhood" retrievals where the exact chunk boundary
        # doesn't match the FinDER passage. Available for every row (no corpus dependency).
        # soft_MRR      — 1/rank of the first soft-relevant chunk; 0 if none in top-k
        # soft_Recall@3 — ≥1 soft-relevant chunk in the top 3 (binary)
        # soft_NDCG@5   — graded NDCG using raw overlap scores over the top 5
        **soft_metrics,

        # ─── Merged-context lexical coverage ──────────────────────────────────
        # Truth passages compared against the full concatenated retrieved context string.
        # Measures whether the right information was retrieved, before generation.
        # word_recall  — fraction of truth words present in the context
        # num_recall   — fraction of truth numbers present in the context
        # evidence_hit — word_recall ≥ 0.50 OR num_recall ≥ 0.70
        "word_recall":         coverage["word_recall"],
        "num_recall":          coverage["num_recall"],
        "evidence_hit":        coverage["evidence_hit"],

        # ─── Query / answer features (for correlation analysis) ────────────────
        # Word count of the question text
        "query_length":        query_length,
        # 1 if the query spans multiple fiscal years, 0 otherwise
        "year_is_multi":       year_is_multi,
        # Word count of the LLM-generated answer
        "answer_length":       answer_length,
        # 1 if the generated answer contains any numeric token, 0 otherwise
        "answer_has_number":   answer_has_number,

        # ─── Pitfall flags (diagnostic) ────────────────────────────────────────
        # WRONG_YEAR          — chunks from wrong fiscal year in top-k
        # TRUTH_NOT_IN_CORPUS — no truth chunk found in indexed corpus
        # TRUTH_NOT_RETRIEVED — truth chunk in corpus but absent from top-k
        "pitfalls":            ", ".join(pitfalls) if pitfalls else "—",

        # ─── Answers ───────────────────────────────────────────────────────────
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
      pitfalls  — comma-separated flag codes
    """
    W = 120
    print("\n" + "═" * W)
    print("PER-QUERY RETRIEVAL ANALYSIS")
    print("═" * W)
    print(
        f"{'#':>4}  {'mode':<7}  {'run_id':<28}  {'query':<35}  "
        f"{'hit':>5}  {'wr':>5}  {'nr':>5}  "
        f"{'in_corp':>7}  {'rank':>5}  pitfalls"
    )
    print("─" * W)

    for r in rows:
        rank_str = str(r["best_truth_rank"]) if r["best_truth_rank"] else "—"
        q_short  = r["query"][:33] + ".." if len(r["query"]) > 35 else r["query"]
        in_corp  = f"{r['n_truth_in_corpus']}/{r['n_truth_refs']}"
        print(
            f"{r['idx']:>4}  {r['mode']:<7}  {r['run_id']:<28}  {q_short:<35}  "
            f"{'Y' if r['evidence_hit'] else 'N':>5}  "
            f"{r['word_recall']:>5.2f}  {r['num_recall']:>5.2f}  "
            f"{in_corp:>7}  {rank_str:>5}  {r['pitfalls']}"
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
    3. Truth Chunk Presence — corpus coverage + per-mode retrieval rate
    4. Pitfall Breakdown — flag counts per mode
    5. Cross-Year — global filter leakage count
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

    # ── Cross-year ─────────────────────────────────────────────────────────────
    print("── Cross-Year ──────────────────────────────────────")
    cy = (df["n_wrong_year"] > 0).sum()
    print(f"  cross-year retrieval : {cy}/{n}")


# ── Answer judges ─────────────────────────────────────────────────────────────

# config_key format from run_rag_lazy.py's write_config: level 1 (header), BM25
# (sparse), no query enhancement, section_alpha=0 (no section routing).
BASELINE_CONFIG_KEY = "L1_BM25_PLAIN_A0"


def _find_overlap_groups(records: list[dict]) -> list[dict]:
    """
    Group rows whose retrieval is EXACTLY identical (same question, same ordered
    retrieved chunk list) across different configs.

    Buckets by (finder_id, ticker_filter, retrieved_ids); only buckets with ≥2
    members and a non-empty retrieved_ids count as an overlap (rows that both
    retrieved nothing aren't a meaningful "overlap"). Returns [] immediately if
    "retrieved_ids" isn't present in the records — lets callers degrade
    gracefully on pickles saved before this field existed.

    Each group dict:
      member_idxs   — record-index labels (the actual DataFrame index / list
                       position of each member — required for later .at[] writes)
      canonical_idx — the baseline row if config_key == BASELINE_CONFIG_KEY is in
                       the group, else the member with the lexicographically
                       smallest (config_key, position) — deterministic, and the
                       position tiebreak guards against literal duplicate rows
                       in merged eval JSONs.
      answers_match — True only if every member's rag_answer is byte-identical.
                       This is the "avoid corruption" gate: callers must only
                       share a judge verdict across a group when this is True.
      contains_baseline, config_keys, n_configs, finder_id, category, query,
      ticker_filter, n_retrieved — for the overlapping_context report sheet.
    """
    if not records or "retrieved_ids" not in records[0]:
        return []

    buckets: dict[tuple, list[int]] = {}
    for i, row in enumerate(records):
        retrieved_ids = row.get("retrieved_ids", "")
        if not retrieved_ids:
            continue
        key = (row.get("finder_id"), row.get("ticker_filter"), retrieved_ids)
        buckets.setdefault(key, []).append(i)

    groups = []
    for (finder_id, ticker_filter, retrieved_ids), idxs in buckets.items():
        if len(idxs) < 2:
            continue

        answers = {records[i].get("rag_answer") for i in idxs}
        answers_match = len(answers) == 1

        baseline_member = next(
            (i for i in idxs if records[i].get("config_key") == BASELINE_CONFIG_KEY), None
        )
        canonical_idx = baseline_member if baseline_member is not None else min(
            idxs, key=lambda i: (str(records[i].get("config_key")), i)
        )

        groups.append({
            "finder_id":       finder_id,
            "ticker_filter":   ticker_filter,
            "retrieved_ids":   retrieved_ids,
            "member_idxs":     idxs,
            "canonical_idx":   canonical_idx,
            "answers_match":   answers_match,
            "contains_baseline": baseline_member is not None,
            "config_keys":     [records[i].get("config_key") for i in idxs],
            "n_configs":       len(idxs),
            "category":        records[canonical_idx].get("category"),
            "query":           records[canonical_idx].get("query"),
            "n_retrieved":     records[canonical_idx].get("n_retrieved"),
        })
    return groups


def _attach_baseline_answers(rows: list[dict]) -> None:
    """
    Fetch each row's baseline_answer: the rag_answer from the fixed baseline
    config (BASELINE_CONFIG_KEY) for the same (finder_id, ticker_filter) pair.

    Keyed on (finder_id, ticker_filter) not just finder_id so that multi-ticker
    evals (e.g. TSLA + PYPL in one JSON) each get their own company's baseline
    answer rather than whichever company's row was processed last.

    Rows where no baseline exists in the sweep get baseline_answer = "".
    """
    baseline_by_key = {
        (row["finder_id"], row.get("ticker_filter", "")): row.get("rag_answer", "")
        for row in rows
        if row.get("config_key") == BASELINE_CONFIG_KEY
    }
    n_missing = 0
    for row in rows:
        key = (row.get("finder_id"), row.get("ticker_filter", ""))
        row["baseline_answer"] = baseline_by_key.get(key, "")
        if not row["baseline_answer"]:
            n_missing += 1
    if n_missing:
        print(f"  [warn] {n_missing} row(s) have no baseline answer "
              f"(config '{BASELINE_CONFIG_KEY}' absent or ticker mismatch)")


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


def _should_swap(*key_parts: str) -> bool:
    """
    Deterministic pseudo-random coin flip for one judge comparison, used to decide
    whether rag_answer/baseline_answer get swapped into prompt slots A/B.

    Hash-based (not random.seed()) so the same (question, truth_answer, rag_answer,
    baseline_answer) tuple always swaps the same way across reruns — required for
    _judge_unjudged_rows's resumability and for _judge_unjudged_rows_deduped, where
    a dedup group's shared verdict must correspond to one consistent prompt layout.
    """
    digest = hashlib.md5("||".join(key_parts).encode("utf-8")).digest()
    return digest[0] & 1 == 1


def _unswap_verdict(value: str) -> str:
    """Map a raw A/B judge verdict back to A=rag_answer/B=baseline_answer terms."""
    return {"A": "B", "B": "A"}.get(value, value)


def judge_llm(question: str, truth_answer: str, rag_answer: str, baseline_answer: str, model, tokenizer) -> dict:
    """
    LLM-based judge using JUDGE_PROMPT. Pairwise-compares rag_answer against
    baseline_answer on relevance and completeness relative to the ground truth.

    Uses Phi-4 (`mlx-community/phi-4-4bit`) — a different architecture and training
    from Llama-3.2-3B (the generation model) — which eliminates self-serving bias.
    Runs post-hoc from saved eval JSON; no concurrency conflict with generation.

    Which answer is physically placed in prompt slot A vs B is randomized per call
    (see _should_swap) to mitigate the judge's positional/order bias. The returned
    relevance/completeness are flipped back before returning, so callers always see
    them in "A=rag_answer, B=baseline_answer" terms regardless of prompt order.

    Returns relevance, completeness ("A"/"B" strings — which answer won),
    and judge_response (full raw model output for inspection).
    """
    sys.path.append(str(Path(__file__).resolve().parent.parent))
    from prompts import JUDGE_PROMPT
    from mlx_lm import generate

    swapped = _should_swap(question, truth_answer, rag_answer, baseline_answer)
    prompt_text = JUDGE_PROMPT.format(
        question=question,
        answer_ref=truth_answer,
        answer_a=baseline_answer if swapped else rag_answer,
        answer_b=rag_answer if swapped else baseline_answer,
    )
    messages  = [{"role": "user", "content": prompt_text}]
    formatted = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    response  = generate(model, tokenizer, prompt=formatted, verbose=False, max_tokens=1000)

    result = _parse_judge_response(response)
    if swapped:
        result["relevance"]    = _unswap_verdict(result["relevance"])
        result["completeness"] = _unswap_verdict(result["completeness"])
    return result


def _parse_judge_response(response: str) -> dict:
    """Shared JSON-extraction logic for one judge_llm-style raw model response."""
    try:
        m = re.search(r'\{[^}]+\}', response, re.DOTALL)
        if m:
            parsed = json.loads(m.group())
            return {
                "relevance":     parsed.get("relevance",    "?"),
                "completeness":  parsed.get("completeness", "?"),
                "judge_response": response.strip(),
            }
    except Exception:
        pass
    return {
        "relevance":     "parse_error",
        "completeness":  "parse_error",
        "judge_response": response.strip(),
    }


def judge_llm_batch(
    items: list[tuple], model, tokenizer, max_tokens: int = 1000,
    completion_batch_size: int = 6, prefill_batch_size: int = 2,
) -> list[dict]:
    """
    Batched version of judge_llm: judges a list of (question, truth_answer, rag_answer,
    baseline_answer) tuples in ONE batched forward pass instead of one generate() call
    per item — lets a single GPU (e.g. Apple Silicon/Metal) process multiple judge
    prompts at once instead of serializing them.

    completion_batch_size/prefill_batch_size are forwarded to mlx_lm's
    BatchGenerator, which otherwise defaults to 32/8 concurrent sequences —
    that many prompts' KV caches held at once is what was OOM-ing. Lowering
    these caps concurrency (and memory) without changing how many items are
    logically judged in one call.

    Same per-item A/B randomization as judge_llm (see _should_swap): each item's
    prompt independently gets rag_answer/baseline_answer swapped or not, and each
    result is flipped back to "A=rag_answer, B=baseline_answer" terms before return.

    Returns a list of score-dicts (same shape as judge_llm's return: relevance,
    completeness, judge_response), one per item, in the same order as `items`.
    """
    sys.path.append(str(Path(__file__).resolve().parent.parent))
    from prompts import JUDGE_PROMPT
    from mlx_lm import batch_generate

    swaps = [
        _should_swap(question, truth_answer, rag_answer, baseline_answer)
        for question, truth_answer, rag_answer, baseline_answer in items
    ]

    prompts_text = []
    for (question, truth_answer, rag_answer, baseline_answer), swapped in zip(items, swaps):
        prompt_text = JUDGE_PROMPT.format(
            question=question,
            answer_ref=truth_answer,
            answer_a=baseline_answer if swapped else rag_answer,
            answer_b=rag_answer if swapped else baseline_answer,
        )
        messages = [{"role": "user", "content": prompt_text}]
        prompts_text.append(
            tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        )

    token_prompts = []
    for p in prompts_text:
        add_special = tokenizer.bos_token is None or not p.startswith(tokenizer.bos_token)
        token_prompts.append(tokenizer.encode(p, add_special_tokens=add_special))

    batch = batch_generate(
        model, tokenizer, token_prompts, max_tokens=max_tokens, verbose=True,
        completion_batch_size=completion_batch_size, prefill_batch_size=prefill_batch_size,
    )
    results = [_parse_judge_response(response) for response in batch.texts]
    for result, swapped in zip(results, swaps):
        if swapped:
            result["relevance"]    = _unswap_verdict(result["relevance"])
            result["completeness"] = _unswap_verdict(result["completeness"])
    return results


JUDGE_CHECKPOINT_CHUNK = 20  # rows per judge_llm_batch call before an atomic checkpoint write


def _checkpoint_df(df: pd.DataFrame, checkpoint_path: Path) -> None:
    """Atomically write df to checkpoint_path (write-then-rename, mirrors run_rag_lazy's _checkpoint)."""
    tmp_path = checkpoint_path.with_suffix(checkpoint_path.suffix + ".tmp")
    df.to_pickle(tmp_path)
    tmp_path.replace(checkpoint_path)


def _judge_unjudged_rows(rows_df: pd.DataFrame, judge_model, judge_tok,
                          checkpoint_path: Path | None = None) -> pd.DataFrame:
    """
    Judge every row that hasn't been judged yet, mutating rows_df in place. Rows
    needing a real verdict are judged in chunks of JUDGE_CHECKPOINT_CHUNK via
    judge_llm_batch (instead of one generate() call per row, or one giant call for
    every unjudged row) so that a crash mid-run only loses at most one chunk's
    worth of judging, not the whole pass.

    If checkpoint_path is given, rows_df is atomically written to it (write-then-
    rename, see _checkpoint_df) after every chunk — same crash-safety pattern as
    run_rag_lazy.py's per-question _checkpoint().

    "Unjudged" = relevance is null. This makes judging resumable/idempotent: calling
    this twice on the same DataFrame (e.g. apply_llm_judge run again after an interrupted
    pass) only pays for the rows still missing a verdict — already-judged rows (including
    "is_baseline"/"no_baseline"/"parse_error" sentinels) are left untouched.

    Requires columns: query, truth_answer, rag_answer, baseline_answer, config_key.
    """
    required = {"query", "truth_answer", "rag_answer", "baseline_answer", "config_key"}
    missing = required - set(rows_df.columns)
    if missing:
        raise ValueError(
            f"rows_df is missing columns required for judging: {sorted(missing)}. "
            "Was it produced by the stage-1 analysis (run_analysis / run_multi_analysis)?"
        )

    # Creation order fixes column order in the DataFrame: judge_response, relevance,
    # completeness — appended right after baseline_answer.
    for col in ("judge_response", "relevance", "completeness"):
        if col not in rows_df.columns:
            rows_df[col] = None

    todo = rows_df.index[rows_df["relevance"].isna()]
    print(f"  Judging {len(todo)}/{len(rows_df)} unjudged row(s) with Phi-4…")

    batch_idxs:  list = []
    batch_items: list[tuple] = []
    for idx in todo:
        row      = rows_df.loc[idx]
        baseline = row.get("baseline_answer", "")
        if row.get("config_key") == BASELINE_CONFIG_KEY:
            # This row IS the baseline — skip self-comparison
            rows_df.at[idx, "relevance"]      = "is_baseline"
            rows_df.at[idx, "completeness"]   = "is_baseline"
            rows_df.at[idx, "judge_response"] = ""
        elif not baseline:
            # Baseline config absent from this sweep — skip rather than judge against ""
            rows_df.at[idx, "relevance"]      = "no_baseline"
            rows_df.at[idx, "completeness"]   = "no_baseline"
            rows_df.at[idx, "judge_response"] = ""
        else:
            batch_idxs.append(idx)
            batch_items.append((
                row["query"], row.get("truth_answer", ""), row.get("rag_answer", ""), baseline,
            ))

    if batch_items:
        n_chunks = math.ceil(len(batch_items) / JUDGE_CHECKPOINT_CHUNK)
        print(f"  Judging {len(batch_items)} row(s) needing a real verdict in "
              f"{n_chunks} chunk(s) of ≤{JUDGE_CHECKPOINT_CHUNK}…")
        for c in range(n_chunks):
            lo, hi = c * JUDGE_CHECKPOINT_CHUNK, (c + 1) * JUDGE_CHECKPOINT_CHUNK
            chunk_idxs  = batch_idxs[lo:hi]
            chunk_items = batch_items[lo:hi]
            results = judge_llm_batch(chunk_items, judge_model, judge_tok)
            for idx, scores in zip(chunk_idxs, results):
                for k, v in scores.items():
                    rows_df.at[idx, k] = v
            if checkpoint_path is not None:
                _checkpoint_df(rows_df, checkpoint_path)
                print(f"    [checkpoint] {hi if hi < len(batch_items) else len(batch_items)}/"
                      f"{len(batch_items)} judged → {checkpoint_path}")

    return rows_df


_PENDING_DUP_COPY = "__pending_dup_copy__"


def _judge_unjudged_rows_deduped(rows_df: pd.DataFrame, judge_model, judge_tok,
                                  checkpoint_path: Path | None = None) -> pd.DataFrame:
    """
    Same contract as _judge_unjudged_rows, but shares one verdict across rows whose
    retrieval is exactly identical for the same question (see _find_overlap_groups),
    instead of calling judge_llm redundantly for each.

    Only ever acts on a row whose relevance is currently null — never overwrites an
    already-judged row (real verdict or sentinel), whether judged in this call or a
    prior run. This is what makes it safe to call repeatedly / resume from a
    partially-judged pickle without corrupting existing results.

    For an eligible group (answers_match=True — every member's rag_answer is
    byte-identical, so a shared verdict is actually correct, not just assumed):
      - if the canonical member IS the baseline row, every other (currently-null)
        member gets the "same_as_baseline" sentinel directly — comparing an answer
        to itself is meaningless, so it's never sent to judge_llm.
      - otherwise, every other (currently-null) member gets a temporary placeholder
        so _judge_unjudged_rows's own null-check skips it (letting the canonical
        member get judged for real, exactly once), then the placeholder is replaced
        with the canonical's real verdict once judging completes.

    Groups where answers_match is False (retrieval coincided but the generated
    answers differ — an edge case worth surfacing, not silently trusting) get NO
    dedup treatment: every member is still judged independently, same as today.
    """
    groups = _find_overlap_groups(rows_df.reset_index().to_dict("records"))
    eligible = [g for g in groups if g["answers_match"]]

    # Ensure the judge columns exist before writing to them — mirrors the same
    # column-creation _judge_unjudged_rows does internally, needed here because we
    # may pre-populate sentinels/placeholders before that function ever runs.
    for col in ("judge_response", "relevance", "completeness"):
        if col not in rows_df.columns:
            rows_df[col] = None

    pending_placeholders: list[int] = []
    for g in eligible:
        canon = g["canonical_idx"]
        canon_is_baseline = rows_df.at[canon, "config_key"] == BASELINE_CONFIG_KEY
        for m in g["member_idxs"]:
            if m == canon:
                continue
            if pd.notna(rows_df.at[m, "relevance"]):
                continue  # already judged (this run or a prior one) — never touch it
            if canon_is_baseline:
                rows_df.at[m, "relevance"]      = "same_as_baseline"
                rows_df.at[m, "completeness"]   = "same_as_baseline"
                rows_df.at[m, "judge_response"] = ""
            else:
                rows_df.at[m, "relevance"]    = _PENDING_DUP_COPY
                rows_df.at[m, "completeness"] = _PENDING_DUP_COPY
                pending_placeholders.append(m)

    rows_df = _judge_unjudged_rows(rows_df, judge_model, judge_tok, checkpoint_path=checkpoint_path)

    for g in eligible:
        canon = g["canonical_idx"]
        for m in g["member_idxs"]:
            if m in pending_placeholders:
                for col in ("relevance", "completeness", "judge_response"):
                    rows_df.at[m, col] = rows_df.at[canon, col]

    if checkpoint_path is not None:
        _checkpoint_df(rows_df, checkpoint_path)

    return rows_df


def print_judge_summary(rows: list[dict], use_llm_judge: bool) -> None:
    """
    Print answer-quality summary.

    Lexical section: always printed. Shows lex_hit rate, mean word/num recall per mode.
    LLM section: only printed when use_llm_judge=True. Shows the rate at which rag_answer
    (A) was judged better than baseline_answer (B) on relevance and completeness, per mode.
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
        print("── LLM Judge / Phi-4 (rag_answer vs baseline_answer) ───")
        for mode in modes:
            sub       = df[df["mode"] == mode]
            mn        = len(sub)
            rel_win   = (sub["relevance"]    == "A").sum()
            comp_win  = (sub["completeness"] == "A").sum()
            parse_err = (sub["relevance"] == "parse_error").sum()
            print(f"  {mode:<8}: relevance_win={rel_win}/{mn} ({rel_win/mn*100:.1f}%)  "
                  f"completeness_win={comp_win}/{mn} ({comp_win/mn*100:.1f}%)"
                  + (f"  [parse_errors={parse_err}]" if parse_err else ""))
        print()


# ── Core analysis (callable directly for batch runs) ─────────────────────────

def run_analysis(
    eval_path:    Path,
    corpus:       pd.DataFrame,
    fp_index:     dict,
    corpus_text_map: dict[str, str] | None = None,
    judge_model   = None,
    judge_tok     = None,
) -> None:
    """
    Stage 1 (+ optional stage 2): analyse one eval JSON file and save an Excel + TXT
    report alongside it. Accepts pre-loaded corpus and judge model so batch callers
    can load them once and reuse across multiple eval files.

    Retrieval analysis + the lexical judge always run and never require a model.
    LLM judging only runs when judge_model is given, via the same resumable
    _judge_unjudged_rows() path used by apply_llm_judge() — see that function to
    add judging later, on top of the saved pickle, without rerunning retrieval
    analysis.
    """
    print(f"\nEval file : {eval_path}")
    print(f"LLM judge : {'Phi-4 (enabled)' if judge_model is not None else 'disabled'}")
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

    # ── Baseline answer — fetched from the matching baseline-config row, if present ──
    _attach_baseline_answers(rows)

    # ── LLM judge — optional, post-hoc, uses Phi-4 ──
    if judge_model is not None:
        rows = _judge_unjudged_rows(pd.DataFrame(rows), judge_model, judge_tok).to_dict("records")

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
    _capture(print_judge_summary, rows, judge_model is not None)

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
    # Cast all bool-dtype columns to int first (pd.to_numeric silently no-ops on bool)
    for col in df.select_dtypes(include="bool").columns:
        df[col] = df[col].astype(int)
    for col in ("enhance_query_flag", "use_tiered_years", "year_is_multi",
                "evidence_hit", "answer_has_number"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # "_int" = 1.0 when rag_answer (A) beat baseline_answer (B), 0.0 when B won,
    # NaN for rows that weren't judged (is_baseline / no_baseline / parse_error).
    for col in ("relevance", "completeness"):
        if col in df.columns:
            df[col + "_int"] = df[col].map({"A": 1.0, "B": 0.0})

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
            "n_truth_in_corpus": "sum",
        }
        for col in ("relevance_int", "completeness_int"):
            if col in df.columns:
                agg_dict[col] = "mean"
        for col in ("level", "mode", "enhance_query_flag", "use_tiered_years", "section_alpha"):
            if col in df.columns:
                agg_dict[col] = "first"

        cfg_sum = df.groupby(group_col, dropna=False).agg(agg_dict).reset_index()
        cfg_sum.rename(columns={"finder_id": "n_rows"}, inplace=True)

        sort_col = "relevance_int" if "relevance_int" in cfg_sum.columns else "evidence_hit"
        cfg_sum = cfg_sum.sort_values(sort_col, ascending=False)
        cfg_sum.to_excel(writer, sheet_name="config_summary", index=False)

        # ── Sheet 3: category_breakdown ─────────────────────────────────────
        cat_cols = ["category", group_col]
        cat_agg: dict = {"finder_id": "count", "evidence_hit": "mean", "soft_MRR": "mean",
                         "soft_Recall@3": "mean"}
        for col in ("relevance_int", "completeness_int"):
            if col in df.columns:
                cat_agg[col] = "mean"
        if "category" in df.columns:
            cat_bd = df.groupby(cat_cols, dropna=False).agg(cat_agg).reset_index()
            cat_bd.rename(columns={"finder_id": "n_rows"}, inplace=True)
            cat_bd.to_excel(writer, sheet_name="category_breakdown", index=False)

        # ── Sheet 4: failure_analysis ─── rows where baseline_answer (B) beat rag_answer (A)
        if "relevance" in df.columns:
            fail_mask = (df["relevance"] == "B") | (df["completeness"] == "B")
        else:
            fail_mask = df["evidence_hit"] == 0
        fail_df = df[fail_mask].copy()
        if "category" in fail_df.columns:
            fail_df = fail_df.sort_values(["category", group_col])
        fail_df.to_excel(writer, sheet_name="failure_analysis", index=False)

        # ── Sheet 5: correlations ────────────────────────────────────────────
        corr_cols = [c for c in [
            "query_length", "year_is_multi", "n_retrieved", "n_unique_sections",
            "enhance_query_flag", "use_tiered_years", "section_alpha",
            "relevance_int", "completeness_int",
            "evidence_hit", "soft_MRR", "soft_Recall@3", "soft_NDCG@5",
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
            "relevance_int", "completeness_int",
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

        # ── Sheet 7: baseline_comparison ─────────────────────────────────────
        bc_df, _ = _build_delta_df(df, group_col)
        if not bc_df.empty:
            bc_df.to_excel(writer, sheet_name="baseline_comparison", index=False)

        # ── Sheet 8: axis_x_category ──────────────────────────────────────────
        _write_axis_category_sheet(df, writer)

        # ── Sheet 9: section_routing ──────────────────────────────────────────
        _write_section_routing_sheet(df, writer)

        # ── Sheet 10: overlapping_context ──────────────────────────────────────
        _write_overlap_sheet(df, writer)

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
    One-click entry point for multi-config eval JSONs produced by run_rag_lazy.py.

    Runs analyze_entry + the (free) lexical judge for every row, then — if judge_model
    is given — runs the LLM judge (judge_llm, using JUDGE_PROMPT) inline in the same
    pass. Writes one pickle + one multi-sheet Excel and returns the rows DataFrame.

    Pass judge_model=None to skip LLM judging entirely (fast, no model load).
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

    # ── Lexical judge — skip rows that overlap another row's exact retrieval; ──
    # ── copy that row's (already-computed) score instead of recomputing it.    ──
    overlap_groups = _find_overlap_groups(rows)
    eligible_groups = [g for g in overlap_groups if g["answers_match"]]
    lex_dup_map = {
        m: g["canonical_idx"]
        for g in eligible_groups
        for m in g["member_idxs"]
        if m != g["canonical_idx"]
    }
    for i, row in enumerate(rows):
        if i not in lex_dup_map:
            row.update(judge_lexical(row.get("rag_answer", ""), row.get("truth_answer", "")))
    for dup_i, canon_i in lex_dup_map.items():
        for k in ("lex_word_recall", "lex_num_recall", "lex_hit"):
            rows[dup_i][k] = rows[canon_i][k]

    # ── Baseline answer — fetched from the matching baseline-config row, if present ──
    _attach_baseline_answers(rows)

    rows_df = pd.DataFrame(rows)

    out_dir  = eval_path.parent / "analysis"
    out_xlsx = out_dir / eval_path.name.replace("eval_", "analysis_").replace(".json", ".xlsx")
    out_pkl  = out_xlsx.with_suffix(".pkl")
    out_pkl.parent.mkdir(parents=True, exist_ok=True)

    # ── LLM judge — optional, runs inline as part of the same pass ──────────────
    # checkpoint_path=out_pkl: judging writes to the same pickle atomically after
    # every chunk, so a crash mid-judging loses at most one chunk, not the whole pass.
    if judge_model is not None:
        rows_df = _judge_unjudged_rows_deduped(rows_df, judge_model, judge_tok, checkpoint_path=out_pkl)

    rows_df.to_pickle(out_pkl)
    _write_multi_excel(rows_df, out_xlsx)
    print(f"Saved → {out_pkl}")

    _print_multi_summary(rows_df)
    _print_baseline_delta_summary(rows_df)

    return rows_df


def apply_llm_judge(analysis_pkl: Path, judge_model, judge_tok) -> pd.DataFrame:
    """
    Optional standalone helper: load an existing analysis pickle and run the LLM judge
    on any rows without a verdict yet, in place. Not part of the one-click flow —
    run_multi_analysis / run_analysis already run judging inline when given a
    judge_model. Use this only to add judging on top of a pickle that was produced
    without a judge_model.
    """
    print(f"\nAnalysis pickle : {analysis_pkl}")
    rows_df = pd.read_pickle(analysis_pkl)
    rows_df = _judge_unjudged_rows_deduped(rows_df, judge_model, judge_tok, checkpoint_path=analysis_pkl)

    rows_df.to_pickle(analysis_pkl)
    out_xlsx = analysis_pkl.with_suffix(".xlsx")
    _write_multi_excel(rows_df, out_xlsx)
    print(f"Saved → {analysis_pkl}")

    _print_multi_summary(rows_df)
    _print_baseline_delta_summary(rows_df)
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
    # "_int" = 1.0 when rag_answer (A) beat baseline_answer (B), 0.0 when B won,
    # NaN for rows that weren't judged (is_baseline / no_baseline / parse_error).
    for col in ("relevance", "completeness"):
        if col in df.columns:
            df[col + "_int"] = df[col].map({"A": 1.0, "B": 0.0})

    agg: dict = {"finder_id": "count", "evidence_hit": "mean", "word_recall": "mean",
                 "num_recall": "mean", "soft_MRR": "mean", "soft_Recall@3": "mean"}
    for c in ("relevance_int", "completeness_int"):
        if c in df.columns:
            agg[c] = "mean"
    ranked = df.groupby(group_col).agg(agg).reset_index()
    ranked.rename(columns={"finder_id": "n"}, inplace=True)
    sort_col = "relevance_int" if "relevance_int" in ranked else "evidence_hit"
    ranked = ranked.sort_values(sort_col, ascending=False)

    print(f"\n{'config_key':<55} {'n':>4}  {'rel_win%':>8}  {'cmp_win%':>9}  {'soft_MRR':>8}  {'Recall@3':>8}  {'evi_hit%':>8}  {'word_rec':>8}  {'num_rec':>7}")
    print("─" * 125)
    for _, r in ranked.iterrows():
        rel  = f"{r.get('relevance_int', float('nan'))*100:.1f}" if "relevance_int" in r and pd.notna(r.get("relevance_int")) else "  n/a"
        comp = f"{r.get('completeness_int', float('nan'))*100:.1f}" if "completeness_int" in r and pd.notna(r.get("completeness_int")) else "  n/a"
        mrr  = f"{r['soft_MRR']:.3f}" if pd.notna(r.get("soft_MRR")) else "  n/a"
        rc3  = f"{r.get('soft_Recall@3', float('nan')):.3f}" if pd.notna(r.get("soft_Recall@3")) else "  n/a"
        hit  = f"{r['evidence_hit']*100:.1f}" if pd.notna(r.get("evidence_hit")) else "  n/a"
        wr   = f"{r['word_recall']:.3f}" if pd.notna(r.get("word_recall")) else "  n/a"
        nr   = f"{r['num_recall']:.3f}" if pd.notna(r.get("num_recall")) else "  n/a"
        print(f"  {str(r[group_col]):<53} {int(r['n']):>4}  {rel:>8}  {comp:>9}  {mrr:>8}  {rc3:>8}  {hit:>8}  {wr:>8}  {nr:>7}")

    if "category" in df.columns and "relevance" in df.columns:
        fail = df[df["relevance"] == "B"]
        if not fail.empty:
            top_fail = fail.groupby("category").size().sort_values(ascending=False).head(5)
            print(f"\nTop failure categories (baseline beat rag on relevance, relevance=B):")
            for cat, cnt in top_fail.items():
                total_cat = (df["category"] == cat).sum()
                print(f"  {cat:<40} {cnt}/{total_cat}")

    if "n_truth_in_corpus" in df.columns:
        hard_rows = df[df["n_truth_in_corpus"].fillna(0) > 0]
        print(f"\nHard eval coverage: {len(hard_rows)}/{n} rows have truth chunk(s) in corpus")


# ── Baseline-delta comparison ─────────────────────────────────────────────────
#
# Baseline = Level 1 | sparse/BM25 retrieval | section_alpha=0 | tiered=False
# Every other config is shown as an incremental gain/loss vs this reference point.
#
# Metric columns in the comparison table: (df_column, display_label, format_spec)
_CMP_METRICS: list[tuple[str, str, str]] = [
    ("evidence_hit",         "evi_hit", ".3f"),
    ("word_recall",          "wrd_rec", ".3f"),
    ("num_recall",           "num_rec", ".3f"),
    ("soft_MRR",             "sft_MRR", ".3f"),
    ("soft_Recall@3",        "R@3",     ".3f"),
    ("soft_NDCG@5",          "NDCG@5",  ".3f"),
    ("relevance_int",        "rel",     ".3f"),
    ("completeness_int",     "cmp",     ".3f"),
]


def _find_baseline_cfg(df: pd.DataFrame, group_col: str) -> str | None:
    """
    Return the config_key that represents the baseline.

    Fixed to BASELINE_CONFIG_KEY ("L1_BM25_PLAIN_A0") — no performance-based
    fallback, since the baseline is a fixed reference point and may legitimately
    be outperformed by other configs. Returns None if that key isn't in this
    sweep (callers already handle None).
    """
    if BASELINE_CONFIG_KEY in df[group_col].astype(str).values:
        return BASELINE_CONFIG_KEY
    return None


def _build_delta_df(df: pd.DataFrame, group_col: str) -> tuple[pd.DataFrame, str | None]:
    """
    Aggregate per-config means and attach a Δ_<metric> column for each metric
    showing the difference vs the baseline config.

    Returns (comparison_df, baseline_config_key).
    """
    df = df.copy()
    for col in ("evidence_hit", "use_tiered_years", "enhance_query_flag", "section_alpha"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    # "_int" = 1.0 when rag_answer (A) beat baseline_answer (B), 0.0 when B won,
    # NaN for rows that weren't judged (is_baseline / no_baseline / parse_error).
    for col in ("relevance", "completeness"):
        if col in df.columns:
            df[col + "_int"] = df[col].map({"A": 1.0, "B": 0.0})

    present = [col for col, _, _ in _CMP_METRICS if col in df.columns]
    if not present:
        return pd.DataFrame(), None

    agg   = df.groupby(group_col, dropna=False)[present].mean().round(4)
    n_map = df.groupby(group_col, dropna=False).size()

    baseline_cfg = _find_baseline_cfg(df, group_col)
    if baseline_cfg is None or baseline_cfg not in agg.index:
        return agg.reset_index(), None

    base_row = agg.loc[baseline_cfg]
    rows = []
    for cfg in agg.index:
        row = agg.loc[cfg]
        rec: dict = {group_col: str(cfg), "is_baseline": (cfg == baseline_cfg), "n": int(n_map[cfg])}
        for col in present:
            bv = base_row[col]
            v  = row[col]
            rec[col]        = v
            rec[f"Δ_{col}"] = round(float(v - bv), 4) if (pd.notna(v) and pd.notna(bv)) else None
        rows.append(rec)

    result   = pd.DataFrame(rows)
    sort_col = next((c for c in ["relevance_int", "evidence_hit", "soft_MRR"] if c in result.columns), None)
    if sort_col:
        result = result.sort_values(sort_col, ascending=False, ignore_index=True)
    return result, baseline_cfg


def _delta_str(delta, fmt: str = ".3f") -> str:
    """Format a numeric delta with sign and directional arrow (↑ ↓ →)."""
    if delta is None or (isinstance(delta, float) and math.isnan(delta)):
        return "   n/a"
    sign  = "+" if delta >= 0 else ""
    arrow = "↑" if delta >= 0.05 else ("↓" if delta <= -0.05 else "→")
    return f"{sign}{delta:{fmt}}{arrow}"


# Axes used to determine which single factor changed vs baseline
_AXIS_COLS  = ["level", "mode", "enhance_query_flag", "use_tiered_years", "section_alpha"]
_AXIS_LABEL = {
    "level":              "LEVEL",
    "mode":               "RETRIEVAL MODE",
    "enhance_query_flag": "QUERY ENHANCEMENT",
    "use_tiered_years":   "TIERED YEAR FILTER",
    "section_alpha":      "SECTION ROUTING (alpha)",
}


def _norm_axis(v) -> str:
    """Canonical string for axis-value comparison (handles bool/int/float/str)."""
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return "none"
    try:
        f = float(v)
        return str(int(f)) if f == int(f) else f"{f:.2f}"
    except (TypeError, ValueError):
        return str(v).strip().lower()


def _print_baseline_delta_summary(df: pd.DataFrame) -> None:
    """
    Print incremental config impact grouped by which single axis changed vs baseline.

    Baseline = alpha=0, tiered=False (+ level 1 and sparse as tiebreakers).
    Configs differing on exactly ONE axis are shown under that axis header.
    Multi-axis combos are listed separately at the bottom.
    Arrow legend: ↑ ≥ +0.05   → within ±0.05   ↓ ≤ −0.05
    """
    group_col = "config_key" if "config_key" in df.columns and df["config_key"].notna().any() else "mode"
    delta_df, baseline_cfg = _build_delta_df(df, group_col)
    if delta_df.empty:
        return
    if baseline_cfg is None:
        print(f"\n  (baseline config '{BASELINE_CONFIG_KEY}' not present in this sweep — skipping delta summary)")
        return

    present = [(col, lbl, fmt) for col, lbl, fmt in _CMP_METRICS if col in df.columns]
    if not present:
        return

    # ── Metadata lookup: config_key → first row (for axis comparison) ────────
    first_rows: dict[str, pd.Series] = {
        str(cfg): grp.iloc[0]
        for cfg, grp in df.groupby(group_col, dropna=False)
    }
    base_meta = first_rows.get(str(baseline_cfg), pd.Series())

    # ── Categorise each config by how many axes differ from baseline ──────────
    axis_buckets: dict[str, list] = {ax: [] for ax in _AXIS_COLS}
    multi_axis:   list            = []

    for _, row in delta_df.iterrows():
        if row["is_baseline"]:
            continue
        cfg_str = str(row[group_col])
        meta    = first_rows.get(cfg_str, pd.Series())
        changed = [
            ax for ax in _AXIS_COLS
            if _norm_axis(base_meta.get(ax)) != _norm_axis(meta.get(ax))
        ]
        if len(changed) == 1:
            axis_buckets[changed[0]].append(row)
        else:
            multi_axis.append(row)

    # ── Layout helpers ────────────────────────────────────────────────────────
    cfg_w   = min(52, max(len(str(c)) for c in delta_df[group_col]) + 2)
    col_w   = 15
    divider = "  " + "─" * (cfg_w + 6 + len(present) * col_w)
    hdr_row = f"  {'config':<{cfg_w}}  {'n':>4}" + "".join(
        f"  {lbl:>5}  {'Δ':>6}" for _, lbl, _ in present
    )

    def _fmt_row(row, is_base=False):
        marker = "►" if is_base else " "
        line   = f" {marker} {str(row[group_col]):<{cfg_w}}  {int(row['n']):>4}"
        for col, _, fmt in present:
            v     = row.get(col)
            d     = row.get(f"Δ_{col}")
            v_str = f"{v:{fmt}}" if pd.notna(v) else "  n/a"
            d_str = "  base" if is_base else _delta_str(d, fmt)
            line += f"  {v_str:>5}  {d_str:>6}"
        return line

    # ── Baseline banner ───────────────────────────────────────────────────────
    W = 110
    base_row   = delta_df[delta_df["is_baseline"]].iloc[0] if delta_df["is_baseline"].any() else None
    auto_note  = "" if str(baseline_cfg) in df[group_col].astype(str).values else " (auto-selected)"
    print("\n" + "╔" + "═" * (W - 2) + "╗")
    banner = f"  BASELINE{auto_note}: {baseline_cfg}"
    if base_row is not None:
        parts = [
            f"{lbl}={base_row.get(col):{fmt}}" if pd.notna(base_row.get(col)) else f"{lbl}=n/a"
            for col, lbl, fmt in present
        ]
        banner += "   │   " + "  │  ".join(parts)
    print(f"║{banner:<{W-2}}║")
    print("╚" + "═" * (W - 2) + "╝")

    # ── One section per axis ──────────────────────────────────────────────────
    print("\n  ONE-AXIS IMPACT vs BASELINE")
    any_printed = False
    for axis in _AXIS_COLS:
        rows = axis_buckets[axis]
        if not rows:
            continue
        any_printed = True
        label = _AXIS_LABEL[axis]
        print(f"\n  ── {label} {'─' * max(1, 44 - len(label))}")
        print(hdr_row)
        print(divider)
        if base_row is not None:
            print(_fmt_row(base_row, is_base=True))
        for r in sorted(rows, key=lambda r: -float(r.get("evidence_hit") or 0)):
            print(_fmt_row(r))
        print(divider)

    if not any_printed:
        print("  (no single-axis neighbours found — run the 6 missing configs first)")

    # ── Multi-axis combos ─────────────────────────────────────────────────────
    if multi_axis:
        print(f"\n  ── MULTI-AXIS COMBOS {'─' * 24}")
        print(hdr_row)
        print(divider)
        if base_row is not None:
            print(_fmt_row(base_row, is_base=True))
        for r in sorted(multi_axis, key=lambda r: -float(r.get("evidence_hit") or 0)):
            print(_fmt_row(r))
        print(divider)

    print("  ↑ ≥ +0.05   → within ±0.05   ↓ ≤ −0.05\n")


# ── Axis × Category impact ────────────────────────────────────────────────────
#
# Sheets 8 & 9: cross-tabulate config axes (and retrieved SEC sections) against
# query categories to reveal which categories benefit from each axis change.
#
# Metrics: evidence_hit, word_recall, num_recall, soft_MRR, soft_NDCG@5, soft_Recall@3
# Δ column: binary axes → signed (other − baseline); 3-value axes → max − min range.

_AX_CAT_METRICS: list[tuple[str, str]] = [
    ("evidence_hit",  "evi_hit"),
    ("word_recall",   "wrd_rec"),
    ("num_recall",    "num_rec"),
    ("soft_MRR",      "sft_MRR"),
    ("soft_NDCG@5",   "NDCG@5"),
    ("soft_Recall@3", "R@3"),
]

# Baseline value per axis (normalised string, used to determine Δ direction)
_AX_BASELINE_NORM: dict[str, str] = {
    "use_tiered_years":   "0",
    "enhance_query_flag": "0",
    "section_alpha":      "0",
    "mode":               "sparse",
    # "level" handled via substring match in _is_ax_baseline
}


def _ax_val_label(axis: str, v) -> str:
    """Human-readable column header for one axis value."""
    if axis == "use_tiered_years":
        return "TIERED" if pd.notna(v) and float(v or 0) else "FLAT"
    if axis == "enhance_query_flag":
        return "ENH" if pd.notna(v) and float(v or 0) else "PLAIN"
    if axis == "section_alpha":
        return f"α={float(v or 0):.2f}"
    if axis == "level":
        m = re.search(r"\d", str(v or ""))
        return f"L{m.group()}" if m else str(v)
    return str(v).lower() if pd.notna(v) else "none"


def _is_ax_baseline(axis: str, v) -> bool:
    """True if v is the reference/baseline value for this axis."""
    n = _norm_axis(v)
    if axis == "level":
        return "level 1" in n or n.startswith("l1")
    b = _AX_BASELINE_NORM.get(axis)
    return n == b if b is not None else False


def _axis_category_block(
    df: pd.DataFrame,
    axis: str,
    present_metrics: list[tuple[str, str]],
) -> "pd.DataFrame | None":
    """
    Wide comparison table: rows = categories + OVERALL, columns = per-axis-value
    metrics + Δ_<metric> vs the baseline axis value.

    Returns None when axis is absent or has < 2 unique values.
    """
    if axis not in df.columns or "category" not in df.columns:
        return None

    ax = df[axis].copy()
    if axis in ("use_tiered_years", "enhance_query_flag", "section_alpha"):
        ax = pd.to_numeric(ax, errors="coerce")

    unique_vals = sorted(ax.dropna().unique(), key=lambda v: _norm_axis(v))
    if len(unique_vals) < 2:
        return None

    val_labels = [_ax_val_label(axis, v) for v in unique_vals]
    bline_lbl  = next(
        (lbl for v, lbl in zip(unique_vals, val_labels) if _is_ax_baseline(axis, v)),
        val_labels[0],
    )

    categories = sorted(df["category"].dropna().unique())
    records: list[dict] = []

    for cat in list(categories) + ["OVERALL"]:
        mask_cat = (df["category"] == cat) if cat != "OVERALL" else pd.Series(True, index=df.index)
        rec: dict = {"category": cat}
        val_avgs: dict[str, dict[str, float]] = {}

        for v, vlbl in zip(unique_vals, val_labels):
            sub = df[mask_cat & (ax == v)]
            rec[f"n_{vlbl}"] = len(sub)
            avgs: dict[str, float] = {}
            for col, mlbl in present_metrics:
                avg = pd.to_numeric(sub[col], errors="coerce").mean()
                rec[f"{mlbl}_{vlbl}"] = round(float(avg), 4) if pd.notna(avg) else None
                avgs[mlbl] = float(avg) if pd.notna(avg) else float("nan")
            val_avgs[vlbl] = avgs

        # Δ vs baseline value
        for _, mlbl in present_metrics:
            base_v = val_avgs.get(bline_lbl, {}).get(mlbl, float("nan"))
            if len(val_labels) == 2:
                other_lbl = next(l for l in val_labels if l != bline_lbl)
                other_v   = val_avgs.get(other_lbl, {}).get(mlbl, float("nan"))
                rec[f"Δ_{mlbl}"] = (
                    round(other_v - base_v, 4)
                    if math.isfinite(base_v) and math.isfinite(other_v) else None
                )
            else:
                valid = [
                    val_avgs[l].get(mlbl, float("nan"))
                    for l in val_labels
                    if math.isfinite(val_avgs[l].get(mlbl, float("nan")))
                ]
                rec[f"Δ_{mlbl}"] = round(max(valid) - min(valid), 4) if len(valid) >= 2 else None

        records.append(rec)

    block = pd.DataFrame(records)
    return pd.concat(
        [block[block["category"] != "OVERALL"], block[block["category"] == "OVERALL"]],
        ignore_index=True,
    )


def _write_axis_category_sheet(df: pd.DataFrame, writer: "pd.ExcelWriter") -> None:
    """
    Sheet 8: axis_x_category
    Five stacked blocks, one per config axis, showing mean retrieval metrics
    per (axis_value × query_category). Δ columns show signed change vs baseline
    axis value (binary axes) or max-min range (multi-value axes).
    """
    present = [(col, lbl) for col, lbl in _AX_CAT_METRICS if col in df.columns]
    if not present:
        return

    all_frames: list[pd.DataFrame] = []
    spacer = pd.DataFrame([{"category": ""}, {"category": ""}])

    for axis in _AXIS_COLS:
        block = _axis_category_block(df, axis, present)
        if block is None or block.empty:
            continue
        title = pd.DataFrame([{"category": f"── {_AXIS_LABEL[axis]} ──"}])
        if all_frames:
            all_frames.append(spacer)
        all_frames.extend([title, block])

    if not all_frames:
        return

    pd.concat(all_frames, ignore_index=True).to_excel(
        writer, sheet_name="axis_x_category", index=False
    )


def _write_section_routing_sheet(df: pd.DataFrame, writer: "pd.ExcelWriter") -> None:
    """
    Sheet 9: section_routing
    Part 1 – Retrieved section × category: for each query category, which SEC
              document sections (Item 7, Item 1A, …) are retrieved, and how
              well does retrieval score for each section.
    Part 2 – Section routing impact: (when section_alpha varies) compare which
              sections are retrieved at α=0 vs α>0, and whether routing shifts
              the section distribution towards better-scoring sections.
    """
    if "top_section" not in df.columns or "category" not in df.columns:
        return

    categories = sorted(df["category"].dropna().unique())
    top_secs   = df["top_section"].dropna().value_counts().head(8).index.tolist()
    if not top_secs:
        return

    # ── Part 1: retrieved section × category performance ─────────────────────
    p1_rows: list[dict] = []
    for cat in list(categories) + ["OVERALL"]:
        sub = df if cat == "OVERALL" else df[df["category"] == cat]
        n_total = len(sub)
        rec: dict = {"category": cat, "total_n": n_total}

        for sec in top_secs + ["OTHER"]:
            sub_sec = sub[sub["top_section"] == sec] if sec != "OTHER" else sub[~sub["top_section"].isin(top_secs)]
            n = len(sub_sec)
            rec[f"n_{sec}"]   = n
            rec[f"pct_{sec}"] = round(n / n_total * 100, 1) if n_total else None
            evi = pd.to_numeric(sub_sec["evidence_hit"], errors="coerce").mean()
            rec[f"evi_{sec}"] = round(float(evi), 3) if pd.notna(evi) else None
            if "soft_MRR" in df.columns:
                mrr = pd.to_numeric(sub_sec["soft_MRR"], errors="coerce").mean()
                rec[f"mrr_{sec}"] = round(float(mrr), 3) if pd.notna(mrr) else None

        p1_rows.append(rec)

    part1_df = pd.DataFrame(p1_rows)

    # ── Part 2: section alpha routing impact ──────────────────────────────────
    part2_df = None
    if "section_alpha" in df.columns:
        alphas = sorted(
            pd.to_numeric(df["section_alpha"], errors="coerce").dropna().unique()
        )
        if len(alphas) >= 2:
            present = [(col, lbl) for col, lbl in _AX_CAT_METRICS if col in df.columns]
            p2_rows: list[dict] = []

            for cat in list(categories) + ["OVERALL"]:
                sub = df if cat == "OVERALL" else df[df["category"] == cat]
                rec = {"category": cat}
                per_alpha: dict[float, dict[str, float]] = {}

                for av in alphas:
                    lbl  = f"α={av:.2f}"
                    sub_a = sub[pd.to_numeric(sub["section_alpha"], errors="coerce") == av]
                    mode_sec = sub_a["top_section"].dropna().mode()
                    rec[f"top_sec {lbl}"] = mode_sec.iloc[0] if not mode_sec.empty else None
                    m_vals: dict[str, float] = {}
                    for col, mlbl in present:
                        avg = pd.to_numeric(sub_a[col], errors="coerce").mean()
                        val = float(avg) if pd.notna(avg) else float("nan")
                        rec[f"{mlbl} {lbl}"] = round(val, 3) if math.isfinite(val) else None
                        m_vals[mlbl] = val
                    per_alpha[av] = m_vals

                sec_modes = [rec.get(f"top_sec α={av:.2f}") for av in alphas]
                rec["section_shifted"] = "YES" if len({s for s in sec_modes if s}) > 1 else "no"

                # Δ between last and first alpha value for each metric
                for _, mlbl in present:
                    b = per_alpha[alphas[0]].get(mlbl, float("nan"))
                    c = per_alpha[alphas[-1]].get(mlbl, float("nan"))
                    rec[f"Δ_{mlbl}"] = (
                        round(c - b, 3) if math.isfinite(b) and math.isfinite(c) else None
                    )

                p2_rows.append(rec)
            part2_df = pd.DataFrame(p2_rows)

    # ── Assemble and write ────────────────────────────────────────────────────
    t1 = pd.DataFrame([{"category": "── RETRIEVED SECTION × CATEGORY PERFORMANCE ──"}])
    frames: list[pd.DataFrame] = [t1, part1_df]

    if part2_df is not None:
        t2     = pd.DataFrame([{"category": "── SECTION ROUTING (α) IMPACT BY CATEGORY ──"}])
        spacer = pd.DataFrame([{"category": ""}, {"category": ""}])
        frames += [spacer, t2, part2_df]

    pd.concat(frames, ignore_index=True).to_excel(
        writer, sheet_name="section_routing", index=False
    )


def _write_overlap_sheet(df: pd.DataFrame, writer: "pd.ExcelWriter") -> None:
    """
    Sheet 10: overlapping_context
    One row PER overlap group PER finder_id (not per member row) — i.e. if 3 configs
    for the same question retrieved the exact same ordered chunk list, that's one
    row here listing all 3 config_keys, not 3 rows. A finder_id with two unrelated
    overlap clusters gets two rows. Sorted by finder_id.

    Shows ALL groups found, including answers_match=False ones (retrieval coincided
    but the generated answers differ anyway) — those are diagnostically important
    and are NOT judge/lexical-deduped elsewhere, so they must stay visible here.
    """
    groups = _find_overlap_groups(df.reset_index().to_dict("records"))
    if not groups:
        return

    groups = sorted(groups, key=lambda g: str(g["finder_id"]))
    rows = []
    group_counter: dict[str, int] = {}
    for g in groups:
        fid = str(g["finder_id"])
        group_counter[fid] = group_counter.get(fid, 0) + 1
        rows.append({
            "finder_id":         g["finder_id"],
            "group_id":          f"{fid}_g{group_counter[fid]}",
            "category":          g["category"],
            "query":             g["query"],
            "ticker_filter":     g["ticker_filter"],
            "n_configs":         g["n_configs"],
            "config_keys":       ", ".join(str(c) for c in g["config_keys"]),
            "contains_baseline": g["contains_baseline"],
            "answers_match":     g["answers_match"],
            "n_retrieved":       g["n_retrieved"],
            "retrieved_ids_hash": hashlib.sha1(g["retrieved_ids"].encode()).hexdigest()[:12],
        })

    pd.DataFrame(rows).to_excel(writer, sheet_name="overlapping_context", index=False)


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    """
    One-click pipeline. Analyses every file in eval_files in one pass each:
    retrieval analysis + lexical judge always run; set USE_LLM_JUDGE=True to also
    run the LLM judge (judge_llm, via JUDGE_PROMPT) inline, in the same pass, for
    every file.
    """
    USE_LLM_JUDGE = True

    eval_files = [
        "/Users/nadavsmacbookair/Desktop/Thesis/data/eval_results/ready_for_analysis/tsla_pypl-corr_nvda_aapl.json",
    ]

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


if __name__ == "__main__":
    main()
