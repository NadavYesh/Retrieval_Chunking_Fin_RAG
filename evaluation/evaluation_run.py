#!/usr/bin/env python3
"""
RAG evaluation analysis: retrieval statistics, truth-chunk corpus lookup, and pitfall detection.
Supports single-mode (dense-only) and multi-mode (dense/sparse/hybrid) eval JSON files.

Two distinct retrieval quality metrics are tracked:

  evidence_hit   — soft content coverage. Checks whether the retrieved context contains
                   enough words and numbers from the truth passage (word_recall ≥ 0.50 OR
                   num_recall ≥ 0.70). Can be True even if the exact truth chunk was never
                   retrieved, because the same figures often appear in multiple overlapping chunks.

  truth retrieved — hard chunk-identity check. The specific chunk ID from the PKL corpus that
                    was identified as the authoritative answer must appear in the retrieved list.
                    Strict: a sibling chunk with identical numbers will not satisfy this.

These two can diverge: high evidence_hit + 0% truth retrieved means the system surfaces the
right information from neighbouring chunks, but not the canonical one FinDER labelled.

Usage:
    python analysis.py [path/to/eval_YYYYMMDD_HHMM.json]
"""

import contextlib
import io
import json
import pickle
import re
import sys
from difflib import SequenceMatcher
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from evaluation_functions import score_row

# ── Config ────────────────────────────────────────────────────────────────────

EVAL_FILE  = Path("/Users/nadavsmacbookair/Desktop/Thesis/data/eval_results/WMT_eval_20260626_1603.json")
CHUNKS_DIR = Path("/Users/nadavsmacbookair/Desktop/Thesis/data/financial_corpora/chunks/hierarchical/indexed-at-26-06-26/header")

LOW_SCORE_THRESH = 0.60   # cosine similarity threshold — only meaningful for dense mode
SCORE_GAP_THRESH = 0.03   # min gap between best and worst retrieved score; small gap = undiscriminated results
FINGERPRINT_LEN  = 200    # chars of normalised text used as a fast exact-match key in fp_index
FUZZY_THRESH     = 0.80   # SequenceMatcher ratio needed to declare a fuzzy truth-chunk match


# ── Corpus loading ────────────────────────────────────────────────────────────

def _norm(text: str) -> str:
    """Collapse whitespace and lowercase — used for all text comparisons."""
    return re.sub(r"\s+", " ", str(text).strip()).lower()


def load_corpus() -> tuple[pd.DataFrame, dict[str, str]]:
    """
    Load every PKL chunk file under CHUNKS_DIR into a single DataFrame and build a
    fingerprint index for O(1) truth-chunk lookup.

    Returns
    -------
    corpus : pd.DataFrame
        All chunks concatenated. Columns include at least 'id', 'text', 'metadata',
        'source_file'. One row per indexed chunk.

    fp_index : dict[str, str]
        Maps the first FINGERPRINT_LEN chars of normalised chunk text → chunk UUID.
        Used as the primary (exact) truth-chunk lookup; fuzzy search is the fallback.
    """
    frames = []
    for p in sorted(CHUNKS_DIR.glob("*.pkl")):
        with open(p, "rb") as f:
            df = pickle.load(f)
        df["source_file"] = p.name
        frames.append(df)
    corpus = pd.concat(frames, ignore_index=True)

    fp_index: dict[str, str] = {}
    for _, row in corpus.iterrows():
        fp = _norm(row["text"])[:FINGERPRINT_LEN]
        if fp and fp not in fp_index:
            fp_index[fp] = row["id"]
    return corpus, fp_index


def find_truth_chunk(truth_text: str, corpus: pd.DataFrame, fp_index: dict[str, str]) -> str | None:
    """
    Locate the corpus chunk that best matches a FinDER truth passage.

    Search strategy (in order of cost):
      1. Exact fingerprint match against fp_index (O(1)).
      2. Shorter prefix match (100 and 50 chars) to handle trailing whitespace differences.
      3. SequenceMatcher fuzzy match on first 500 chars (O(n) over full corpus — slow but
         only reached when exact match fails, which typically means the filing year is not
         indexed or the text was extracted with a different parser).

    Returns the chunk UUID string, or None if no match meets FUZZY_THRESH.
    """
    norm = _norm(truth_text)

    fp = norm[:FINGERPRINT_LEN]
    if fp in fp_index:
        return fp_index[fp]

    for plen in (100, 50):
        short = norm[:plen]
        for idx_fp, cid in fp_index.items():
            if idx_fp[:plen] == short:
                return cid

    for _, row in corpus.iterrows():
        ratio = SequenceMatcher(None, norm[:500], _norm(row["text"])[:500]).ratio()
        if ratio >= FUZZY_THRESH:
            return row["id"]

    return None


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

def analyze_entry(idx: str, data: dict, corpus: pd.DataFrame, fp_index: dict) -> dict:
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
    run_id        = data["run_id"].get(idx, "")
    query         = data["query"].get(idx, "")
    truth_refs    = data["truth_ref"].get(idx, [])
    retrieved_str = data["rag_retrieved"].get(idx, "")
    rag_answer    = data.get("rag_answer",   {}).get(idx, "")
    truth_answer  = data.get("truth_answer", {}).get(idx, "")

    if isinstance(truth_refs, str):
        truth_refs = [truth_refs]
    if truth_refs is None:
        truth_refs = []

    q_ticker, q_years, mode = parse_run_id(run_id)
    # Prefer the explicit "mode" column when present (new eval format adds it)
    if "mode" in data:
        mode = data["mode"].get(idx, mode)

    retrieved     = parse_retrieved(retrieved_str)
    retrieved_ids = [s["chunk_id"] for s in retrieved if s["chunk_id"]]

    # ── Score stats (only meaningful for dense; BM25 scores are unbounded) ──
    scores     = [s["score"] for s in retrieved if s["score"] is not None]
    max_score  = max(scores) if scores else None
    min_score  = min(scores) if scores else None
    score_gap  = round(max_score - min_score, 4) if scores else None
    mean_score = round(float(np.mean(scores)), 4) if scores else None

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
    # truth_corpus_ids: for each FinDER reference passage, the matched PKL chunk UUID
    # (or None if the filing is not in the indexed corpus at all)
    truth_corpus_ids: list[str | None] = [
        find_truth_chunk(ref, corpus, fp_index) for ref in truth_refs
    ]
    n_truth_in_corpus = sum(c is not None for c in truth_corpus_ids)

    retrieved_id_set  = set(retrieved_ids)
    n_truth_retrieved = sum(
        c is not None and c in retrieved_id_set for c in truth_corpus_ids
    )

    best_truth_rank = None
    for cid in truth_corpus_ids:
        if cid and cid in retrieved_ids:
            rank = retrieved_ids.index(cid) + 1
            if best_truth_rank is None or rank < best_truth_rank:
                best_truth_rank = rank

    # ── Word / number recall ───────────────────────────────────────────────────
    # score_row compares the truth_refs text against the full retrieved context string
    coverage = score_row(truth_refs, retrieved_str)

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
        "idx":               int(idx),
        "run_id":            run_id,
        "mode":              mode or "dense",
        "query":             query,
        "n_retrieved":       len(retrieved),
        "max_score":         round(max_score, 4) if max_score is not None else None,
        "min_score":         round(min_score, 4) if min_score is not None else None,
        "mean_score":        mean_score,
        "score_gap":         score_gap,
        "n_wrong_ticker":    n_wrong_ticker,
        "n_wrong_year":      n_wrong_year,
        "wrong_year_detail": "; ".join(wrong_year_detail),
        "n_truth_refs":      len(truth_refs),
        "n_truth_in_corpus": n_truth_in_corpus,
        "n_truth_retrieved": n_truth_retrieved,
        "best_truth_rank":   best_truth_rank,
        "truth_chunk_ids":   " | ".join(str(c) if c else "NOT_IN_CORPUS" for c in truth_corpus_ids),
        "word_recall":       coverage["word_recall"],
        "num_recall":        coverage["num_recall"],
        "evidence_hit":      coverage["evidence_hit"],
        "pitfalls":          ", ".join(pitfalls) if pitfalls else "—",
        "rag_answer":        rag_answer,
        "truth_answer":      truth_answer,
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
    use_llm_judge: bool = False,
    judge_model   = None,   # pre-loaded Phi-4 model; only used when use_llm_judge=True
    judge_tok     = None,
) -> None:
    """
    Analyse one eval JSON file and save a CSV + TXT report alongside it.

    Accepts pre-loaded corpus and judge model so batch callers can load them
    once and reuse across multiple eval files.

    Parameters
    ----------
    eval_path     : path to an eval JSON produced by run_rag.py
    corpus        : full PKL corpus DataFrame from load_corpus()
    fp_index      : fingerprint index from load_corpus()
    use_llm_judge : whether to run the Phi-4 LLM judge
    judge_model   : pre-loaded Phi-4 model (required when use_llm_judge=True)
    judge_tok     : tokenizer paired with judge_model
    """
    print(f"\nEval file : {eval_path}")
    print(f"LLM judge : {'Phi-4 (enabled)' if use_llm_judge else 'disabled'}")
    with open(eval_path) as f:
        data = json.load(f)

    indices = sorted(data["run_id"].keys(), key=int)
    print(f"  Analyzing {len(indices)} rows …")

    rows = []
    for idx in indices:
        row = analyze_entry(idx, data, corpus, fp_index)
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

    # ── Print and simultaneously capture all sections for the text report ──
    report_parts = []

    def _run(fn, *args, **kwargs):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            fn(*args, **kwargs)
        text = buf.getvalue()
        print(text, end="")
        report_parts.append(text)

    _run(print_per_query_table,    rows)
    _run(print_truth_chunk_detail, rows)
    _run(print_summary,            rows)
    _run(print_judge_summary,      rows, use_llm_judge)

    out_csv = eval_path.parent / eval_path.name.replace("eval_", "analysis_").replace(".json", ".csv")
    out_txt = out_csv.with_suffix(".txt")

    pd.DataFrame(rows).to_csv(out_csv, index=False)
    out_txt.write_text("".join(report_parts), encoding="utf-8")

    print(f"\nSaved → {out_csv}")
    print(f"Saved → {out_txt}")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    """
    CLI entry point. Parses flags, loads corpus and optionally Phi-4 once, then
    delegates to run_analysis().

    CLI flags:
      positional arg  : path to eval JSON (default: EVAL_FILE constant)
      --judge         : enable Phi-4 LLM judge
    """
    use_llm_judge = "--judge" in sys.argv
    positional    = [a for a in sys.argv[1:] if not a.startswith("--")]
    eval_path     = Path(positional[0]) if positional else EVAL_FILE

    print(f"Corpus dir: {CHUNKS_DIR}")
    corpus, fp_index = load_corpus()
    print(f"  Loaded {len(corpus)} chunks from {corpus['source_file'].nunique()} PKL files")

    judge_model, judge_tok = None, None
    if use_llm_judge:
        from mlx_lm import load as mlx_load
        print("\nLoading Phi-4 judge model…")
        judge_model, judge_tok = mlx_load("mlx-community/phi-4-4bit")

    run_analysis(eval_path, corpus, fp_index, use_llm_judge, judge_model, judge_tok)


if __name__ == "__main__":
    # ── Batch mode: define eval files to analyse in sequence ──────────────────
    # Corpus and judge model are loaded ONCE and reused across all files.
    # To run a single file interactively, pass it as a CLI arg instead:
    #   python evaluation_run.py path/to/eval_YYYYMMDD_HHMM.json [--judge]

    USE_LLM_JUDGE = False   # set True to enable Phi-4 judging for all runs

    eval_files = [
        # Path("/Users/nadavsmacbookair/Desktop/Thesis/data/eval_results/eval_20260626_1537.json"),
    ]

    if not eval_files:
        # Fall back to CLI / EVAL_FILE constant when no batch list is defined
        main()
    else:
        print(f"Corpus dir: {CHUNKS_DIR}")
        corpus, fp_index = load_corpus()
        print(f"  Loaded {len(corpus)} chunks from {corpus['source_file'].nunique()} PKL files")

        judge_model, judge_tok = None, None
        if USE_LLM_JUDGE:
            from mlx_lm import load as mlx_load
            print("\nLoading Phi-4 judge model…")
            judge_model, judge_tok = mlx_load("mlx-community/phi-4-4bit")

        for i, ef in enumerate(eval_files):
            print(f"\n{'='*60}\nBatch {i+1}/{len(eval_files)}: {Path(ef).name}\n{'='*60}")
            run_analysis(Path(ef), corpus, fp_index, USE_LLM_JUDGE, judge_model, judge_tok)
