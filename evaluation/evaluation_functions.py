"""
Relaxed retrieval coverage evaluation for FinDER evidence.

For each result row we check whether the retrieved context contains enough
of the ground-truth evidence passage to count as a "hit".  Because chunking
can split a passage, we use two soft signals:

  word_recall  – fraction of the evidence's content words (stopwords removed)
                 that appear anywhere in the context (≥ 0.50 → hit).
  num_recall   – fraction of the evidence's *salient* numbers that appear in
                 the context (≥ 0.70 → hit).  Numbers are the load-bearing
                 tokens in financial text, so this catches cases where
                 surrounding prose differs but the figures are present.
                 Only computed when the evidence carries at least
                 MIN_SALIENT_NUMS of them — see salient_numbers().

evidence_hit = word_recall >= 0.50  OR  num_recall >= 0.70

When FinDER supplies multiple reference passages we compute the metrics for
each and keep the best (most generous) match.

This module owns the tokenizers for the whole evaluation stage — evaluation_run.py
imports them rather than defining its own, so the numeric/word notion of "overlap"
cannot drift between the merged-context metrics here and the per-chunk soft
relevance metrics there.
"""

import re
from typing import Union
import pandas as pd


# ── tokenizers ────────────────────────────────────────────────────────────────

# The leading \d is load-bearing: a character class of [\d,] alone also matches a
# bare "," , which made every comma in ordinary prose register as a "number". Truth
# passages without figures then had the number-set {","}, which any chunk in the
# corpus trivially covers, so num_recall was 1.0 (and evidence_hit unconditionally
# True) for every narrative question.
_NUM_RE = re.compile(r"-?\d[\d,]*(?:\.\d+)?")

# A bare 4-digit year is not evidence: it is a filing-wide constant repeated in the
# header of nearly every chunk of the document, so matching on it says nothing about
# whether a chunk carries the passage's actual figures.
_YEAR_RE = re.compile(r"^(?:19|20)\d{2}$")

# Content words only (≥3 letters). Numbers are deliberately excluded — they are
# scored separately and far more precisely by the numeric leg.
_WORD_RE = re.compile(r"[a-z][a-z\-]{2,}")

# Financial-boilerplate + English function words. These appear in essentially every
# 10-K chunk, so leaving them in inflates word_recall towards a high, uninformative floor.
_STOPWORDS = frozenset("""
the and for our are was were with that this from has have had not any all its their which
such other than been will may can into under over more most upon also both each per
company companies inc corporation business financial statements year years fiscal
including include includes included related respectively approximately during
""".split())

# Minimum salient numbers a truth passage must carry before its numeric overlap is
# trusted. Below this the denominator is so small that a single coincidental match
# (one shared figure) scores ≥ 0.30 — measured false-positive rate against randomly
# drawn chunks was ~85% at 1-3 numbers, versus ~4% at 6+. Mirrors MIN_CHUNK_NUMS in
# evaluation_run.find_truth_chunks, which guards the same failure from the other side.
MIN_SALIENT_NUMS = 4


def _numbers(text: str) -> set[str]:
    """Numeric tokens: integers, decimals, negatives, comma-formatted. Digits required."""
    return {m.strip(",") for m in _NUM_RE.findall(str(text))}


def salient_numbers(text: str) -> set[str]:
    """Numeric tokens minus bare years — the figures that actually identify a passage."""
    return {n for n in _numbers(text) if not _YEAR_RE.match(n)}


def _words(text: str) -> set[str]:
    """Content words: ≥3 letters, lowercased, stopwords removed."""
    return {w for w in _WORD_RE.findall(str(text).lower()) if w not in _STOPWORDS}


# ── recall over a reference/context pair ──────────────────────────────────────

def _word_recall(evidence: str, context: str) -> float:
    """Fraction of the evidence's content words present anywhere in the context."""
    ev_toks = _words(evidence)
    if not ev_toks:
        return 0.0
    return len(ev_toks & _words(context)) / len(ev_toks)


def _num_recall(evidence: str, context: str) -> float:
    """
    Fraction of the evidence's salient numbers present in the context.

    Returns 0.0 when the evidence carries fewer than MIN_SALIENT_NUMS of them: the
    ratio is not measuring anything at that point, and letting it through would fire
    evidence_hit's num_recall >= 0.70 leg on coincidence alone.
    """
    ev_nums = salient_numbers(evidence)
    if len(ev_nums) < MIN_SALIENT_NUMS:
        return 0.0
    return len(ev_nums & salient_numbers(context)) / len(ev_nums)


# ── per-row scoring ───────────────────────────────────────────────────────────

WORD_THRESH = 0.50
NUM_THRESH  = 0.70


def score_row(
    references: Union[list, str],
    context: str,
) -> dict:
    """
    Score one result row.

    Parameters
    ----------
    references : list[str] | str
        Ground-truth evidence passage(s) from FinDER.
    context : str
        Concatenated retrieved context from the pipeline.

    Returns
    -------
    dict with word_recall, num_recall, evidence_hit (all best-passage values).
    """
    if isinstance(references, str):
        passages = [references]
    elif not references:
        return {"word_recall": 0.0, "num_recall": 0.0, "evidence_hit": False}
    else:
        passages = list(references)

    best_word = 0.0
    best_num  = 0.0

    for passage in passages:
        wr = _word_recall(passage, context)
        nr = _num_recall(passage, context)
        if wr > best_word:
            best_word = wr
        if nr > best_num:
            best_num = nr
    
    # hit is defined as either of the threshold.
    hit = best_word >= WORD_THRESH or best_num >= NUM_THRESH
    return {"word_recall": round(best_word, 4),
            "num_recall":  round(best_num,  4),
            "evidence_hit": hit}


# ── dataframe-level evaluation ────────────────────────────────────────────────

def evaluate_retrieval_coverage(
    results_df: pd.DataFrame,
    finder_df:  pd.DataFrame,
) -> pd.DataFrame:
    """
    Merge pipeline results with FinDER ground truth and compute coverage.

    Expected columns in results_df  : original_query, context_text, answer, run_id, level, …
    Expected columns in finder_df   : text (question), references (list[str]), answer

    Returns a tidy DataFrame with columns:
        run_id | raw_query | level | evidence | context | answer_pipeline
        word_recall | num_recall | evidence_hit
    """
    # Build reference lookup: question → references list
    ref_lookup = dict(zip(finder_df["text"], finder_df["references"]))

    rows = []
    for _, r in results_df.iterrows():
        query      = r.get("original_query", "")
        context    = r.get("context_text", "")
        references = ref_lookup.get(query, [])

        scores = score_row(references, context)

        rows.append({
            "run_id":          r.get("run_id", ""),
            "raw_query":       query,
            "level":           r.get("level", ""),
            "ticker":          r.get("ticker", ""),
            "year":            r.get("year", ""),
            "evidence":        references,
            "context":         context,
            "answer_pipeline": r.get("answer", ""),
            **scores,
        })

    eval_df = pd.DataFrame(rows)

    # Summary per level
    if not eval_df.empty:
        summary = (
            eval_df.groupby("level")["evidence_hit"]
            .agg(n="count", hits="sum")
            .assign(hit_rate=lambda d: (d["hits"] / d["n"]).round(3))
        )
        print("\n── Retrieval coverage summary ──")
        print(summary.to_string())
        print(f"\nOverall hit rate: {eval_df['evidence_hit'].mean():.3f}  "
              f"({eval_df['evidence_hit'].sum()}/{len(eval_df)})")

    return eval_df
