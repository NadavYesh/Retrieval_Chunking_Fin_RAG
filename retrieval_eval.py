"""
Relaxed retrieval coverage evaluation for FinDER evidence.

For each result row we check whether the retrieved context contains enough
of the ground-truth evidence passage to count as a "hit".  Because chunking
can split a passage, we use two soft signals:

  word_recall  – fraction of evidence word-tokens that appear anywhere in
                 the context (≥ 0.50 → hit).
  num_recall   – fraction of numeric tokens in the evidence that appear in
                 the context (≥ 0.70 → hit).  Numbers are the load-bearing
                 tokens in financial text, so this catches cases where
                 surrounding prose differs but the figures are present.

evidence_hit = word_recall >= 0.50  OR  num_recall >= 0.70

When FinDER supplies multiple reference passages we compute the metrics for
each and keep the best (most generous) match.
"""

import re
from typing import Union
import pandas as pd


# ── helpers ──────────────────────────────────────────────────────────────────

def _tokens(text: str) -> set[str]:
    """Lowercase alphabetic/numeric tokens, strip punctuation."""
    return set(re.findall(r"[a-z0-9,.\-]+", text.lower()))

print(_tokens('heloo, my name is "nada;v" '))

def _numbers(text: str) -> set[str]:
    """Numeric tokens: integers, decimals, negatives, comma-formatted."""
    return set(re.findall(r"-?[\d,]+\.?\d*", text))


def _word_recall(evidence: str, context: str) -> float:
    ev_toks = _tokens(evidence)
    if not ev_toks:
        return 0.0
    ctx_toks = _tokens(context)
    return len(ev_toks & ctx_toks) / len(ev_toks)


def _num_recall(evidence: str, context: str) -> float:
    ev_nums = _numbers(evidence)
    if not ev_nums:
        return 0.0
    ctx_nums = _numbers(context)
    return len(ev_nums & ctx_nums) / len(ev_nums)


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
