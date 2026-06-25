#!/usr/bin/env python3
"""
RAG evaluation analysis: retrieval statistics, truth-chunk corpus lookup, and pitfall detection.
Supports single-mode (dense-only) and multi-mode (dense/sparse/hybrid) eval JSON files.

Usage:
    python analysis.py [path/to/eval_YYYYMMDD_HHMM.json]
"""

import json
import pickle
import re
import sys
from difflib import SequenceMatcher
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from retrieval_eval import score_row

# ── Config ────────────────────────────────────────────────────────────────────

EVAL_FILE  = Path("/Users/nadavsmacbookair/Desktop/Thesis/data/eval_results/eval_20260625_2313.json")
CHUNKS_DIR = Path("/Users/nadavsmacbookair/Desktop/Thesis/data/financial_corpora/chunks/hierarchical/indexed-at-26-06-26/header")

LOW_SCORE_THRESH = 0.60
SCORE_GAP_THRESH = 0.03
FINGERPRINT_LEN  = 200
FUZZY_THRESH     = 0.80


# ── Corpus loading ────────────────────────────────────────────────────────────

def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", str(text).strip()).lower()


def load_corpus() -> tuple[pd.DataFrame, dict[str, str]]:
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
    m = re.search(pattern, text)
    return m.group(1).strip() if m else None


def _re_float(pattern: str, text: str) -> float | None:
    m = re.search(pattern, text)
    return float(m.group(1)) if m else None


def parse_retrieved(retrieved_str: str) -> list[dict]:
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
    Handles both old format (ticker_year_level_idx) and new format (ticker_year_level_mode_idx).
    Returns (ticker, years, mode).
    """
    if "_header_" in run_id:
        prefix, suffix = run_id.split("_header_", 1)
        # suffix is either "mode_idx" (new) or "idx" (old)
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
    if not fy_str:
        return None
    m = re.match(r"(\d{4})", fy_str)
    return int(m.group(1)) if m else None


# ── Per-query analysis ────────────────────────────────────────────────────────

def analyze_entry(idx: str, data: dict, corpus: pd.DataFrame, fp_index: dict) -> dict:
    run_id        = data["run_id"].get(idx, "")
    query         = data["query"].get(idx, "")
    truth_refs    = data["truth_ref"].get(idx, [])
    retrieved_str = data["rag_retrieved"].get(idx, "")

    if isinstance(truth_refs, str):
        truth_refs = [truth_refs]
    if truth_refs is None:
        truth_refs = []

    q_ticker, q_years, mode = parse_run_id(run_id)
    # Prefer the explicit "mode" column when present (new eval format)
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
        # Scores: comparable only within dense mode
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
    }


# ── Output ────────────────────────────────────────────────────────────────────

def print_per_query_table(rows: list[dict]) -> None:
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
        # Show scores as-is but mark non-dense as "n/a" for score gap
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
    # Only show once per question (use dense rows, or first mode available)
    df   = pd.DataFrame(rows)
    seen = set()
    print("\n" + "═" * 100)
    print("TRUTH CHUNK CORPUS LOOKUP")
    print("═" * 100)
    for r in rows:
        key = r["run_id"].replace(f"_{r['mode']}_", "_")  # strip mode to deduplicate
        if key in seen:
            continue
        seen.add(key)
        ids    = r["truth_chunk_ids"]
        status = "IN_CORPUS" if "NOT_IN_CORPUS" not in ids else (
            "PARTIAL" if any(c != "NOT_IN_CORPUS" for c in ids.split(" | ")) else "NOT_IN_CORPUS"
        )
        print(f"  {r['run_id']:<35}  {status:<15} {ids[:55]}{'…' if len(ids)>55 else ''}")


def print_summary(rows: list[dict]) -> None:
    df = pd.DataFrame(rows)
    n  = len(df)
    modes = sorted(df["mode"].unique())

    print("\n" + "═" * 60)
    print("AGGREGATE SUMMARY")
    print("═" * 60)
    print(f"Total rows: {n}  |  Modes present: {modes}\n")

    # ── Per-mode coverage comparison ──────────────────────────────────────────
    print("── Evidence Hit Rate by Mode ───────────────────────")
    for mode in modes:
        sub = df[df["mode"] == mode]
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
        # Pivot: question (idx stripped of mode) × mode
        df["q_key"] = df["run_id"].apply(
            lambda r: re.sub(r"_(dense|sparse|hybrid)_", "_", r)
        )
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

    # ── Truth chunk presence (shared across modes — same truth, same corpus) ──
    print("── Truth Chunk Presence ────────────────────────────")
    # Use one mode's rows (truth lookup result is identical across modes)
    ref_rows = df[df["mode"] == modes[0]]
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


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    eval_path = Path(sys.argv[1]) if len(sys.argv) > 1 else EVAL_FILE
    print(f"\nEval file : {eval_path}")
    with open(eval_path) as f:
        data = json.load(f)

    print(f"Corpus dir: {CHUNKS_DIR}")
    corpus, fp_index = load_corpus()
    print(f"  Loaded {len(corpus)} chunks from {corpus['source_file'].nunique()} PKL files")

    indices = sorted(data["run_id"].keys(), key=int)
    print(f"  Analyzing {len(indices)} rows …")

    rows = []
    for idx in indices:
        row = analyze_entry(idx, data, corpus, fp_index)
        rows.append(row)
        sys.stdout.write(f"\r  [{int(idx)+1:>3}/{len(indices)}] {row['run_id']:<40} mode={row['mode']}")
        sys.stdout.flush()
    print()

    print_per_query_table(rows)
    print_truth_chunk_detail(rows)
    print_summary(rows)

    out_csv = eval_path.parent / eval_path.name.replace("eval_", "analysis_").replace(".json", ".csv")
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(f"\nSaved → {out_csv}")


if __name__ == "__main__":
    main()
