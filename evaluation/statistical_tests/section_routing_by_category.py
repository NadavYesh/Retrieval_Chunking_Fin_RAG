"""
Per-category significance testing for the section-routing ablation (alpha 0 vs 1).

Reads the `detail` sheet of an eval workbook, where each row is one
(query x retrieval-config) trial. Every query is run under both section_alpha=0
and section_alpha=1 with an otherwise identical config, so trials pair exactly
on (query, base_config). That pairing is what the tests below exploit.

Metrics
  soft_Recall@3  binary per trial  -> exact McNemar (binomial on discordant pairs)
  soft_MRR       continuous        -> Wilcoxon signed-rank on paired differences
  ordinal        continuous (mean of relevance_int/completeness_int vs. the fixed
                 BM25 baseline, Section~\ref{sec:results-judge}'s primary judge
                 score) -> Wilcoxon signed-rank, same pairing, when the workbook
                 carries judge columns (skipped otherwise -- see load_trials).

soft_NDCG@5 is deliberately not computed or reported. The harness builds its
IDCG by sorting the *retrieved* relevance scores rather than consulting the
corpus, and retrieval depth equals the cutoff (k=5), so the "ideal" ranking is
always a permutation of the returned list. The ratio can only see whether the
returned chunks are ordered among themselves, never whether any is relevant.
Recall@3 and MRR carry the retrieval signal; NDCG@5 is excluded.

Why the judge score is included despite being relative to a fixed reference
  ordinal is each config's per-query win/tie/loss against the single fixed
  baseline (L1_BM25_PLAIN_A0, section_alpha=0), not an absolute quality score.
  That is not a problem for this test: both members of an alpha=0 vs. alpha=1
  pair are scored against the SAME fixed reference, so comparing them still
  isolates what routing changes -- it asks whether routing raises or lowers the
  win-rate against a common yardstick, the same logical structure as comparing
  soft_MRR at alpha=0 vs. alpha=1 directly. Using ordinal (the mean of relevance
  and completeness) rather than either axis alone matches the primary score
  Section~\ref{sec:results-judge} ranks configurations on.

Multiplicity
  The eight categories are scanned together and the interesting ones selected
  after the fact, so the per-category p-values are corrected across the family,
  separately per metric (three families: Recall@3, MRR, ordinal). Holm's
  step-down procedure controls the same family-wise error rate as Bonferroni and
  is uniformly more powerful, so Holm is used. Benjamini-Hochberg is reported
  alongside for readers who prefer FDR control.

Independence
  The unit of independence is the QUERY, not the trial. With 2-8 queries per
  category the trial-level n (36-144) overstates the evidence, because the 18
  configurations sharing a query are correlated. The per-query win/loss split is
  reported next to every test; categories with two queries demonstrate a
  mechanism, they do not estimate an effect size.

Usage:
    python section_routing_by_category.py [path/to/workbook.xlsx]
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest, wilcoxon
from statsmodels.stats.multitest import multipletests

DEFAULT_WORKBOOK = (
    Path(__file__).resolve().parents[3]
    / "data/eval_results/ready_for_analysis/analysis"
    / "tsla_pypl-corr_nvda_aapl_no_out-of-corpus-pollution.xlsx"
)

ALPHA_COL = "section_alpha"
PAIR_KEYS = ["query", "base_cfg"]
RECALL = "soft_Recall@3"
MRR = "soft_MRR"
ORDINAL = "ordinal"

SPOTLIGHT = ["Legal", "Shareholder return", "Company overview", "Risk", "Governance"]


def load_trials(workbook: Path) -> pd.DataFrame:
    df = pd.read_excel(workbook, sheet_name="detail")
    df[ALPHA_COL] = df[ALPHA_COL].astype(float)
    # config_key encodes alpha as a trailing _A0/_A1; strip it to pair the trials
    df["base_cfg"] = df["config_key"].str.replace(r"_A[01]$", "", regex=True)
    if ORDINAL not in df.columns and {"relevance_int", "completeness_int"} <= set(df.columns):
        df[ORDINAL] = df[["relevance_int", "completeness_int"]].mean(axis=1)
    elif ORDINAL not in df.columns and "relevance" in df.columns:
        # workbook carries raw A/B/Tie verdicts but not the numeric columns
        # judge_tie_breakdown.py writes -- derive ordinal the same way it does.
        win_map = {"A": 1.0, "B": 0.0}
        rel = df["relevance"].map(win_map).fillna(0.5)
        comp = df["completeness"].map(win_map).fillna(0.5)
        df[ORDINAL] = (rel + comp) / 2
    return df


def has_judge(trials: pd.DataFrame) -> bool:
    return ORDINAL in trials.columns


def pair_on_alpha(trials: pd.DataFrame, metric: str) -> pd.DataFrame:
    """One row per (query, base_cfg); columns a0 / a1 hold the metric."""
    wide = trials.pivot_table(index=PAIR_KEYS, columns=ALPHA_COL, values=metric)
    wide.columns = ["a0", "a1"]
    return wide.dropna()


def mcnemar_exact(paired: pd.DataFrame) -> dict:
    """Exact McNemar for a binary metric: discordant pairs tested against p=0.5."""
    gains = int(((paired.a0 == 0) & (paired.a1 == 1)).sum())
    losses = int(((paired.a0 == 1) & (paired.a1 == 0)).sum())
    discordant = gains + losses
    p = binomtest(gains, discordant, 0.5).pvalue if discordant else 1.0
    return {"gains": gains, "losses": losses, "p": p}


def wilcoxon_paired(paired: pd.DataFrame) -> float:
    if (paired.a0 == paired.a1).all():
        return 1.0
    return wilcoxon(paired.a0, paired.a1).pvalue


def per_query_split(trials: pd.DataFrame, metric: str) -> tuple[int, int, int]:
    """How many QUERIES improve / regress / hold, averaging over configs."""
    by_q = trials.pivot_table(index="query", columns=ALPHA_COL, values=metric)
    by_q.columns = ["a0", "a1"]
    delta = by_q.a1 - by_q.a0
    return int((delta > 0).sum()), int((delta < 0).sum()), int((delta == 0).sum())


def category_report(trials: pd.DataFrame) -> pd.DataFrame:
    judge = has_judge(trials)
    rows = []
    for cat, sub in trials.groupby("category"):
        r3 = pair_on_alpha(sub, RECALL)
        mrr = pair_on_alpha(sub, MRR)
        mc = mcnemar_exact(r3)
        up, down, flat = per_query_split(sub, RECALL)
        row = {
            "category": cat,
            "n_queries": sub["query"].nunique(),
            "n_pairs": len(r3),
            "R@3_a0": r3.a0.mean(),
            "R@3_a1": r3.a1.mean(),
            "R@3_delta": r3.a1.mean() - r3.a0.mean(),
            "gains": mc["gains"],
            "losses": mc["losses"],
            "mcnemar_p": mc["p"],
            "q_up": up,
            "q_down": down,
            "q_flat": flat,
            "MRR_a0": mrr.a0.mean(),
            "MRR_a1": mrr.a1.mean(),
            "MRR_delta": mrr.a1.mean() - mrr.a0.mean(),
            "wilcoxon_p": wilcoxon_paired(mrr),
        }
        if judge:
            ordinal = pair_on_alpha(sub, ORDINAL)
            row.update(
                {
                    "ORD_a0": ordinal.a0.mean(),
                    "ORD_a1": ordinal.a1.mean(),
                    "ORD_delta": ordinal.a1.mean() - ordinal.a0.mean(),
                    "ordinal_wilcoxon_p": wilcoxon_paired(ordinal),
                }
            )
        rows.append(row)
    report = pd.DataFrame(rows)

    # Family-wise correction across the eight categories, per metric family (see docstring).
    families = [
        ("mcnemar_p", "holm", "R@3_holm_p", "R@3_holm_sig"),
        ("mcnemar_p", "fdr_bh", "R@3_bh_p", "R@3_bh_sig"),
        ("wilcoxon_p", "holm", "MRR_holm_p", "MRR_holm_sig"),
    ]
    if judge:
        families.append(("ordinal_wilcoxon_p", "holm", "ORD_holm_p", "ORD_holm_sig"))
    for raw_col, method, adj_col, sig_col in families:
        reject, adjusted, _, _ = multipletests(report[raw_col], alpha=0.05, method=method)
        report[adj_col] = adjusted
        report[sig_col] = np.where(reject, "sig", "ns")

    return report.sort_values("R@3_delta", ascending=False)


def overall_report(trials: pd.DataFrame) -> dict:
    r3 = pair_on_alpha(trials, RECALL)
    mrr = pair_on_alpha(trials, MRR)
    mc = mcnemar_exact(r3)
    out = {
        "n_queries": trials["query"].nunique(),
        "n_pairs": len(r3),
        "R@3_a0": r3.a0.mean(),
        "R@3_a1": r3.a1.mean(),
        "R@3_delta": r3.a1.mean() - r3.a0.mean(),
        "gains": mc["gains"],
        "losses": mc["losses"],
        "mcnemar_p": mc["p"],
        "MRR_a0": mrr.a0.mean(),
        "MRR_a1": mrr.a1.mean(),
        "MRR_delta": mrr.a1.mean() - mrr.a0.mean(),
        "wilcoxon_p": wilcoxon_paired(mrr),
    }
    if has_judge(trials):
        ordinal = pair_on_alpha(trials, ORDINAL)
        out.update(
            {
                "ORD_a0": ordinal.a0.mean(),
                "ORD_a1": ordinal.a1.mean(),
                "ORD_delta": ordinal.a1.mean() - ordinal.a0.mean(),
                "ordinal_wilcoxon_p": wilcoxon_paired(ordinal),
            }
        )
    return out


def routing_destinations(trials: pd.DataFrame, category: str, top: int = 3) -> pd.DataFrame:
    """Which 10-K item the top-ranked chunk came from, before vs after routing."""
    sub = trials[trials.category == category]
    counts = sub.groupby([ALPHA_COL, "top_section"]).size().rename("n_trials").reset_index()
    return (
        counts.sort_values([ALPHA_COL, "n_trials"], ascending=[True, False])
        .groupby(ALPHA_COL)
        .head(top)
    )


def per_query_detail(trials: pd.DataFrame, category: str) -> pd.DataFrame:
    """Recall@3 per query, so single-query drivers of a category effect are visible."""
    sub = trials[trials.category == category]
    table = sub.pivot_table(index="query", columns=ALPHA_COL, values=RECALL)
    table.columns = ["a0", "a1"]
    table["delta"] = table.a1 - table.a0
    return table.sort_values("delta", ascending=False)


def main() -> None:
    workbook = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_WORKBOOK
    trials = load_trials(workbook)

    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 40)

    print(f"workbook: {workbook.name}")
    print(
        f"trials: {len(trials)}  queries: {trials['query'].nunique()}  "
        f"configs per alpha: {trials.groupby(ALPHA_COL)['base_cfg'].nunique().to_dict()}"
    )
    empty = int((trials.n_retrieved == 0).sum())
    print(f"trials retrieving zero chunks: {empty}\n")

    print("== SECTION ROUTING IMPACT BY CATEGORY ==")
    report = category_report(trials)
    print(report.to_string(index=False, float_format="%.4g"))

    if has_judge(trials):
        print("\n== JUDGE IMPACT BY CATEGORY (ordinal score vs. BM25 baseline) ==")
        judge_cols = ["category", "n_queries", "n_pairs",
                      "ORD_a0", "ORD_a1", "ORD_delta", "ordinal_wilcoxon_p", "ORD_holm_sig"]
        print(report[judge_cols].to_string(index=False, float_format="%.4g"))

    print("\n== OVERALL (all categories pooled) ==")
    for key, value in overall_report(trials).items():
        print(f"  {key}: {value:.4g}" if isinstance(value, float) else f"  {key}: {value}")

    print("\n== PER-QUERY RECALL@3 (drivers of each category effect) ==")
    for cat in SPOTLIGHT:
        print(f"\n-- {cat}")
        print(per_query_detail(trials, cat).to_string(float_format="%.3f"))

    print("\n== WHERE ROUTING SENDS THE TOP CHUNK ==")
    for cat in SPOTLIGHT:
        print(f"\n-- {cat}")
        print(routing_destinations(trials, cat).to_string(index=False))


if __name__ == "__main__":
    main()
