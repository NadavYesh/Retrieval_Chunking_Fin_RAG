"""
Ranking the retrieval configurations by the reference-guided pairwise judge.

Every configuration is judged against one fixed baseline (L1_BM25_PLAIN_A0),
separately on relevance and completeness. The judge returns 'A' (configuration
preferred), 'B' (baseline preferred), or one of four labels that record no strict
preference. Those four are not the same kind of thing, and the split matters:

  Judge indifference -- a judgment was elicited and expressed no preference.
    Tie    the judge declined to prefer either answer
    A/B    the judge rated both answers equally good on this axis

  'Tie' and 'A/B' are two surface forms of the same verdict and are scored
  identically. They are counted separately only because 'A/B' is the form the
  judge emits when it ties one axis while deciding the other, so the split shows
  where per-axis indifference actually arises.

  Structural -- no judgment was elicited, and none could have differed from a tie.
    same_as_baseline  the configuration's retrieved context, and therefore its
                      generated answer, was identical to the baseline's
    is_baseline       the 34 rows of the baseline compared against itself

Both groups score 0.5 under the ordinal scheme (win 1, tie 0.5, loss 0), over a
fixed denominator of 34 queries. The distinction is reported because the
structural rows carry no information about the judge's behaviour, so the tie rate
is quoted twice: once over all rows, once with them excluded. Scoring ties at 0.5
rather than dropping them keeps the denominator fixed; a wins/(wins+losses) rate
would score more than half the configurations on fewer than 34 comparisons.

Why no Bradley-Terry / Davidson model
  Those models exist to infer a ranking when comparisons form a connected graph
  and A is never judged against C directly. This design is a STAR: all 35
  configurations meet one common reference and never each other. Each
  configuration's 34 comparisons against that reference are therefore already a
  sufficient statistic for its strength, and a Davidson latent strength would be a
  monotone transform of the ordinal mean below -- it would reorder nothing. What
  the model would add (an interval scale, an explicit tie parameter) is not what
  the ranking question needs, and with a 2.2% tie rate the tie parameter is nearly
  degenerate. The statistical work is done instead by the tests below.

The tests
  PRIMARY -- separation of the top configuration from the runner-up. This is the
    test that licenses the phrase "the winner": ranking by a point estimate is
    free, but a unique winner exists only if rank 1 separates from rank 2.
    Wilcoxon signed-rank on the per-query ordinal difference. Pairing on query is
    exact (both configurations answered all 34 queries), so the paired test is
    available at no assumption cost, and it is the sharpest use of n = 34.
  SUPPORTING -- each configuration against the baseline, same paired test,
    Benjamini-Hochberg corrected across the 35 comparisons, since we are screening
    for which configurations clear the reference rather than defending one claim.
  SUPPORTING -- a cluster bootstrap resampling QUERIES (the unit of independence;
    the 36 configurations sharing a query are correlated), giving each
    configuration a probability of being the best. This is what quantifies how
    fragile the top of the ranking is to the particular 34 questions sampled.

Usage:
    python judge_tie_breakdown.py [path/to/workbook.xlsx]
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from statsmodels.stats.multitest import multipletests

DEFAULT_WORKBOOK = (
    Path(__file__).resolve().parents[3]
    / "data/eval_results/ready_for_analysis/analysis"
    / "tsla_pypl-corr_nvda_aapl_no_out-of-corpus-pollution.xlsx"
)

METRICS = ["relevance", "completeness"]
WIN, LOSS = "A", "B"
# 'A/B' and 'Tie' are the same verdict: neither answer preferred on this axis.
INDIFFERENCE = ["Tie", "A/B"]
STRUCTURAL = ["same_as_baseline", "is_baseline"]
TIES = INDIFFERENCE + STRUCTURAL

BASELINE_KEY = "L1_BM25_PLAIN_A0"
N_QUERIES = 34

ORDINAL = {WIN: 1.0, LOSS: 0.0}  # every tie label falls through to 0.5
TIE_SCORE = 0.5

# Joint score over the two axes. A configuration is credited for winning on both,
# penalised for losing on both, and a win on one axis is worth more when the other
# axis ties than when it loses. Equivalently -- and this is asserted at runtime --
# the mean of the two per-axis scores.
TIE = "tie"
JOINT_SCORE = {
    (WIN, WIN):   1.00,
    (WIN, TIE):   0.75,
    (WIN, LOSS):  0.50,
    (TIE, TIE):   0.50,
    (TIE, LOSS):  0.25,
    (LOSS, LOSS): 0.00,
}

N_BOOT = 5000
SEED = 0
OUT_NAME = "judge_ordinal_scores.xlsx"

# The write-up reports the top two configurations as tied co-winners, and pools the
# top three into one arm for the headline test against the baseline.
CO_WINNERS = 2
TRIO_WINNERS = 3


def load_detail(workbook: Path) -> pd.DataFrame:
    df = pd.read_excel(workbook, sheet_name="detail")
    unknown = set(df[METRICS].stack().unique()) - {WIN, LOSS, *TIES}
    if unknown:
        raise ValueError(f"unrecognised judge verdicts: {sorted(unknown)}")
    return df


def add_ordinal(df: pd.DataFrame) -> pd.DataFrame:
    """
    Score each run jointly over the relevance and completeness verdicts.

    Per axis a verdict is a win, a tie (any of the four tie labels) or a loss.
    The pair of outcomes maps to a single run score via JOINT_SCORE.
    """
    for metric in METRICS:
        outcome = df[metric].where(df[metric].isin([WIN, LOSS]), TIE)
        df[f"outcome_{metric}"] = outcome
        df[f"ordinal_{metric}"] = df[metric].map(ORDINAL).fillna(TIE_SCORE)

    pairs = zip(df[f"outcome_{METRICS[0]}"], df[f"outcome_{METRICS[1]}"])
    # JOINT_SCORE is symmetric in the two axes; only one order of each pair is keyed
    df["ordinal"] = [
        JOINT_SCORE[(a, b)] if (a, b) in JOINT_SCORE else JOINT_SCORE[(b, a)]
        for a, b in pairs
    ]

    axis_mean = df[[f"ordinal_{m}" for m in METRICS]].mean(axis=1)
    if not np.allclose(df["ordinal"], axis_mean):
        raise AssertionError("joint score diverges from the per-axis mean")
    return df


def query_by_config(df: pd.DataFrame, value: str = "ordinal") -> pd.DataFrame:
    """34 x 36 matrix: one row per query, one column per configuration."""
    wide = df.pivot_table(index="query", columns="config_key", values=value)
    if wide.isna().any().any():
        raise ValueError("every configuration must answer every query")
    return wide


def config_scores(df: pd.DataFrame) -> pd.DataFrame:
    """Per-configuration ordinal means, ranked. The baseline scores exactly 0.5."""
    joint = df.groupby("config_key")["ordinal"]
    scores = (
        df.groupby("config_key")
        .agg(
            n=("ordinal", "size"),
            ordinal=("ordinal", "mean"),
            relevance=("ordinal_relevance", "mean"),
            completeness=("ordinal_completeness", "mean"),
        )
        .sort_values("ordinal", ascending=False)
    )
    # how the joint score was earned: share of runs at each of the six outcomes
    shares = (
        df.pivot_table(index="config_key", columns="ordinal", values="idx", aggfunc="count")
        .reindex(columns=sorted(set(JOINT_SCORE.values())))
        .fillna(0)
        .div(joint.size(), axis=0)
    )
    shares.columns = [f"share_{c:.2f}" for c in shares.columns]
    scores = scores.join(shares)
    scores.insert(0, "rank", range(1, len(scores) + 1))
    return scores


def config_summary(df: pd.DataFrame, workbook: Path) -> pd.DataFrame:
    """
    The judge ranking joined onto the workbook's own per-config retrieval metrics.

    Reproducing config_summary here rather than reading the ordinal scores back
    into it keeps this script's output self-contained: one sheet carries the
    retrieval metrics and the judge score side by side, which is what the
    retrieval-versus-generation comparison needs.
    """
    retrieval = pd.read_excel(workbook, sheet_name="config_summary").set_index("config_key")
    keep = [
        "level", "mode", "enhance_query_flag", "section_alpha",
        "soft_MRR", "soft_Recall@3", "evidence_hit", "word_recall", "num_recall",
    ]
    summary = config_scores(df).join(retrieval[keep])
    for metric in ("soft_MRR", "soft_Recall@3"):
        summary[f"{metric}_rank"] = summary[metric].rank(ascending=False, method="min").astype(int)
    return summary


def write_workbook(path: Path, runs: pd.DataFrame, summary: pd.DataFrame,
                   verdicts: pd.DataFrame, ties: pd.DataFrame,
                   vs_base: pd.DataFrame, p_best: pd.Series) -> None:
    """One sheet per artefact, so the whole judge analysis travels as a single file."""
    with pd.ExcelWriter(path, engine="openpyxl") as xl:
        runs.to_excel(xl, sheet_name="runs_ordinal", index=False)
        summary.to_excel(xl, sheet_name="config_summary")
        verdicts.to_excel(xl, sheet_name="verdict_distribution")
        ties.to_excel(xl, sheet_name="tie_summary")
        vs_base.to_excel(xl, sheet_name="vs_baseline")
        p_best.rename("p_best").to_excel(xl, sheet_name="p_best")


def paired_wilcoxon(wide: pd.DataFrame, a: str, b: str) -> tuple[float, float]:
    """Mean per-query ordinal difference a - b, and its signed-rank p-value."""
    diff = wide[a] - wide[b]
    nonzero = diff[diff != 0]
    p = 1.0 if nonzero.empty else wilcoxon(nonzero).pvalue
    return diff.mean(), p


def separation_table(wide: pd.DataFrame, scores: pd.DataFrame) -> pd.DataFrame:
    """Rank-1 against every lower-ranked configuration, paired on query."""
    top = scores.index[0]
    rows = []
    for rank, cfg in enumerate(scores.index[1:], start=2):
        delta, p = paired_wilcoxon(wide, top, cfg)
        rows.append({"rank": rank, "config_key": cfg, "mean_delta": delta, "p": p})
    return pd.DataFrame(rows).set_index("config_key")


def separation_test(wide: pd.DataFrame, scores: pd.DataFrame, alpha: float = 0.05) -> dict:
    """
    PRIMARY TEST. Does the top-ranked configuration separate from those below it?

    A ranking always names a first place; only this test says whether that first
    place is distinguishable from the next. Rank 1 is compared against each
    lower-ranked configuration in turn, and the LEADING SET is every configuration
    down to (but excluding) the first one that separates. Reporting the leading set
    rather than a single winner is the honest summary whenever it has size > 1: the
    ranking inside it is a point-estimate ordering that the data do not support.

    CO_WINNERS names the first two as tied leaders, which is the reporting choice
    made in the write-up. The leading set is computed independently so the write-up
    cannot silently understate how far the tie extends.
    """
    table = separation_table(wide, scores)
    separated = table.index[table["p"] < alpha]
    first_sep = table.loc[separated[0], "rank"] if len(separated) else len(scores) + 1
    leaders = list(scores.index[: int(first_sep) - 1])

    co = list(scores.index[:CO_WINNERS])
    delta, p = paired_wilcoxon(wide, co[0], co[1])
    return {
        "co_winners": co,
        "trio_winners": list(scores.index[:TRIO_WINNERS]),
        "co_winner_delta": delta,
        "co_winner_p": p,
        "leaders": leaders,
        "first_separated_rank": int(first_sep),
        "table": table,
    }


def pooled_vs_baseline(df: pd.DataFrame, winners: list[str]) -> tuple[float, float]:
    """
    The tied leaders pooled into one arm, tested against the baseline.

    Averaging the winners' per-query scores before testing asks whether the leading
    group clears the reference, which is the only claim their mutual
    indistinguishability permits. Works for any number of winners.
    """
    wide = query_by_config(df)
    pooled = wide[winners].mean(axis=1)
    diff = pooled - wide[BASELINE_KEY]
    nonzero = diff[diff != 0]
    p = 1.0 if nonzero.empty else wilcoxon(nonzero).pvalue
    return diff.mean(), p


def versus_baseline(wide: pd.DataFrame, scores: pd.DataFrame) -> pd.DataFrame:
    """Every configuration against the reference, BH-corrected across the family."""
    rows = []
    for cfg in scores.index:
        if cfg == BASELINE_KEY:
            continue
        delta, p = paired_wilcoxon(wide, cfg, BASELINE_KEY)
        rows.append({"config_key": cfg, "mean_delta": delta, "p": p})
    out = pd.DataFrame(rows).set_index("config_key")
    out["p_bh"] = multipletests(out["p"], method="fdr_bh")[1]
    out["beats_baseline"] = out["p_bh"] < 0.05
    return out


def bootstrap_best(wide: pd.DataFrame, n_boot: int = N_BOOT) -> pd.Series:
    """
    params: 
        wide = n_configs X n_queries 
    P(configuration is best), resampling QUERIES with replacement.

    The query is the unit of independence: the 36 configurations sharing a query
    are correlated, so resampling runs would understate the uncertainty.
    """
    rng = np.random.default_rng(SEED)
    queries = wide.index.to_numpy()
    wins = np.zeros(wide.shape[1], dtype=int)
    for _ in range(n_boot):
        means = wide.loc[rng.choice(queries, len(queries), replace=True)].mean().to_numpy() # this collapses across queries
        wins[np.argmax(means)] += 1
    print("===============================\nBootstrap Details:\n")
    for m in means:
        print(m)
    print("===============================\n\n")

    return pd.Series(wins / n_boot, index=wide.columns).sort_values(ascending=False)


def bootstrap_ci(wide: pd.DataFrame, a: str, b: str, n_boot: int = N_BOOT) -> tuple[float, float]:
    """95% cluster-bootstrap CI on the mean per-query ordinal gap a - b."""
    rng = np.random.default_rng(SEED)
    diff = (wide[a] - wide[b]).to_numpy()
    means = [rng.choice(diff, diff.size, replace=True).mean() for _ in range(n_boot)]
    return tuple(np.percentile(means, [2.5, 97.5]))


def verdict_table(df: pd.DataFrame) -> pd.DataFrame:
    """Counts of every verdict label, per metric, in reporting order."""
    order = [WIN, LOSS, *INDIFFERENCE, *STRUCTURAL]
    table = pd.DataFrame(
        {m: df[m].value_counts().reindex(order).fillna(0).astype(int) for m in METRICS}
    )
    table.index.name = "verdict"
    return table


def tie_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Tie totals and rates, with and without the structural (non-judged) rows."""
    n = len(df)
    n_structural = int(df[METRICS[0]].isin(STRUCTURAL).sum())
    rows = {}
    for m in METRICS:
        indiff = int(df[m].isin(INDIFFERENCE).sum())
        struct = int(df[m].isin(STRUCTURAL).sum())
        rows[m] = {
            "decided": int(df[m].isin([WIN, LOSS]).sum()),
            "judge_indifference": indiff,
            "structural": struct,
            "ties_total": indiff + struct,
            "tie_pct": 100 * (indiff + struct) / n,
            # structural rows are excluded from numerator and denominator alike:
            # they were never put to the judge, so they cannot inform its tie rate
            "tie_pct_judged_only": 100 * indiff / (n - n_structural),
        }
    return pd.DataFrame(rows)


def tie_agreement(df: pd.DataFrame) -> pd.DataFrame:
    """Do the two dimensions tie on the same rows?"""
    return pd.crosstab(
        df["relevance"].isin(TIES).rename("relevance_tie"),
        df["completeness"].isin(TIES).rename("completeness_tie"),
    )


def decided_denominators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Per-configuration count of decided comparisons, which is the denominator a
    wins/(wins+losses) rate would use. Spread away from N_QUERIES is the bias the
    ordinal scoring avoids. The baseline is excluded: it has no decided rows.
    """
    rows = {}
    for m in METRICS:
        decided = df[df[m].isin([WIN, LOSS])].groupby("config_key").size()
        decided = decided.drop(index=BASELINE_KEY, errors="ignore")
        rows[m] = {
            "configs": len(decided),
            "min": int(decided.min()),
            "max": int(decided.max()),
            "mean": decided.mean(),
            f"at_full_{N_QUERIES}": int((decided == N_QUERIES).sum()),
        }
    return pd.DataFrame(rows)


def main() -> None:
    workbook = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_WORKBOOK
    df = add_ordinal(load_detail(workbook))

    print(f"{workbook.name}\n{len(df)} runs judged against baseline {BASELINE_KEY}\n")
    # judge: 
    print("Verdict distribution")
    print(verdict_table(df).to_string(), "\n")

    print("Tie summary")
    print(tie_summary(df).round(1).to_string(), "\n")

    print("Tie agreement across the two dimensions")
    print(tie_agreement(df).to_string(), "\n")

    print(f"Decided-comparison denominators per configuration (full = {N_QUERIES})")
    print(decided_denominators(df).round(1).to_string(), "\n")

    scores = config_summary(df, workbook)
    wide = query_by_config(df)


    print("Configuration ranking by joint ordinal judge score")
    cols = ["rank", "n", "ordinal", "relevance", "completeness",
            "soft_MRR_rank", "soft_Recall@3_rank"]
    print(scores.head(8)[cols].round(3).to_string())
    print(f"  ... baseline {BASELINE_KEY} scores {scores.loc[BASELINE_KEY, 'ordinal']:.3f} "
          f"at rank {scores.loc[BASELINE_KEY, 'rank']}")
    print(f"  {(scores['ordinal'] > 0.5).sum()} of {len(scores)} configurations exceed 0.5\n")

    sep = separation_test(wide, scores)
    first, second = sep["co_winners"]
    verdict = "SEPARATE" if sep["co_winner_p"] < 0.05 else "do NOT separate"

    # print("separation test results:")
    # for el in sep:
    #     print (el)

    print("PRIMARY TEST -- rank 1 vs rank 2 (Wilcoxon signed-rank, paired on query)")
    print(f"  {first} vs {second}")
    print(f"  mean ordinal gap {sep['co_winner_delta']:+.3f}, "
          f"p = {sep['co_winner_p']:.3f}  ->  {verdict}")
    if sep["co_winner_p"] >= 0.05:
        print(f"  Reported as tied co-winners: {first} and {second}")

    print(f"\n  Rank 1 against each lower rank (leading set ends where p < 0.05):")
    print("  " + sep["table"].head(12).round(4).to_string().replace("\n", "\n  "))
    print(f"\n  Leading set = ranks 1..{len(sep['leaders'])}, i.e. {len(sep['leaders'])} "
          f"configurations statistically indistinguishable from rank 1.")
    print(f"  First separation at rank {sep['first_separated_rank']}.")
    if len(sep["leaders"]) > CO_WINNERS:
        print(f"  NOTE: naming {CO_WINNERS} co-winners is a reporting choice, not a "
              f"data-driven cut -- the tie extends to rank {len(sep['leaders'])}.\n")
    else:
        print()




    # most valuable
    delta, p = pooled_vs_baseline(df, sep["trio_winners"])
    print(f"PRIMARY TEST -- {TRIO_WINNERS} best pooled vs baseline: "
          f"mean gap {delta:+.3f}, p = {p:.4f}\n")

    print("SUPPORTING -- configurations that clear the baseline (Benjamini–Hochberg-corrected)")
    vb = versus_baseline(wide, scores)
    print(vb[vb["beats_baseline"]].round(4).to_string())
    print(f"  {int(vb['beats_baseline'].sum())} of {len(vb)} survive BH at q < 0.05")
    print(f"  ({int((vb['p'] < 0.05).sum())} would at raw p < 0.05, "
          f"smallest q = {vb['p_bh'].min():.3f})\n")

    for cfg in sep["co_winners"]:
        lo, hi = bootstrap_ci(wide, cfg, BASELINE_KEY)
        print(f"SUPPORTING -- {cfg} vs baseline: 95% cluster-bootstrap CI [{lo:+.3f}, {hi:+.3f}]")

    p_best = bootstrap_best(wide)
    print(f"\nSUPPORTING -- P(configuration is best), {N_BOOT} query-cluster bootstraps")
    print(p_best.head(6).round(3).to_string())
    print(f"  P(best is one of the {CO_WINNERS} co-winners) = "
          f"{p_best[sep['co_winners']].sum():.3f}")

    out_path = workbook.parent / OUT_NAME
    write_workbook(out_path, df, scores, verdict_table(df), tie_summary(df), vb, p_best)
    print(f"\nJudge analysis written to {out_path}")


if __name__ == "__main__":
    main()
