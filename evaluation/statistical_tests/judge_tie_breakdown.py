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
    is_baseline       the rows of the baseline compared against itself

Both groups score 0.5 under the ordinal scheme (win 1, tie 0.5, loss 0), over a
denominator fixed at the full query count. The distinction is reported because the
structural rows carry no information about the judge's behaviour, so the tie rate
is quoted twice: once over all rows, once with them excluded. Scoring ties at 0.5
rather than dropping them keeps the denominator fixed; a wins/(wins+losses) rate
would score more than half the configurations on fewer than the full set of
comparisons.

Why no Bradley-Terry / Davidson model
  Those models exist to infer a ranking when comparisons form a connected graph
  and A is never judged against C directly. This design is a STAR: every
  configuration meets one common reference and never another configuration. Each
  configuration's comparisons against that reference are therefore already a
  sufficient statistic for its strength, and a Davidson latent strength would be a
  monotone transform of the ordinal mean below -- it would reorder nothing. What
  the model would add (an interval scale, an explicit tie parameter) is not what
  the ranking question needs, and with a tie rate this low the tie parameter is
  nearly degenerate. The statistical work is done instead by the tests below.

The tests
  PRIMARY -- separation of the top configuration from the runner-up. This is the
    test that licenses the phrase "the winner": ranking by a point estimate is
    free, but a unique winner exists only if rank 1 separates from rank 2.
    Wilcoxon signed-rank on the per-query ordinal difference. Pairing on query is
    exact (every configuration answered every query), so the paired test is
    available at no assumption cost, and it is the sharpest use of the sample.
  SUPPORTING -- each configuration against the baseline, same paired test,
    Benjamini-Hochberg corrected across the family, since we are screening for
    which configurations clear the reference rather than defending one claim.
  SUPPORTING -- a cluster bootstrap resampling QUERIES (the unit of independence;
    the configurations sharing a query are correlated), giving each configuration a
    probability of being the best. This is what quantifies how fragile the top of
    the ranking is to the particular questions sampled.

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
    """queries x configurations matrix: one row per query, one column per configuration."""
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
        "soft_Recall@3", "word_recall", "num_recall",
    ]
    summary = config_scores(df).join(retrieval[keep])
    for metric in ("soft_Recall@3",):
        summary[f"{metric}_rank"] = summary[metric].rank(ascending=False, method="min").astype(int)
    return summary


def baseline_summary(comparisons: dict[str, dict]) -> pd.DataFrame:
    """
    The leader/pooled comparisons against the sparse baseline, one row per arm, for
    the judge's ordinal score and each retrieval metric side by side. This is the
    table the write-up quotes: SQ1 and SQ2 read off the same reference by the same
    test, so their effect sizes and p-values are directly comparable.
    """
    rows = []
    for metric, cmp in comparisons.items():
        for arm, key in (("leader", "leader_vs_baseline"), ("pooled_top3", "pooled_vs_baseline")):
            delta, p = cmp[key]
            rows.append({
                "metric": metric,
                "arm": arm,
                "configs": cmp["leader"] if arm == "leader" else ", ".join(cmp["top3"]),
                "mean_delta": delta,
                "p": p,
                "significant": p < 0.05,
            })
    return pd.DataFrame(rows).set_index(["metric", "arm"])


def write_workbook(path: Path, runs: pd.DataFrame, summary: pd.DataFrame,
                   verdicts: pd.DataFrame, ties: pd.DataFrame,
                   vs_base: pd.DataFrame, p_best: pd.Series,
                   vs_base_pooled: pd.DataFrame | None = None,
                   retrieval: dict[str, dict] | None = None,
                   marginals: pd.DataFrame | None = None,
                   marginals_alpha0: pd.DataFrame | None = None) -> None:
    """One sheet per artefact, so the whole judge (and retrieval) analysis travels as a single file."""
    with pd.ExcelWriter(path, engine="openpyxl") as xl:
        runs.to_excel(xl, sheet_name="runs_ordinal", index=False)
        summary.to_excel(xl, sheet_name="config_summary")
        verdicts.to_excel(xl, sheet_name="verdict_distribution")
        ties.to_excel(xl, sheet_name="tie_summary")
        vs_base.to_excel(xl, sheet_name="vs_baseline")
        p_best.rename("p_best").to_excel(xl, sheet_name="p_best")
        if vs_base_pooled is not None:
            vs_base_pooled.to_excel(xl, sheet_name="vs_baseline_leader_pooled")
        if marginals is not None:
            marginals.to_excel(xl, sheet_name="marginals")
        if marginals_alpha0 is not None:
            marginals_alpha0.to_excel(xl, sheet_name="marginals_alpha0")
        for metric, report in (retrieval or {}).items():
            tag = metric.replace("soft_", "").replace("@", "")  # e.g. "soft_Recall@3" -> "Recall3"
            report["scores"].rename(metric).to_excel(xl, sheet_name=f"retr_{tag}_scores")
            report["sep"]["table"].to_excel(xl, sheet_name=f"retr_{tag}_separation")
            report["p_best"].rename("p_best").to_excel(xl, sheet_name=f"retr_{tag}_bootstrap")


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


def pooled_vs_baseline(df: pd.DataFrame, winners: list[str], metric: str = "ordinal") -> tuple[float, float]:
    """
    The tied leaders pooled into one arm, tested against the baseline.

    Averaging the winners' per-query scores before testing asks whether the leading
    group clears the reference, which is the only claim their mutual
    indistinguishability permits. Works for any number of winners, and for any
    per-query metric column (judge ordinal score, or a retrieval metric).
    """
    wide = query_by_config(df, value=metric)
    pooled = wide[winners].mean(axis=1)
    diff = pooled - wide[BASELINE_KEY]
    nonzero = diff[diff != 0]
    p = 1.0 if nonzero.empty else wilcoxon(nonzero).pvalue
    return diff.mean(), p


def baseline_comparisons(df: pd.DataFrame, wide: pd.DataFrame, scores: pd.DataFrame | pd.Series,
                         metric: str = "ordinal") -> dict:
    """
    The two headline comparisons against the sparse baseline, at two levels of pooling.

    LEADER -- the single top-ranked configuration against the baseline. The sharpest
      claim available, and the most fragile: it is conditioned on which configuration
      happened to come first, and the leading set shows that choice is usually not
      supported by the data.
    POOLED TOP THREE -- the top three configurations excluding the baseline, averaged
      per query into one arm and then tested. Whenever the leaders are mutually
      indistinguishable, this is the claim their tie actually licenses: that the
      leading group clears the reference, not that any one member of it does.

    Metric-agnostic, so the judge ordinal score and the retrieval metrics are put
    through an identical procedure and their numbers are directly comparable.
    """
    leader = scores.index[0]
    leader_delta, leader_p = paired_wilcoxon(wide, leader, BASELINE_KEY)

    top3 = [c for c in scores.index if c != BASELINE_KEY][:TRIO_WINNERS]
    pooled_delta, pooled_p = pooled_vs_baseline(df, top3, metric=metric)

    return {
        "leader": leader,
        "leader_vs_baseline": (leader_delta, leader_p),
        "top3": top3,
        "pooled_vs_baseline": (pooled_delta, pooled_p),
    }


def print_baseline_comparisons(cmp: dict, indent: str = "  ") -> None:
    """The leader/pooled block, printed identically for the judge and for retrieval."""
    ld, lp = cmp["leader_vs_baseline"]
    pd_, pp = cmp["pooled_vs_baseline"]
    print(f"{indent}Against the sparse baseline ({BASELINE_KEY}):")
    print(f"{indent}  leader {cmp['leader']}")
    print(f"{indent}    vs. sparse baseline:                       delta={ld:+.4f}  p={lp:.4f}")
    print(f"{indent}  pooled top three (excl. baseline): {', '.join(cmp['top3'])}")
    print(f"{indent}    vs. sparse baseline:                       delta={pd_:+.4f}  p={pp:.4f}")


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

    The query is the unit of independence: the configurations sharing a query are
    correlated, so resampling runs would understate the uncertainty.
    """
    rng = np.random.default_rng(SEED)
    queries = wide.index.to_numpy()
    wins = np.zeros(wide.shape[1], dtype=int)
    for _ in range(n_boot):
        means = wide.loc[rng.choice(queries, len(queries), replace=True)].mean().to_numpy()  # collapses across queries
        wins[np.argmax(means)] += 1
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


def n_queries(df: pd.DataFrame) -> int:
    """The judged query count, read off the data rather than hard-coded."""
    return int(df["query"].nunique())


def decided_denominators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Per-configuration count of decided comparisons, which is the denominator a
    wins/(wins+losses) rate would use. Spread away from the full query count is the
    bias the ordinal scoring avoids. The baseline is excluded: it has no decided rows.
    """
    full = n_queries(df)
    rows = {}
    for m in METRICS:
        decided = df[df[m].isin([WIN, LOSS])].groupby("config_key").size()
        decided = decided.drop(index=BASELINE_KEY, errors="ignore")
        rows[m] = {
            "configs": len(decided),
            "min": int(decided.min()),
            "max": int(decided.max()),
            "mean": decided.mean(),
            f"at_full_{full}": int((decided == full).sum()),
        }
    return pd.DataFrame(rows)


# ── Retrieval metrics (Section~\ref{sec:results-retrieval} of the thesis) ─────
#
# The judge ranking above answers SQ2 on the ordinal score; this reruns the exact
# same battery of tests (separation from the field, a query-cluster bootstrap,
# leader/pooled/marginal comparisons against the sparse baseline) on the two
# retrieval metrics instead, which is what SQ1 is actually decided on. Every
# helper above is metric-agnostic (query_by_config, separation_test,
# pooled_vs_baseline, bootstrap_best all take a `value`/`metric` argument), so
# nothing above needed to change -- this section only adds the retrieval-specific
# plumbing: per-config mean scores (no judge verdict labels involved) and the
# hybrid/dense-vs-sparse marginal comparison that has no judge analogue.

# soft_NDCG@5 is no longer computed (graded gains over a binary relevance label add
# nothing). soft_MRR is rank-variant, which used to make it meaningless for the
# parent-fetch configs — level-2/3 child hits were collapsed onto their level-1
# parents through a set, discarding the fused rank. _collapse_to_parents now ranks
# parents by an RRF-sum over their child ranks, so the position of a retrieved chunk
# is meaningful at every level and soft_MRR is reported alongside soft_Recall@3.
RETRIEVAL_METRICS = ["soft_Recall@3", "soft_MRR"]
MARGINAL_MODES = ["hybrid", "dense"]
# The factors marginalised over when a mode is compared against sparse: the sweep is
# fully crossed, so each mode has the same 12 sub-configs (2 alphas x 2 enhancement
# settings x 3 levels) and the pooled arms are balanced by construction.
MARGINAL_FACTORS = ["section_alpha", "enhance_query_flag", "level"]


def metric_scores(df: pd.DataFrame, metric: str) -> pd.Series:
    """Per-configuration mean of a retrieval metric, ranked descending. No judge involved."""
    return df.groupby("config_key")[metric].mean().sort_values(ascending=False)


def marginal_mode_vs_sparse(df: pd.DataFrame, metric: str, mode: str) -> tuple[float, float]:
    """
    Paired Wilcoxon test on the retrieval mode, marginalising over every other factor.

    Each arm is one mode pooled over all of its sub-configurations -- section_alpha,
    query enhancement and chunking level alike -- averaged per query, so `mode` is the
    only thing that differs between the two arms. The sparse arm is therefore all 12
    sparse runs, not the single BASELINE_KEY configuration: this is the marginal read
    on SQ1 (hybrid/dense vs. sparse as retrieval strategies), which asks a coarser
    question than the per-configuration tests and does not depend on any single
    configuration being the leader.
    """
    balance = df.groupby("mode")[MARGINAL_FACTORS].nunique()
    if balance.nunique().gt(1).any():
        raise ValueError(f"modes are not crossed over the same sub-configs:\n{balance}")

    pivot = df.pivot_table(index="query", columns="mode", values=metric, aggfunc="mean")
    diff = pivot[mode] - pivot["sparse"]
    nonzero = diff[diff != 0]
    p = 1.0 if nonzero.empty else wilcoxon(nonzero).pvalue
    return diff.mean(), p


# ── Marginal effects of the design axes ──────────────────────────────────────
#
# The per-configuration tests above cannot separate one cell of the grid from
# another: 36 configurations over 108 queries leaves each cell estimated from too
# little data, and the leading set runs most of the way down the ranking. The
# design axes, however, are estimated from every run in the sweep -- each arm of a
# contrast pools 12 or 18 configurations -- and that is the aggregation at which
# the sweep has power. These are the numbers the write-up's SQ1 and SQ2 claims
# rest on.
#
# Each contrast pools one arm over every OTHER axis, averages within a query, and
# pairs on the query, exactly as marginal_mode_vs_sparse does for the mode axis.
# The sweep is fully crossed, so the arms are balanced by construction; the
# balance is asserted rather than assumed.
#
# Direction is always TREATMENT minus CONTROL, so a positive delta always means
# "the thing being tested helped". For section routing that makes the contrast
# alpha=1 - alpha=0 (routed minus unrouted), and its delta is negative: routing
# hurts. Writing it the other way round would report a positive number for a
# harmful treatment.
ALL_METRICS = ["ordinal", *RETRIEVAL_METRICS]

# (label, column, treatment values, control values)
MARGINAL_CONTRASTS = [
    ("hybrid - sparse",      "mode",               ["hybrid"], ["sparse"]),
    ("dense - sparse",       "mode",               ["dense"],  ["sparse"]),
    ("hybrid - dense",       "mode",               ["hybrid"], ["dense"]),
    ("child (L2/L3) - L1",   "level",              [2, 3],     [1]),
    ("L3 - L2",              "level",              [3],        [2]),
    ("enhanced - plain",     "enhance_query_flag", [1],        [0]),
    ("alpha=1 - alpha=0",    "section_alpha",      [1],        [0]),
]

# Every axis except the one under test; the arms are pooled over these.
AXES = ["mode", "level", "enhance_query_flag", "section_alpha"]


def marginal_contrast(df: pd.DataFrame, metric: str, column: str,
                      treatment: list, control: list) -> tuple[float, float, float, float]:
    """
    One design axis, marginalised over the others, paired on query.

    Both arms are pooled over every axis other than `column` and averaged within a
    query, so `column` is the only systematic difference between them. Returns the
    treatment and control means, their difference, and the signed-rank p-value.

    This asks a coarser question than the per-configuration tests -- "does hybrid
    retrieval help, over the whole sweep?" rather than "is this cell the best?" --
    and it is the only question the sample size supports answering.
    """
    arms = {"treatment": treatment, "control": control}
    others = [ax for ax in AXES if ax != column]

    # The contrast is only interpretable if both arms span the same sub-configs on
    # every other axis; otherwise the delta confounds `column` with whatever is
    # unbalanced. Crossed sweep => this holds, but a dropped run would break it.
    spans = {
        arm: df[df[column].isin(values)].groupby(others).size().index
        for arm, values in arms.items()
    }
    if set(spans["treatment"]) != set(spans["control"]):
        raise ValueError(f"arms of '{column}' are not crossed over the same sub-configs")

    means = {
        arm: df[df[column].isin(values)].groupby("query")[metric].mean()
        for arm, values in arms.items()
    }
    paired = pd.DataFrame(means).dropna()
    diff = paired["treatment"] - paired["control"]
    nonzero = diff[diff != 0]
    p = 1.0 if nonzero.empty else wilcoxon(nonzero).pvalue
    return paired["treatment"].mean(), paired["control"].mean(), diff.mean(), p


def marginal_table(df: pd.DataFrame, metrics: list[str] = None,
                   contrasts: list = None) -> pd.DataFrame:
    """Every contrast against every metric: the table the write-up reports verbatim."""
    metrics = metrics or ALL_METRICS
    contrasts = contrasts or MARGINAL_CONTRASTS
    rows = {}
    for label, column, treatment, control in contrasts:
        row = {}
        for metric in metrics:
            t, c, delta, p = marginal_contrast(df, metric, column, treatment, control)
            row[(metric, "treat")] = t
            row[(metric, "ctrl")] = c
            row[(metric, "delta")] = delta
            row[(metric, "p")] = p
        rows[label] = row
    out = pd.DataFrame(rows).T
    out.columns = pd.MultiIndex.from_tuples(out.columns)
    out.index.name = "contrast"
    return out


def retrieval_report(df: pd.DataFrame, metric: str) -> dict:
    """
    One retrieval metric, put through the same tests as the judge ordinal score:
    ranking, leader-separation, bootstrap P(best), and baseline comparisons at
    three levels of pooling (single leader, pooled non-sparse top three, marginal
    mode-vs-sparse).
    """
    scores = metric_scores(df, metric)
    wide = query_by_config(df, value=metric)
    sep = separation_test(wide, scores)
    p_best = bootstrap_best(wide)

    cmp = baseline_comparisons(df, wide, scores, metric=metric)

    marginal = {
        mode: marginal_mode_vs_sparse(df, metric, mode) for mode in MARGINAL_MODES
    }

    return {
        "scores": scores,
        "sep": sep,
        "p_best": p_best,
        "baseline": cmp,
        "marginal_vs_sparse": marginal,
    }


def main() -> None:
    workbook = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_WORKBOOK
    workbook = Path("/Users/nadavsmacbookair/Desktop/Thesis/data/eval_results/analysis/analysis_multi_AAPL-IFF-JPM-KO-MA-MDLZ-NEM-NVDA-PYPL-TSLA-YUM_20260713_2132_clean_dedup.xlsx")
    df = add_ordinal(load_detail(workbook))

    print(f"{workbook.name}\n{len(df)} runs judged against baseline {BASELINE_KEY}\n")
    # judge: 
    print("Verdict distribution")
    print(verdict_table(df).to_string(), "\n")

    print("Tie summary")
    print(tie_summary(df).round(1).to_string(), "\n")

    print("Tie agreement across the two dimensions")
    print(tie_agreement(df).to_string(), "\n")

    print(f"Decided-comparison denominators per configuration (full = {n_queries(df)})")
    print(decided_denominators(df).round(1).to_string(), "\n")

    scores = config_summary(df, workbook)
    wide = query_by_config(df)


    print("Configuration ranking by joint ordinal judge score")
    cols = ["rank", "n", "ordinal", "relevance", "completeness",
            "soft_Recall@3_rank"]
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




    # most valuable -- the same two comparisons the retrieval metrics get below, so
    # the judge's read on the baseline and SQ1's read on it are computed identically.
    judge_cmp = baseline_comparisons(df, wide, scores)
    print("PRIMARY TEST -- the judge's ordinal score against the sparse baseline")
    print_baseline_comparisons(judge_cmp)
    print()

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

    # ── Retrieval metrics: same battery of tests, on soft_Recall@3 ──
    print("\n" + "=" * 70)
    print("RETRIEVAL METRICS (soft_Recall@3) -- SQ1, same procedure as above")
    print("=" * 70)

    retrieval_reports: dict[str, dict] = {}
    for metric in RETRIEVAL_METRICS:
        report = retrieval_report(df, metric)
        retrieval_reports[metric] = report
        r_scores, r_sep, r_p_best = report["scores"], report["sep"], report["p_best"]

        print(f"\n-- {metric} --")
        print(r_scores.head(8).round(4).to_string())
        base_rank = r_scores.index.get_loc(BASELINE_KEY) + 1
        print(f"  baseline {BASELINE_KEY}: {r_scores[BASELINE_KEY]:.4f} (rank {base_rank})")

        print(f"\n  Leader {r_scores.index[0]} against each lower-scoring configuration:")
        print("  " + r_sep["table"].head(10).round(4).to_string().replace("\n", "\n  "))
        print(f"  Leading set = {len(r_sep['leaders'])} of {len(r_scores)} configurations "
              f"indistinguishable from rank 1.")

        print(f"\n  P(best), {N_BOOT} query-cluster bootstraps:")
        print(r_p_best.head(6).round(3).to_string())

        print()
        print_baseline_comparisons(report["baseline"], indent="  ")
        for mode, (md, mp) in report["marginal_vs_sparse"].items():
            print(f"    {mode:<6} vs. sparse (marginal over alpha/enhance/level):  "
                  f"delta={md:+.4f}  p={mp:.4f}")

    pooled = baseline_summary(
        {"judge_ordinal": judge_cmp}
        | {m: r["baseline"] for m, r in retrieval_reports.items()}
    )
    print("\n" + "=" * 70)
    print("LEADER / POOLED-TOP-3 vs SPARSE BASELINE -- judge and retrieval side by side")
    print("=" * 70)
    print(pooled.drop(columns="configs").round(4).to_string())

    # ── Marginal effects: the aggregation the conclusions are drawn at ──
    print("\n" + "=" * 70)
    print("MARGINAL EFFECT OF EACH DESIGN AXIS (pooled over the other three)")
    print("Direction is treatment - control: a positive delta means the treatment helped.")
    print("=" * 70)
    marginals = marginal_table(df)
    print(marginals.round(4).to_string())

    # The mode axis again, inside the alpha=0 arm only. alpha=1 is shown below to be
    # both harmful and variance-destroying (it collapses the retrieval-mode signal),
    # so pooling over it dilutes the mode contrast; alpha=0 is the operating regime.
    alpha0 = df[df["section_alpha"] == 0]
    mode_contrasts = [c for c in MARGINAL_CONTRASTS if c[1] == "mode"]
    marginals_a0 = marginal_table(alpha0, contrasts=mode_contrasts)
    print("\nRetrieval mode within alpha=0 only (the operating regime)")
    print(marginals_a0.round(4).to_string())

    out_path = workbook.parent / OUT_NAME
    write_workbook(out_path, df, scores, verdict_table(df), tie_summary(df), vb, p_best,
                   vs_base_pooled=pooled, retrieval=retrieval_reports,
                   marginals=marginals, marginals_alpha0=marginals_a0)
    print(f"\nJudge + retrieval analysis written to {out_path}")


if __name__ == "__main__":
    main()
