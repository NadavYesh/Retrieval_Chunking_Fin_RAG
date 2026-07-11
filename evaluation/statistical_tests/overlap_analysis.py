"""
Context-collapse (overlap) analysis for the retrieval configuration sweep.

Reproduces the grouping performed by `_find_overlap_groups` in evaluation_run.py:
runs are bucketed by (finder_id, ticker_filter, retrieved_ids), where retrieved_ids
is the pipe-joined chunk-ID list *in rank order*. A bucket of size >= 2 is an overlap
group: those configurations returned the same chunks in the same ranks, so no metric
computed downstream can distinguish them.

IDENTITY IS ALWAYS ORDERED. Two runs collapse only if they return the same chunk IDs
in the same positions. Comparing unordered sets instead inflates every count -- 193
groups and 645 runs rather than 172 and 501 -- because it treats a re-ranking of the
same five chunks as no change. soft_MRR and soft_Recall@3 both react to re-ranking,
so the ordered definition is the one that matches the metrics.

TWO DIFFERENT "OVERLAP RATES". They answer different questions and differ by ~14pp:

  runs_in_groups (501, 40.9%)
      How many runs share their exact context with at least one other run. This is
      the exposure figure: the share of the grid whose result was not uniquely
      determined by its own configuration.

  redundant_runs (329, 26.9%)
      How many runs could be deleted with zero information loss, i.e. every group
      member except one representative (501 - 172). This is the compression figure,
      and it is the one that governs effective sample size: 1224 runs carry only
      895 distinct (query, context) observations.

Quote 40.9% when describing how much of the grid is contaminated; quote 26.9% (or
"895 effective runs") when discussing statistical power. They are not interchangeable.

Usage:
    python overlap_analysis.py [path/to/workbook.xlsx]
"""

from __future__ import annotations

import sys
from itertools import combinations
from pathlib import Path

import pandas as pd

DEFAULT_WORKBOOK = (
    Path(__file__).resolve().parents[3]
    / "data/eval_results/ready_for_analysis/analysis"
    / "tsla_pypl-corr_nvda_aapl_no_out-of-corpus-pollution.xlsx"
)

# Mirrors _find_overlap_groups: identity is (question, ticker, ordered chunk list).
GROUP_KEY = ["finder_id", "ticker_filter", "ids"]
AXES = ["level", "mode", "enhanced", "section_alpha"]
LEVELS = [1, 2, 3]


def load_runs(workbook: Path) -> pd.DataFrame:
    df = pd.read_excel(workbook, sheet_name="detail")
    df["ids"] = df["retrieved_ids"].fillna("")
    # Runs that retrieved nothing are not a meaningful "overlap" (see _find_overlap_groups).
    df = df[df["ids"] != ""].copy()
    df["key"] = list(df[GROUP_KEY].itertuples(index=False, name=None))
    df["gsize"] = df["key"].map(df.groupby("key").size())
    return df


def headline(runs: pd.DataFrame) -> dict:
    sizes = runs.groupby("key").size()
    groups = sizes[sizes >= 2]
    n = len(runs)
    return {
        "total_runs": n,
        "groups": len(groups),
        "runs_in_groups": int(groups.sum()),
        "pct_runs_in_groups": groups.sum() / n * 100,
        "redundant_runs": int(groups.sum() - len(groups)),
        "pct_redundant": (groups.sum() - len(groups)) / n * 100,
        "distinct_contexts": runs["key"].nunique(),
        "queries_with_a_group": runs[runs.gsize >= 2]["finder_id"].nunique(),
        "total_queries": runs["finder_id"].nunique(),
        "group_size_distribution": dict(sorted(groups.value_counts().items())),
    }


def axis_table(runs: pd.DataFrame) -> pd.DataFrame:
    """Per axis value: share of runs collapsed; per axis: share of groups it varies within."""
    collapsed = runs[runs.gsize >= 2]
    n_groups = collapsed["key"].nunique()
    rows = []
    for axis in AXES:
        varies = collapsed.groupby("key")[axis].nunique()
        varies_pct = (varies > 1).sum() / n_groups * 100
        for value in sorted(runs[axis].dropna().unique()):
            sub = runs[runs[axis] == value]
            hit = int((sub.gsize >= 2).sum())
            rows.append(
                {
                    "axis": axis,
                    "value": value,
                    "runs_collapsed": hit,
                    "runs_total": len(sub),
                    "pct": hit / len(sub) * 100,
                    "varies_within_group_pct": varies_pct,
                }
            )
    return pd.DataFrame(rows)


def level_composition(runs: pd.DataFrame) -> pd.DataFrame:
    """Which chunking levels co-occur inside each overlap group."""
    collapsed = runs[runs.gsize >= 2]
    levels = collapsed.groupby("key")["level"].apply(lambda s: tuple(sorted(set(s))))
    sizes = collapsed.groupby("key").size()
    frame = pd.DataFrame({"levels": levels, "size": sizes})
    out = frame.groupby("levels").agg(groups=("size", "size"), runs=("size", "sum"))
    return out.sort_values("runs", ascending=False)


def level_attribution(runs: pd.DataFrame) -> pd.DataFrame:
    """
    For each level: of its collapsed runs, how many have a partner at a DIFFERENT
    level (cross-level duplication) versus a partner at the SAME level? A run can
    have both. Runs whose ONLY partners are at another level are collapsed solely
    because of cross-level duplication -- those are what merging levels would fix.
    """
    collapsed = runs[runs.gsize >= 2]
    members = collapsed.groupby("key")["level"].apply(list)
    rows = []
    for level in LEVELS:
        sub = collapsed[collapsed.level == level]
        cross = same = 0
        for key in sub["key"]:
            peers = members[key]
            cross += any(p != level for p in peers)
            same += sum(1 for p in peers if p == level) > 1
        rows.append(
            {
                "level": level,
                "collapsed": len(sub),
                "has_cross_level_partner": cross,
                "has_same_level_partner": same,
                "collapsed_solely_by_cross_level": len(sub) - same,
            }
        )
    return pd.DataFrame(rows)


def within_level_collapse(runs: pd.DataFrame) -> pd.DataFrame:
    """Collapse recomputed inside each level alone, so cross-level duplication cannot contribute."""
    rows = []
    for level in LEVELS:
        sub = runs[runs.level == level]
        sizes = sub.groupby(GROUP_KEY).size()
        groups = sizes[sizes >= 2]
        rows.append(
            {
                "level": level,
                "groups": len(groups),
                "runs_collapsed": int(groups.sum()),
                "runs_total": len(sub),
                "pct": groups.sum() / len(sub) * 100,
            }
        )
    return pd.DataFrame(rows)


def drop_level_counterfactual(runs: pd.DataFrame) -> pd.DataFrame:
    """Overall collapse rate if one chunking level were removed from the grid."""
    rows = []
    for keep, label in [
        (LEVELS, "all three (baseline)"),
        ([1, 2], "drop L3"),
        ([1, 3], "drop L2"),
        ([2, 3], "drop L1"),
    ]:
        sub = runs[runs.level.isin(keep)]
        sizes = sub.groupby(GROUP_KEY).size()
        groups = sizes[sizes >= 2]
        rows.append(
            {
                "grid": label,
                "runs_collapsed": int(groups.sum()),
                "runs_total": len(sub),
                "pct": groups.sum() / len(sub) * 100,
            }
        )
    return pd.DataFrame(rows)


def _jaccard(a: str, b: str) -> float:
    left, right = set(a.split("|")), set(b.split("|"))
    return len(left & right) / len(left | right) if left | right else 1.0


def cross_level_agreement(runs: pd.DataFrame) -> pd.DataFrame:
    """Hold (query, mode, enhancement, alpha) fixed; compare levels pairwise."""
    cell = ["finder_id", "mode", "enhanced", "section_alpha"]
    wide = runs.pivot_table(index=cell, columns="level", values="ids", aggfunc="first")
    rows = []
    for a, b in combinations(LEVELS, 2):
        if a not in wide or b not in wide:
            continue
        matched = wide[[a, b]].dropna()
        exact = int((matched[a] == matched[b]).sum())
        jaccard = matched.apply(lambda r: _jaccard(r[a], r[b]), axis=1).mean()
        rows.append(
            {
                "pair": f"L{a} vs L{b}",
                "exact_same_ranking": exact,
                "cells": len(matched),
                "pct_exact": exact / len(matched) * 100,
                "mean_jaccard": jaccard,
            }
        )
    return pd.DataFrame(rows)


def alpha_suppresses_mode(runs: pd.DataFrame) -> pd.DataFrame:
    """Within a fixed (query, level, enhancement) cell, do all three modes agree?"""
    cell = runs.groupby(["finder_id", "level", "enhanced", "section_alpha"])["ids"].nunique()
    rows = []
    for alpha in sorted(runs["section_alpha"].dropna().unique()):
        sub = cell.xs(alpha, level="section_alpha")
        rows.append(
            {
                "section_alpha": alpha,
                "cells_where_all_modes_identical": int((sub == 1).sum()),
                "cells": len(sub),
                "pct": (sub == 1).mean() * 100,
            }
        )
    return pd.DataFrame(rows)


def enhancement_inert(runs: pd.DataFrame) -> dict:
    cell = ["finder_id", "level", "mode", "section_alpha"]
    wide = runs.pivot_table(index=cell, columns="enhanced", values="ids", aggfunc="first").dropna()
    identical = int((wide["YES"] == wide["NO"]).sum())
    return {"identical": identical, "pairs": len(wide), "pct": identical / len(wide) * 100}


def category_table(runs: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for cat, sub in runs.groupby("category"):
        collapsed = sub[sub.gsize >= 2]
        rows.append(
            {
                "category": cat,
                "queries": sub["finder_id"].nunique(),
                "groups": collapsed["key"].nunique(),
                "pct_runs_collapsed": len(collapsed) / len(sub) * 100,
                "mean_distinct_contexts": sub.groupby("finder_id")["ids"].nunique().mean(),
            }
        )
    return pd.DataFrame(rows).sort_values("pct_runs_collapsed", ascending=False)


def cross_query_contexts(runs: pd.DataFrame) -> pd.DataFrame:
    """Identical contexts served to DIFFERENT queries -- the routing signal failing to discriminate."""
    agg = runs.groupby("ids").agg(
        queries=("finder_id", "nunique"),
        categories=("category", "nunique"),
        runs=("ids", "size"),
    )
    shared = agg[agg.queries > 1].sort_values("runs", ascending=False)
    return shared


def main() -> None:
    workbook = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_WORKBOOK
    runs = load_runs(workbook)

    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 30)

    print(f"workbook: {workbook.name}\n")

    print("== HEADLINE ==")
    for key, value in headline(runs).items():
        print(f"  {key}: {value:.1f}" if isinstance(value, float) else f"  {key}: {value}")

    print("\n== COLLAPSE BY CONFIGURATION AXIS ==")
    print(axis_table(runs).to_string(index=False, float_format="%.1f"))

    print("\n== OVERLAP GROUP COMPOSITION BY CHUNKING LEVEL ==")
    print(level_composition(runs).to_string())

    print("\n== CROSS-LEVEL vs WITHIN-LEVEL ATTRIBUTION ==")
    print(level_attribution(runs).to_string(index=False))

    print("\n== COLLAPSE WITHIN A SINGLE LEVEL (cross-level duplication excluded) ==")
    print(within_level_collapse(runs).to_string(index=False, float_format="%.1f"))

    print("\n== COUNTERFACTUAL: DROP ONE LEVEL ==")
    print(drop_level_counterfactual(runs).to_string(index=False, float_format="%.1f"))

    print("\n== PAIRWISE CROSS-LEVEL AGREEMENT (query, mode, enh, alpha held fixed) ==")
    print(cross_level_agreement(runs).to_string(index=False, float_format="%.2f"))

    print("\n== DOES alpha=1 SUPPRESS THE RETRIEVAL-MODE AXIS? ==")
    print(alpha_suppresses_mode(runs).to_string(index=False, float_format="%.1f"))

    print("\n== IS QUERY ENHANCEMENT INERT? ==")
    stats = enhancement_inert(runs)
    print(f"  identical context in {stats['identical']}/{stats['pairs']} matched pairs "
          f"({stats['pct']:.1f}%)")

    print("\n== COLLAPSE BY QUERY CATEGORY ==")
    print(category_table(runs).to_string(index=False, float_format="%.1f"))

    print("\n== CONTEXTS SHARED ACROSS DIFFERENT QUERIES ==")
    shared = cross_query_contexts(runs)
    print(f"  {len(shared)} contexts serve >1 query, absorbing {int(shared.runs.sum())} runs")
    bridging = shared[shared.categories > 1]
    print(f"  {len(bridging)} of those bridge >1 category, absorbing {int(bridging.runs.sum())} runs")


if __name__ == "__main__":
    main()
