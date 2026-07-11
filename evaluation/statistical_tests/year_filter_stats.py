"""
Distribution of retrieval-stage ticker and year filters across the eval workbook.

Reads the `detail` sheet, where each row is one (query x retrieval-config) trial.
Both filters applied at retrieval are encoded in the first two underscore-separated
fields of `run_id`, which run_rag_lazy builds as f"{ticker}_{year}_{level}_{mode}_{q}".
Each field is either the sentinel None, a scalar, or a stringified Python list:

    tsla_None_3_dense_5                 -> ticker tsla,        no year filter
    tsla_2023_2_sparse_6                -> ticker tsla,        single year
    aapl_[2023, 2024]_3_hybrid_0        -> ticker aapl,        multiple years
    ['tsla', 'pypl']_2024_1_dense_2     -> multiple tickers,   single year

The list form carries commas and quotes but never underscores, so splitting the
run_id on "_" stays unambiguous and one helper parses both fields.

The sheet also has a `ticker_filter` column, but it collapses any multi-ticker run
to the literal "multi" (run_rag_lazy.py:180), so run_id is the only place the
individual tickers survive. The two are cross-checked at load.

Two units are reported, because they answer different questions:

  runs   unique run_id values. This is the design of the experiment: how the
         configurations were spread across filter regimes.
  trials individual rows. This is the composition of any trial-level statistic
         computed off this sheet, and it differs from the run share whenever
         runs carry unequal query counts.

Usage:
    python year_filter_stats.py [path/to/workbook.xlsx]
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

DEFAULT_WORKBOOK = (
    Path(__file__).resolve().parents[3]
    / "data/eval_results/ready_for_analysis/analysis"
    / "tsla_pypl-corr_nvda_aapl_no_out-of-corpus-pollution.xlsx"
)

TICKER_FIELD, YEAR_FIELD = 0, 1

# Regime label per axis. "unfiltered" means the retriever saw the whole corpus on
# that axis; the singular/plural split is what the counts below are about.
REGIMES = {
    "ticker": ["unfiltered", "single ticker", "multiple tickers"],
    "year": ["unfiltered", "single year", "multiple years"],
}


def parse_field(run_id: str, field: int) -> list[str]:
    """
    Filter values from one run_id field: [] when unfiltered, else one entry per value.

    Handles the scalar form (tsla, 2024) and the stringified-list form
    (['tsla', 'pypl'], [2023, 2024]) that multi-value filters produce.
    """
    raw = run_id.split("_")[field].strip()
    if raw in {"None", "?", ""}:
        return []
    return [v.strip(" '\"") for v in raw.strip("[]").split(",")]


def classify(values: list[str], axis: str) -> str:
    none_lbl, one_lbl, many_lbl = REGIMES[axis]
    if not values:
        return none_lbl
    return one_lbl if len(values) == 1 else many_lbl


def load_runs(workbook: Path) -> pd.DataFrame:
    df = pd.read_excel(workbook, sheet_name="detail")

    for axis, field in (("ticker", TICKER_FIELD), ("year", YEAR_FIELD)):
        vals = df["run_id"].map(lambda r, f=field: parse_field(r, f))
        df[f"{axis}s"] = vals
        df[f"n_{axis}s"] = vals.str.len()
        df[f"{axis}_regime"] = vals.map(lambda v, a=axis: classify(v, a))
        df[f"{axis}_set"] = vals.map(lambda v: "none" if not v else ", ".join(v))

    df["years"] = df["years"].map(lambda ys: [int(y) for y in ys])

    # ticker_filter writes "multi" for multi-ticker runs; single-ticker runs must agree.
    single = df["n_tickers"] == 1
    mismatch = single & (df["ticker_set"] != df["ticker_filter"])
    if mismatch.any():
        bad = df.loc[mismatch, "run_id"].unique()[:3]
        raise ValueError(f"run_id ticker disagrees with ticker_filter, e.g. {list(bad)}")

    return df


def share_table(df: pd.DataFrame, axis: str) -> pd.DataFrame:
    """Regime shares for one axis, by run and by trial, as counts and percentages."""
    runs = df.drop_duplicates("run_id")
    col = f"{axis}_regime"
    table = pd.DataFrame(
        {"runs": runs[col].value_counts(), "trials": df[col].value_counts()}
    ).reindex(REGIMES[axis]).fillna(0).astype(int)
    table["runs_pct"] = 100 * table["runs"] / table["runs"].sum()
    table["trials_pct"] = 100 * table["trials"] / table["trials"].sum()
    return table


def regime_crosstab(df: pd.DataFrame) -> pd.DataFrame:
    """Runs per (ticker regime x year regime) cell: how the two filters co-occur."""
    runs = df.drop_duplicates("run_id")
    return (
        pd.crosstab(runs["ticker_regime"], runs["year_regime"], margins=True, margins_name="total")
        .reindex(index=REGIMES["ticker"] + ["total"], columns=REGIMES["year"] + ["total"])
        .fillna(0)
        .astype(int)
    )


def per_ticker(df: pd.DataFrame) -> pd.DataFrame:
    """
    Year-filter regimes for each individual ticker.

    Runs are exploded over their tickers, so a multi-ticker run is counted once
    under each ticker it filtered on. Column totals therefore exceed the run count
    whenever multi-ticker runs exist; the point is each ticker's own exposure.
    """
    runs = df.drop_duplicates("run_id").explode("tickers")
    table = (
        runs.pivot_table(index="tickers", columns="year_regime", values="run_id", aggfunc="count")
        .reindex(columns=REGIMES["year"])
        .fillna(0)
        .astype(int)
    )
    table["total"] = table.sum(axis=1)
    table["year_filtered_pct"] = 100 * (table["total"] - table["unfiltered"]) / table["total"]
    return table


def by_set(df: pd.DataFrame, axis: str) -> pd.DataFrame:
    """Run and trial counts for each exact filter value set on one axis."""
    runs = df.drop_duplicates("run_id")
    col = f"{axis}_set"
    table = pd.DataFrame({"runs": runs[col].value_counts(), "trials": df[col].value_counts()})
    table["runs_pct"] = 100 * table["runs"] / table["runs"].sum()
    return table.sort_values("runs", ascending=False)


def main() -> None:
    workbook = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_WORKBOOK
    df = load_runs(workbook)

    n_runs, n_trials = df["run_id"].nunique(), len(df)
    print(f"{workbook.name}\n{n_runs} runs, {n_trials} trials\n")

    for axis in ("ticker", "year"):
        print(f"{axis.capitalize()}-filter regime")
        print(share_table(df, axis).round(1).to_string(), "\n")

    print("Runs by ticker regime x year regime")
    print(regime_crosstab(df).to_string(), "\n")

    print("Runs by individual ticker")
    print(per_ticker(df).round(1).to_string(), "\n")

    for axis in ("ticker", "year"):
        print(f"Runs by exact {axis} set")
        print(by_set(df, axis).round(1).to_string(), "\n")

    runs = df.drop_duplicates("run_id")
    for axis in ("ticker", "year"):
        filtered = runs[runs[f"n_{axis}s"] > 0]
        if not filtered.empty:
            mean = filtered[f"n_{axis}s"].mean()
            print(f"Mean {axis}s per {axis}-filtered run: {mean:.2f}")


if __name__ == "__main__":
    main()
