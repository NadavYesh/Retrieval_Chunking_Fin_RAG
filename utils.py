"""
Shared utilities used across the pipeline.
"""

import json
import re
from pathlib import Path

import pandas as pd

# ── FinDER temporal anchor ────────────────────────────────────────────────────
# The dataset was published in early 2024; treat "current"/"latest" as FY2024.
FINDER_ANCHOR_YEAR = 2024

# Number of years in the default fallback window (anchor + N-1 previous years)
# used when a query gives no year signal at all. See _default_year_window().
DEFAULT_YEAR_WINDOW_SIZE = 5


def _default_year_window(anchor: int = FINDER_ANCHOR_YEAR) -> list[int]:
    """Anchor year plus the 4 preceding years, oldest first: [2020..2024]."""
    return list(range(anchor - (DEFAULT_YEAR_WINDOW_SIZE - 1), anchor + 1))


def year_weights(years: list[int]) -> list[tuple[int, float]]:
    """
    Assign retrieval budget weights to years using exponential decay by recency.
    Most recent year gets the largest share of the prefetch budget.

    Examples:
      [2024]             -> [(2024, 1.00)]
      [2023, 2024]       -> [(2024, 0.67), (2023, 0.33)]
      [2020..2024]       -> [(2024, 0.52), (2023, 0.26), (2022, 0.13), (2021, 0.06), (2020, 0.03)]
    """
    if not years:
        return []
    sorted_years = sorted(years, reverse=True)  # newest first
    n = len(sorted_years)
    raw = [2 ** (n - 1 - i) for i in range(n)]
    total = sum(raw)
    return [(yr, r / total) for yr, r in zip(sorted_years, raw)]


# ── Deterministic fiscal-year extraction ──────────────────────────────────────
# Two kinds of signal, both read directly off the raw query text (no LLM):
#   1. Explicit numeric years/ranges ("FY24", "2022-2024", "2022/21") -- these
#      are unambiguous, so a query that states them is respected as-is with a
#      flat, unweighted filter across exactly those years.
#   2. Qualified relative phrases ("current fiscal year", "last year",
#      "year-over-year") -- specific multi-word phrases that essentially never
#      appear with a non-temporal meaning in financial text.
# Deliberately NOT included: bare single-word anchors ("current", "recent",
# "latest", "historical"). Those collide with standard accounting terminology
# that has nothing to do with fiscal year -- "current ratio", "current
# liabilities", "historical cost accounting" -- which is a word-sense-
# disambiguation problem no keyword match can solve safely. A query with only
# one of those bare words and nothing else falls through to None (which then
# triggers the default 5-year decayed window at retrieval time), rather than
# risking a wrong guess.
FINDER_PRIOR_YEAR = FINDER_ANCHOR_YEAR - 1

_RELATIVE_YEAR_MAP = {
    "current fiscal year":   FINDER_ANCHOR_YEAR,
    "current fy":            FINDER_ANCHOR_YEAR,
    "current year":          FINDER_ANCHOR_YEAR,
    "this year":             FINDER_ANCHOR_YEAR,
    "most recent":           FINDER_ANCHOR_YEAR,
    "previous fiscal year":  FINDER_PRIOR_YEAR,
    "last fiscal year":      FINDER_PRIOR_YEAR,
    "prior fiscal year":     FINDER_PRIOR_YEAR,
    "previous year":         FINDER_PRIOR_YEAR,
    "prior year":            FINDER_PRIOR_YEAR,
    "last year":             FINDER_PRIOR_YEAR,
    "year-over-year":        [FINDER_PRIOR_YEAR, FINDER_ANCHOR_YEAR],
    "yoy":                   [FINDER_PRIOR_YEAR, FINDER_ANCHOR_YEAR],
    "prior year trends":     [FINDER_PRIOR_YEAR, FINDER_ANCHOR_YEAR],
    "prior year comparison": [FINDER_PRIOR_YEAR, FINDER_ANCHOR_YEAR],
    "compared to prior":     [FINDER_PRIOR_YEAR, FINDER_ANCHOR_YEAR],
    "prior fiscal years":    [FINDER_ANCHOR_YEAR - 2, FINDER_PRIOR_YEAR, FINDER_ANCHOR_YEAR],
    "prior years":           [FINDER_ANCHOR_YEAR - 2, FINDER_PRIOR_YEAR, FINDER_ANCHOR_YEAR],
    "multi-year":            [FINDER_ANCHOR_YEAR - 2, FINDER_PRIOR_YEAR, FINDER_ANCHOR_YEAR],
}
_PHRASES_BY_LENGTH = sorted(_RELATIVE_YEAR_MAP, key=len, reverse=True)


def _fy_to_year(n: int) -> int:
    return (2000 + n) if n < 100 else n


# "FY22-FY24", "FY22 to FY24", "FY22–FY24"
_FY_RANGE_RE = re.compile(r'\bFY(\d{2,4})\s*(?:-|to|–|—)\s*FY(\d{2,4})\b', re.IGNORECASE)
# "2022-2024", "2022 to 2024", "2022–2024"
_YEAR_RANGE_RE = re.compile(r'\b(20[12][0-9])\s*(?:-|to|–|—)\s*(20[12][0-9])\b')
# "2022/21" -- shorthand for two specific years, e.g. "2023 compared to
# 2022/21" meaning 2022 and 2021. Not treated as a range fill (the two years
# are stated explicitly, not endpoints of a span to expand).
_YEAR_SLASH_RE = re.compile(r'\b(20[12][0-9])/(\d{2})\b')
_FY_SINGLE_RE = re.compile(r'\bFY(\d{2,4})\b', re.IGNORECASE)
_YEAR_SINGLE_RE = re.compile(r'\b(20[12][0-9])\b')


def _explicit_years_from_query(query: str) -> list[int]:
    """Numeric FY/calendar-year mentions, including range expansion."""
    years: set[int] = set()

    for m in _YEAR_SLASH_RE.finditer(query):
        years.add(int(m.group(1)))
        years.add(_fy_to_year(int(m.group(2))))

    for m in _FY_RANGE_RE.finditer(query):
        lo, hi = _fy_to_year(int(m.group(1))), _fy_to_year(int(m.group(2)))
        if lo <= hi:
            years.update(range(lo, hi + 1))
    for m in _YEAR_RANGE_RE.finditer(query):
        lo, hi = int(m.group(1)), int(m.group(2))
        if lo <= hi:
            years.update(range(lo, hi + 1))

    # Single mentions -- skip any span already consumed by a range/slash match
    # above so "FY22-FY24" doesn't ALSO register a bare "FY22"/"FY24" (it's
    # already fully expanded); harmless since it's a set, kept for clarity.
    consumed_spans = [m.span() for m in _FY_RANGE_RE.finditer(query)] + \
                      [m.span() for m in _YEAR_RANGE_RE.finditer(query)] + \
                      [m.span() for m in _YEAR_SLASH_RE.finditer(query)]

    def _in_consumed(span):
        return any(s <= span[0] and span[1] <= e for s, e in consumed_spans)

    for m in _FY_SINGLE_RE.finditer(query):
        if not _in_consumed(m.span()):
            years.add(_fy_to_year(int(m.group(1))))
    for m in _YEAR_SINGLE_RE.finditer(query):
        if not _in_consumed(m.span()):
            years.add(int(m.group(1)))

    return sorted(years)


def _relative_years_from_query(query: str):
    """First (longest) qualified relative-phrase match found in the query text."""
    lower = query.lower()
    for phrase in _PHRASES_BY_LENGTH:
        if re.search(r'\b' + re.escape(phrase) + r'\b', lower):
            return _RELATIVE_YEAR_MAP[phrase]
    return None


def extract_year_deterministic(query: str):
    """
    Deterministic fiscal-year resolver, no LLM involved.

    Precedence:
      1. Explicit numeric years/ranges in the text ("FY24", "2022-2024",
         "2022/21") -- unambiguous, so they win outright.
      2. Qualified relative-phrase anchors ("latest fiscal year", "last
         year", "year-over-year") resolved against the FinDER temporal
         anchor (FY2024 current / FY2023 prior).
      3. None if neither is present. Callers should NOT treat None as "no
         year filter" -- at retrieval time, None means the query gave no
         signal, which triggers the default 5-year decayed window
         (_default_year_window + year_weights) rather than an unfiltered
         search across all years.

    Returns an int (single year), list[int] (multiple explicit years, to
    be applied as a flat/undecayed filter -- we trust what the user asked
    for over a recency prior), or None.
    """
    explicit = _explicit_years_from_query(query)
    if explicit:
        return explicit[0] if len(explicit) == 1 else explicit

    relative = _relative_years_from_query(query)
    if relative is not None:
        return relative

    return None


# ── Deterministic ticker extraction ───────────────────────────────────────────
# Backed by a static snapshot of the S&P 500 constituent list (Symbol /
# Shortname / Longname), fetched once via:
#   import kagglehub; kagglehub.dataset_download("andrewmvd/sp-500-stocks")
# and saved to SP500_CSV_PATH so this module has no runtime network/kagglehub
# dependency.
SP500_CSV_PATH = "/Users/nadavsmacbookair/Desktop/Thesis/data/reference/sp500_companies.csv"

_LEGAL_SUFFIXES = [
    "incorporated", "corporation", "holdings inc", "holding company",
    "holdings", "holding", "company", "companies", "corp", "inc",
    "group", "plc", "ltd", "limited", "llc", "l l c", "co", "the",
    "class a", "class b", "class c", "new",
]
_SUFFIX_RE = re.compile(
    r"\b(" + "|".join(re.escape(s) for s in _LEGAL_SUFFIXES) + r")\b"
)
_PAREN_RE = re.compile(r"\([^)]*\)")


def _normalize_company_name(name: str) -> str:
    """Strip legal boilerplate/punctuation down to a bare, comparable name."""
    n = name.lower()
    n = _PAREN_RE.sub(" ", n)          # "Walt Disney Company (The)" -> drop "(The)"
    n = n.replace("&", " and ")
    n = re.sub(r"[’']s\b", "", n)      # "NVIDIA's" -> "nvidia" (drop possessive, not merge)
    n = re.sub(r"[.,’']", "", n)       # "PayPal Holdings, Inc." -> "paypal holdings inc"
    n = re.sub(r"[^a-z0-9 ]", " ", n)
    n = _SUFFIX_RE.sub(" ", n)
    n = re.sub(r"\s+", " ", n).strip()
    return n


# Hand-curated aliases for colloquial/brand names that diverge from the legal
# entity name. Deliberately curated by hand rather than auto-derived: an
# earlier prototype auto-derived single "distinctive" words out of company
# names (e.g. "Ford" from "Ford Motor") and it pulled in plain English words
# that only happen to be unique *within the S&P 500 name list* -- e.g. "real"
# (Alexandria Real Estate -> ARE), "state" (State Street -> STT), "water"
# (American Water Works -> AWK), "union" (Union Pacific -> UNP), "take"
# (Take-Two -> TTWO), "price" (T. Rowe Price -> TROW). Those are exactly the
# kind of generic terms that show up constantly in 10-K financial text ("real
# estate", "state tax", "union risk", "take rate", "price of raw materials"),
# so auto-derivation would inject false-positive tickers more often than the
# LLM hallucinated them. Hand-curating avoids that.
_MANUAL_TICKER_ALIASES = {
    "google": "GOOGL",
    "alphabet": "GOOGL",
    "facebook": "META",
    "meta": "META",
    "amazon": "AMZN",
    "ford": "F",
    "jpmorgan": "JPM",
    "jp morgan": "JPM",
    "berkshire": "BRK-B",
    "berkshire hathaway": "BRK-B",
    "att": "T",
    "at t": "T",
}

_NOT_TICKERS = {"FOR", "THE", "AND", "INC", "LLC", "SEC", "FY", "US", "AT", "EPS", "GAAP"}

_ticker_alias_to_symbol: dict[str, str] | None = None
_ticker_alias_patterns: list[tuple[str, re.Pattern]] | None = None
_known_tickers: set[str] | None = None


def _load_ticker_tables():
    """Lazily build the alias/pattern tables from the SP500 CSV snapshot, once."""
    global _ticker_alias_to_symbol, _ticker_alias_patterns, _known_tickers
    if _ticker_alias_to_symbol is not None:
        return

    if not Path(SP500_CSV_PATH).exists():
        raise FileNotFoundError(
            f"SP500 reference data not found at {SP500_CSV_PATH}. "
            "Fetch it once via kagglehub.dataset_download('andrewmvd/sp-500-stocks') "
            "and copy sp500_companies.csv there."
        )
    df = pd.read_csv(SP500_CSV_PATH)

    alias_to_symbol: dict[str, str] = {}
    for _, row in df.iterrows():
        ticker = row["Symbol"]
        for raw_name in (row["Shortname"], row["Longname"]):
            alias = _normalize_company_name(str(raw_name))
            if alias and alias not in alias_to_symbol:
                alias_to_symbol[alias] = ticker  # first occurrence = higher index weight

    for alias, ticker in _MANUAL_TICKER_ALIASES.items():
        alias_to_symbol[alias] = ticker  # manual aliases win over auto-derived

    _ticker_alias_to_symbol = alias_to_symbol
    _known_tickers = set(df["Symbol"].str.upper())
    # Longest-first so "berkshire hathaway" matches before "berkshire"
    _ticker_alias_patterns = [
        (alias, re.compile(r"\b" + re.escape(alias) + r"\b"))
        for alias in sorted(alias_to_symbol, key=len, reverse=True)
    ]


def extract_ticker_hint(query: str):
    """
    Regex pre-extraction of ticker symbol from common FinDER query patterns.
    Checks parenthetical '(TICKER)' and trailing uppercase word. Used as a
    last-resort fallback inside extract_ticker_deterministic for tickers
    outside the S&P 500 alias table. Returns lowercase ticker string or None.
    """
    m = re.search(r'\(([A-Z]{1,5})\)', query)
    if m:
        return m.group(1).lower()
    m = re.search(r'\b([A-Z]{2,5})\s*[.,]?\s*$', query.rstrip())
    if m:
        candidate = m.group(1)
        if candidate not in _NOT_TICKERS:
            return candidate.lower()
    return None


def extract_ticker_deterministic(query: str):
    """
    Deterministic ticker resolver, no LLM involved.

    Precedence:
      1. Explicit ticker already in the text -- "(NVDA)" or a bare known
         symbol token (2+ chars only: single letters like "A"/"V"/"F" are
         too easily confused with ordinary capitalized words, roman
         numerals, or list markers -- e.g. "Section V" or "Item A").
      2. Company-name match against the S&P 500 alias table (full
         Shortname/Longname + hand-curated aliases), longest alias wins so
         "Berkshire Hathaway" beats "Berkshire".
      3. Fallback to the trailing-uppercase-word heuristic
         (extract_ticker_hint) for tickers outside the S&P 500.
    """
    _load_ticker_tables()

    m = re.search(r'\(([A-Z]{1,5})\)', query)
    if m and m.group(1) in _known_tickers:
        return m.group(1).lower()

    for tok in re.findall(r'\b[A-Z]{2,5}(?:-[A-Z])?\b', query):
        if tok in _known_tickers and tok not in _NOT_TICKERS:
            return tok.lower()

    norm_query = _normalize_company_name(query)
    for alias, pattern in _ticker_alias_patterns:
        if pattern.search(norm_query):
            return _ticker_alias_to_symbol[alias].lower()

    return extract_ticker_hint(query)


# ── LLM response parsing ──────────────────────────────────────────────────────
METADATA_FALLBACK = {"optimized_query": None}


def parse_metadata_response(response: str, fallback_query: str) -> dict:
    """
    Extract the optimized-query rewrite from an LLM response that should
    contain a JSON object with an "optimized_prompt" field. ticker/year/
    form_type are no longer sourced from the LLM (see
    extract_ticker_deterministic / extract_year_deterministic in this module
    and the hardcoded form_type in evaluation/rag_functions.py) -- this
    function's only remaining job is the query rewrite.

    Falls back to the original query on any parse failure.
    """
    try:
        match = re.search(r'\{.*\}', response, re.DOTALL)
        if not match:
            return {"optimized_query": fallback_query}
        data = json.loads(match.group())
        return {"optimized_query": data.get("optimized_prompt", fallback_query)}
    except Exception as e:
        print(f"  [parse_metadata] WARNING: {e}; using original query")
        return {"optimized_query": fallback_query}
