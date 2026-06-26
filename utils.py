"""
Shared utilities used across the pipeline.
"""

import json
import re

# ── FinDER temporal anchor ────────────────────────────────────────────────────
# The dataset was published in early 2024; treat "current"/"latest" as FY2024.
FINDER_ANCHOR_YEAR = 2024
FINDER_PRIOR_YEAR  = 2023

_RELATIVE_YEAR_MAP = {
    # ── Single-year anchors ────────────────────────────────────────────────────
    "current fiscal year":   FINDER_ANCHOR_YEAR,
    "current fy":            FINDER_ANCHOR_YEAR,
    "current year":          FINDER_ANCHOR_YEAR,
    "this year":             FINDER_ANCHOR_YEAR,
    "latest":                FINDER_ANCHOR_YEAR,
    "recent":                FINDER_ANCHOR_YEAR,
    "most recent":           FINDER_ANCHOR_YEAR,
    "current":               FINDER_ANCHOR_YEAR,
    "prior year":            FINDER_PRIOR_YEAR,
    "previous year":         FINDER_PRIOR_YEAR,
    "last year":             FINDER_PRIOR_YEAR,
    "previous fiscal year":  FINDER_PRIOR_YEAR,
    "last fiscal year":      FINDER_PRIOR_YEAR,
    "prior fiscal year":     FINDER_PRIOR_YEAR,
    # ── Multi-year phrases → list (2 years: anchor-1 + anchor) ────────────────
    "year-over-year":        [FINDER_PRIOR_YEAR, FINDER_ANCHOR_YEAR],
    "yoy":                   [FINDER_PRIOR_YEAR, FINDER_ANCHOR_YEAR],
    "prior year trends":     [FINDER_PRIOR_YEAR, FINDER_ANCHOR_YEAR],
    "prior year comparison": [FINDER_PRIOR_YEAR, FINDER_ANCHOR_YEAR],
    "compared to prior":     [FINDER_PRIOR_YEAR, FINDER_ANCHOR_YEAR],
    # ── Multi-year phrases → list (3 years: anchor-2..anchor) ─────────────────
    "prior fiscal years":    [FINDER_ANCHOR_YEAR - 2, FINDER_PRIOR_YEAR, FINDER_ANCHOR_YEAR],
    "prior years":           [FINDER_ANCHOR_YEAR - 2, FINDER_PRIOR_YEAR, FINDER_ANCHOR_YEAR],
    "historical":            [FINDER_ANCHOR_YEAR - 2, FINDER_PRIOR_YEAR, FINDER_ANCHOR_YEAR],
    "multi-year":            [FINDER_ANCHOR_YEAR - 2, FINDER_PRIOR_YEAR, FINDER_ANCHOR_YEAR],
}


def _resolve_fy_from_query(query: str) -> list[int]:
    """Extract and resolve FY-style and calendar-year references from raw query text."""
    years: set[int] = set()
    for m in re.findall(r'\bFY(\d{2,4})\b', query, re.IGNORECASE):
        n = int(m)
        yr = (2000 + n) if n < 100 else n
        years.add(yr)
    for yr_str in re.findall(r'\b(20[12][0-9])\b', query):
        years.add(int(yr_str))
    return sorted(years)


def sanitize_year_extraction(year_val, query: str = ""):
    """
    Converts LLM-extracted year values to a clean int or list[int].
    Handles: int, list, relative-time strings, FY-shorthand strings.
    Returns None when no valid year can be resolved.
    Single year → int.  Multiple years → list[int].
    """
    if year_val is None:
        if query:
            extracted = _resolve_fy_from_query(query)
            if extracted:
                return extracted[0] if len(extracted) == 1 else extracted
        return None

    if isinstance(year_val, int):
        if year_val < 100:
            year_val = 2000 + year_val
        return year_val if 1990 <= year_val <= 2100 else None

    if isinstance(year_val, list):
        out: list[int] = []
        for v in year_val:
            resolved = sanitize_year_extraction(v, query=query)
            if resolved is None:
                continue
            if isinstance(resolved, list):
                out.extend(resolved)
            else:
                out.append(resolved)
        unique = sorted(set(out))
        if not unique:
            return None
        return unique[0] if len(unique) == 1 else unique

    if isinstance(year_val, str):
        lower = year_val.lower().strip()
        for phrase, yr in _RELATIVE_YEAR_MAP.items():
            if phrase in lower:
                return yr
        fy_match = re.search(r'\bfy(\d{2,4})\b', lower)
        if fy_match:
            n = int(fy_match.group(1))
            return (2000 + n) if n < 100 else n
        try:
            n = int(lower[:4])
            if 1990 <= n <= 2100:
                return n
        except ValueError:
            pass
        if query:
            extracted = _resolve_fy_from_query(query)
            if extracted:
                return extracted[0] if len(extracted) == 1 else extracted
        return None

    try:
        n = int(year_val)
        if n < 100:
            n = 2000 + n
        return n if 1990 <= n <= 2100 else None
    except (TypeError, ValueError):
        return None


def extract_ticker_hint(query: str):
    """
    Regex pre-extraction of ticker symbol from common FinDER query patterns.
    Checks parenthetical '(TICKER)' and trailing uppercase word.
    Returns lowercase ticker string or None.
    """
    m = re.search(r'\(([A-Z]{1,5})\)', query)
    if m:
        return m.group(1).lower()
    # Trailing 2-5 char uppercase word (possibly followed by punctuation/whitespace)
    m = re.search(r'\b([A-Z]{2,5})\s*[.,]?\s*$', query.rstrip())
    if m:
        candidate = m.group(1)
        # Exclude common English words that are not tickers
        _NOT_TICKERS = {"FOR", "THE", "AND", "INC", "LLC", "SEC", "FY", "US", "AT"}
        if candidate not in _NOT_TICKERS:
            return candidate.lower()
    return None


METADATA_FALLBACK = {"optimized_query": None, "ticker": None, "year": None, "form_type": None}


def parse_metadata_response(response: str, fallback_query: str) -> dict:
    """
    Extract structured metadata from an LLM response that should contain a JSON object.

    Returns a dict with keys: optimized_query, ticker, year, form_type.
    Falls back gracefully on any parse failure.
    """
    try:
        match = re.search(r'\{.*\}', response.lower(), re.DOTALL)
        if not match:
            return {**METADATA_FALLBACK, "optimized_query": fallback_query}
        data = json.loads(match.group())
        return {
            "optimized_query": data.pop("optimized_prompt", fallback_query),
            "ticker":          data.get("ticker"),
            "year":            data.get("year"),
            "form_type":       data.get("form_type"),
        }
    except Exception as e:
        print(f"  [parse_metadata] WARNING: {e}; using original query")
        return {**METADATA_FALLBACK, "optimized_query": fallback_query}
