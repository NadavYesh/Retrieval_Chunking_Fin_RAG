"""
Category-to-section routing for 10-K retrieval.

Two retrieval tracks:
  Track A  —  Qdrant MatchText pre-filter restricted to the expected Items
               for the query category (via make_section_filter).
  Track B  —  Existing unfiltered retrieval.

Fusion is controlled by section_alpha:
  alpha = 0.0  →  Track B only (no section routing; Track A is skipped entirely)
  alpha = 1.0  →  Track A only (hard routing)
  0 < alpha < 1 →  weighted RRF blend (soft routing)

`subsection` is TEXT-indexed in Qdrant (WORD tokenizer, lowercase=True),
so MatchText is both supported and efficient.

Regex handles OCR artifacts ('item 1a ri sk factors') by only needing
to match 'item' and the item-number token (e.g. '1a'), which survive
OCR corruption intact.
"""

import re
from qdrant_client import models as qdrant_models

_ITEM_RE = re.compile(r'\bitem\s+(\d+[a-z]*)', re.IGNORECASE)
_RRF_K   = 60  # standard RRF constant


# ── category → target item numbers ──────────────────────────────────────────
CATEGORY_ITEMS: dict[str, frozenset[str]] = {
    "company overview":   frozenset(["1"]),
    "financials":         frozenset(["7", "7a", "8"]),
    "footnotes":          frozenset(["8"]),
    "governance":         frozenset(["10", "11", "12", "13", "14"]),
    "accounting":         frozenset(["8"]),
    "shareholder return": frozenset(["5"]),
    "risk":               frozenset(["1a"]),
    "legal":              frozenset(["3"]),
}


# ── core helpers ─────────────────────────────────────────────────────────────

def extract_item_nums(subsection: str) -> list[str]:
    """Extract all item numbers from a (possibly noisy) subsection string."""
    return [m.group(1).lower() for m in _ITEM_RE.finditer(subsection or "")]


# ── Qdrant filter factory ────────────────────────────────────────────────────

def make_section_filter(category: str) -> qdrant_models.Filter | None:
    """
    Build a Qdrant Filter (should = OR across target items) for Track A retrieval.

    Returns None when the category is unknown or has no mapping.
    """
    target_items = CATEGORY_ITEMS.get((category or "").lower().strip())
    if not target_items:
        return None

    conds = [
        qdrant_models.FieldCondition(
            key="subsection",
            match=qdrant_models.MatchText(text=f"item {item_num}"),
        )
        for item_num in sorted(target_items)
    ]

    if len(conds) == 1:
        return qdrant_models.Filter(must=conds)
    return qdrant_models.Filter(should=conds)


# ── Weighted RRF fusion ──────────────────────────────────────────────────────

def section_fuse(track_a: list, track_b: list, alpha: float, top_k: int) -> list:
    """
    Weighted RRF fusion of section-filtered (Track A) and unfiltered (Track B) results.

    Parameters
    ----------
    track_a : points from section-filtered retrieval (Track A)
    track_b : points from unfiltered retrieval (Track B)
    alpha   : weight on Track A; (1 - alpha) is applied to Track B
              0.0 → Track B only, 1.0 → Track A only, in-between → soft blend
    top_k   : number of results to return

    Points appearing in both tracks accumulate scores from both.
    Within each track, original retrieval rank is preserved via RRF scoring.
    """
    scores: dict = {}
    id_to_point: dict = {}

    for rank, p in enumerate(track_a):
        scores[p.id] = scores.get(p.id, 0.0) + alpha / (_RRF_K + rank + 1)
        id_to_point[p.id] = p

    for rank, p in enumerate(track_b):
        scores[p.id] = scores.get(p.id, 0.0) + (1.0 - alpha) / (_RRF_K + rank + 1)
        id_to_point[p.id] = p

    ranked = sorted(scores, key=lambda pid: scores[pid], reverse=True)
    return [id_to_point[pid] for pid in ranked[:top_k]]
