"""
Query -> FinDER category classifier, and the routing agent that reasons over it.

Section routing (section_routing.py) maps a query's FinDER category to the 10-K
Items it should be retrieved from. FinDER hands that category over with every
question; a deployed system gets no such label. This module predicts it instead,
so routing can run on the query text alone.

Classifier
  TF-IDF (word 1-2 grams + char_wb 2-5 grams) -> multinomial logistic regression.
  Chosen over a LinearSVC and a frozen-encoder (MiniLM) + LogReg in a 5-fold
  comparison over all of FinDER: the SVM was ~1pt more accurate but needs a Platt
  calibration layer bolted on and came out twice as miscalibrated (ECE 0.077 vs
  0.036), and the frozen encoder was ~4pt less accurate -- FinDER queries are
  ~11-word abbreviation soup ("comp. positioning", "ASR", "FCF"), which char
  n-grams handle and a general sentence encoder does not. The agent below acts
  on the probabilities, so calibration matters more than the last point of
  accuracy. A ticker-grouped CV gave the same accuracy (0.846), so the model is
  not keying on company names.

Routing policy (three bands over the calibrated probabilities)
  confident  p1 >= tau_hard            -> route to the top category's Items.
  ambiguous  p1 + p2 >= tau_pair       -> deterministic: route to the union of
                                          the top-2 categories' Items.
  uncertain  otherwise                 -> deterministic: abstain (alpha 0, i.e.
                                          the unrouted baseline).
  tau_hard / tau_pair are tuned on out-of-fold predictions over the training
  queries (never on the eval questions) to hit TARGET_HARD_PRECISION and
  TARGET_PAIR_COVERAGE, where "correct" means the routed Items cover every Item
  of the true category. Correctness is judged at Item level, not category level:
  footnotes and accounting both route to Item 8, so confusing them costs nothing.

Agent
  For the confident band the classifier's decision stands (no LLM call). In the
  ambiguous and uncertain bands an LLM sees the query, the top candidates with
  their probabilities and the n-grams that drove each, and the 10-K Item each
  candidate routes to, and chooses a subset of the candidates or abstains. It
  cannot route to a category the classifier did not propose. Unparseable or
  invalid output falls back to the deterministic decision for that band.

Two routers come out of this, so the thesis can ablate the LLM's contribution:
  "classifier"  the deterministic three-band policy alone
  "agent"       the same policy with the LLM deciding the non-confident bands
plus the existing "oracle" router, which uses FinDER's given category.

Run directly to train, tune and save the model (see __main__).
"""

import hashlib
import json
import re
import sys
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.parent))

from evaluation.section_routing import CATEGORY_ITEMS

# ── CONFIG ───────────────────────────────────────────────────────────────────
FINDER_PATH = "/Users/nadavsmacbookair/Desktop/Thesis/data/FinDER/train.parquet"
MODEL_PATH  = "/Users/nadavsmacbookair/Desktop/Thesis/data/models/section_classifier.joblib"

LOGREG_C    = 5.0
CV_FOLDS    = 5
CV_SEED     = 0

# Fraction of confident-band queries whose top-1 route must cover the true Items.
TARGET_HARD_PRECISION = 0.95
# Fraction of ambiguous-band queries whose top-2 union must cover the true Items.
TARGET_PAIR_COVERAGE  = 0.90
# A threshold is only accepted if at least this many OOF queries sit above it,
# so a handful of lucky high-confidence predictions cannot set it.
MIN_THRESHOLD_SUPPORT = 100

N_AGENT_CANDIDATES = 3   # categories the agent may choose among
N_EVIDENCE_TERMS   = 4   # n-grams shown per candidate
AGENT_MAX_TOKENS   = 300

ROUTERS = ("oracle", "classifier", "agent")

# Human-readable 10-K Items, shown to the agent next to each candidate category.
ITEM_TITLES = {
    "1":  "Item 1 Business",
    "1a": "Item 1A Risk Factors",
    "3":  "Item 3 Legal Proceedings",
    "5":  "Item 5 Market for Common Equity, Stockholder Matters and Issuer Purchases (buybacks, dividends)",
    "7":  "Item 7 MD&A",
    "7a": "Item 7A Market Risk Disclosures",
    "8":  "Item 8 Financial Statements and Notes",
    "10": "Item 10 Directors, Executive Officers and Corporate Governance",
    "11": "Item 11 Executive Compensation",
    "12": "Item 12 Security Ownership",
    "13": "Item 13 Related Transactions and Director Independence",
    "14": "Item 14 Principal Accountant Fees",
}


def canonical_items(category: str) -> frozenset[str]:
    """Item numbers for a category with OCR-variant spellings ('7 a') collapsed.
    This basically cleans up names. 
    """
    return frozenset(i.replace(" ", "") for i in CATEGORY_ITEMS.get(category, ()))


def _items_title(category: str) -> str:
    items = sorted(canonical_items(category), key=lambda s: 
                   (int(re.match(r"\d+", s).group()), s))
    return "; ".join(ITEM_TITLES.get(i, f"Item {i}") for i in items)


# ── Route ────────────────────────────────────────────────────────────────────

@dataclass
class Route:
    """
    One routing decision for one query.

    categories  FinDER categories whose Items Track A is restricted to (union).
    alpha       scale on the config's section_alpha: 1.0 routes, 0.0 abstains.
    band        "confident" / "ambiguous" / "uncertain" ("oracle" for the oracle router).
    source      who made the final call: "oracle", "classifier", "llm", or
                "fallback" (LLM output unusable -> deterministic band decision).
    probs       classifier probabilities, rounded, for analysis.
    reason      the agent's one-line rationale, when it made the call.
    """
    categories: tuple[str, ...]
    alpha:      float
    band:       str
    source:     str
    probs:      dict = field(default_factory=dict)
    reason:     str | None = None

    @property
    def items(self) -> frozenset[str]:
        """Raw CATEGORY_ITEMS spellings (incl. OCR variants) -- what the Qdrant filter needs.
        מקבל אובייקט של ראוט """
        out: set[str] = set()
        for c in self.categories:
            out |= CATEGORY_ITEMS.get(c, frozenset()) # |= is the union operator. we output the target sections. 
        return frozenset(out)

    @property
    def active(self) -> bool:
        return self.alpha > 0 and bool(self.items)

    def to_json(self) -> str:
        d = asdict(self)
        d["categories"] = list(self.categories)
        return json.dumps(d)

    @classmethod
    def from_json(cls, s: str) -> "Route":
        d = json.loads(s)
        d["categories"] = tuple(d["categories"])
        return cls(**d)

    @classmethod
    def oracle(cls, category: str) -> "Route":
        cat = (category or "").lower().strip()
        known = cat in CATEGORY_ITEMS
        return cls(categories=(cat,) if known else (), alpha=1.0 if known else 0.0,
                   band="oracle", source="oracle")


# ── Classifier ───────────────────────────────────────────────────────────────

def build_pipeline():
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import FeatureUnion, Pipeline


    #** THIS IS THE MODEL TO TRAIN  **
    features = FeatureUnion([ # TFIDF features. 
        ("word", TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=2)), # transformer object 1
        ("char", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), sublinear_tf=True, min_df=2)), # transformer object 2
    ])
    return Pipeline(
        [("features", features),
         ("clf", LogisticRegression(C=LOGREG_C, max_iter=3000))])
# **END OF MODEL**


def _lowest_threshold(score: np.ndarray, ok: np.ndarray, target: float, min_support: int) -> float:
    """
    Lowest cut t such that, among queries with score >= t, the fraction `ok` is
    >= target and there are at least min_support of them. Returns inf (the band
    never fires) when no cut qualifies.
    """
    order = np.argsort(-score)
    s, hit = score[order], ok[order].astype(float)
    n      = np.arange(1, len(s) + 1)
    prec   = np.cumsum(hit) / n
    valid  = (prec >= target) & (n >= min_support)
    if not valid.any():
        return float("inf")
    return float(s[np.where(valid)[0].max()])


def tune_thresholds(P: np.ndarray, y: np.ndarray, classes: np.ndarray) -> dict:
    top2  = np.argsort(-P, axis=1)[:, :2]
    p1    = P[np.arange(len(P)), top2[:, 0]]
    p12   = p1 + P[np.arange(len(P)), top2[:, 1]]
    truth = [canonical_items(t) for t in y]
    cover1 = np.array([t <= canonical_items(classes[a]) for t, a in zip(truth, top2[:, 0])])
    cover2 = np.array([t <= (canonical_items(classes[a]) | canonical_items(classes[b]))
                       for t, (a, b) in zip(truth, top2)])

    tau_hard = _lowest_threshold(p1, cover1, TARGET_HARD_PRECISION, MIN_THRESHOLD_SUPPORT)
    rest     = p1 < tau_hard
    tau_pair = _lowest_threshold(p12[rest], cover2[rest], TARGET_PAIR_COVERAGE, MIN_THRESHOLD_SUPPORT)

    confident = ~rest
    ambiguous = rest & (p12 >= tau_pair)
    uncertain = rest & ~ambiguous
    report = {
        "tau_hard": tau_hard,
        "tau_pair": tau_pair,
        "share_confident": float(confident.mean()),
        "share_ambiguous": float(ambiguous.mean()),
        "share_uncertain": float(uncertain.mean()),
        "cover_confident": float(cover1[confident].mean()) if confident.any() else None,
        "cover_ambiguous": float(cover2[ambiguous].mean()) if ambiguous.any() else None,
        # what routing the uncertain band to its top-1 / top-2 would have achieved
        "uncertain_top1_cover": float(cover1[uncertain].mean()) if uncertain.any() else None,
        "uncertain_top2_cover": float(cover2[uncertain].mean()) if uncertain.any() else None,
    }
    return report


def _fingerprint(train_ids: list[str]) -> str:
    import sklearn
    h = hashlib.sha1()
    h.update("\n".join(sorted(train_ids)).encode())
    h.update(f"C={LOGREG_C}|sk={sklearn.__version__}|hp={TARGET_HARD_PRECISION}|"
             f"pc={TARGET_PAIR_COVERAGE}|ms={MIN_THRESHOLD_SUPPORT}".encode())
    return h.hexdigest()[:12]


def train_classifier(
    exclude_ids:     set[str],
    exclude_tickers: set[str],
    model_path:      str = MODEL_PATH,
) -> dict:
    """
    Train on FinDER minus the evaluation questions, tune thresholds on OOF
    probabilities, fit the final model on all training rows, and save it.

    Excluding by ticker as well as by id holds out every question about an eval
    company, not just the eval questions themselves, so the eval measures routing
    on companies the classifier has never seen. Ticker resolution uses the same
    deterministic extractor the pipeline uses; ids catch anything it misses.
    """
    import joblib
    import sklearn
    from sklearn.metrics import accuracy_score, f1_score
    from sklearn.model_selection import StratifiedKFold, cross_val_predict
    from utils import extract_ticker_deterministic

    df = pd.read_parquet(FINDER_PATH)
    df = df[df["category"].notna()]
    tickers  = df["text"].map(lambda q: str(extract_ticker_deterministic(q) or "").lower()) # extract tickers
    excl_tk  = {t.lower() for t in exclude_tickers}
    held_out = df["_id"].isin(exclude_ids) | tickers.isin(excl_tk)
    train    = df[~held_out] # the DF without the eval tickers and questions
    print(f"FinDER: {len(df)} labelled | held out {held_out.sum()} "
          f"({df['_id'].isin(exclude_ids).sum()} eval ids, tickers {sorted(excl_tk)}) | "
          f"training on {len(train)}")
    
    # X and y data: only the necessary for fields for the classifier.
    X = train["text"].astype(object).to_numpy()
    y = train["category"].str.lower().astype(object).to_numpy()
    
    pipe    = build_pipeline() # this silently IS THE MODEL
    cv      = StratifiedKFold(CV_FOLDS, shuffle=True, random_state=CV_SEED)
    
    # oof = out of folds
    # here, most logic happens. The pipeline with the TFIDF features and the logistic regression 
    P_oof   = cross_val_predict(pipe, X, y, cv=cv, method="predict_proba") # Generate cross-validated estimates for each input data point.
    
    classes = np.array(sorted(set(y)))   # cross_val_predict orders columns like LogisticRegression.classes_
    pred    = classes[P_oof.argmax(1)]
    tuning  = tune_thresholds(P_oof, y, classes)

    print(f"OOF  acc={accuracy_score(y, pred):.3f}  macroF1={f1_score(y, pred, average='macro'):.3f}")
    print(f"     tau_hard={tuning['tau_hard']:.3f}  tau_pair={tuning['tau_pair']:.3f}")
    print(f"     bands  confident={tuning['share_confident']:.1%} (cover {tuning['cover_confident']:.3f})  "
          f"ambiguous={tuning['share_ambiguous']:.1%} (cover {tuning['cover_ambiguous']:.3f})  "
          f"uncertain={tuning['share_uncertain']:.1%}")
    if tuning["share_uncertain"]:
        print(f"     uncertain band would cover {tuning['uncertain_top1_cover']:.3f} (top-1) / "
              f"{tuning['uncertain_top2_cover']:.3f} (top-2) if routed")

    pipe.fit(X, y)
    assert list(pipe.classes_) == list(classes)

    artifact = {
        "pipeline":     pipe,
        "classes":      classes,
        "thresholds":   {"tau_hard": tuning["tau_hard"], "tau_pair": tuning["tau_pair"]},
        "oof_report":   tuning,
        "oof_accuracy": float(accuracy_score(y, pred)),
        "n_train":      len(train),
        "excluded_ids": sorted(exclude_ids),
        "excluded_tickers": sorted(excl_tk),
        "fingerprint":  _fingerprint(train["_id"].tolist()),
        "sklearn":      sklearn.__version__,
        "trained_at":   datetime.now().isoformat(),
    }
    Path(model_path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(value=artifact, filename=model_path)
    print(f"Saved -> {model_path}  (fingerprint {artifact['fingerprint']})")
    return artifact


class SectionClassifier:
    """Thin wrapper over the saved artifact: probabilities, evidence terms, routing."""

    def __init__(self, artifact: dict):
        self.pipeline    = artifact["pipeline"]
        # plain str, not np.str_, so routes serialize and compare as ordinary strings
        self.classes     = np.array([str(c) for c in artifact["classes"]], dtype=object)
        self.tau_hard    = artifact["thresholds"]["tau_hard"]
        self.tau_pair    = artifact["thresholds"]["tau_pair"]
        self.fingerprint = artifact["fingerprint"]
        self.artifact    = artifact

        word_vec          = self.pipeline.named_steps["features"].transformer_list[0][1]
        self._word_vec    = word_vec
        self._word_names  = word_vec.get_feature_names_out()
        n_word            = len(self._word_names)
        self._word_coef   = self.pipeline.named_steps["clf"].coef_[:, :n_word]

    @classmethod
    def load(cls, path: str = MODEL_PATH) -> "SectionClassifier":
        import joblib
        if not Path(path).exists():
            raise FileNotFoundError(f"No section classifier at {path}. Run evaluation/section_classifier.py first.")
        return cls(joblib.load(path))

    def predict_proba(self, queries: list[str]) -> np.ndarray:
        return self.pipeline.predict_proba(list(queries))

    def evidence_terms(self, query: str, class_idx: int, k: int = N_EVIDENCE_TERMS) -> list[str]:
        """
        Word n-grams in the query that pushed hardest toward this class (tf-idf
        weight x class coefficient). Char n-grams carry signal too but are
        unreadable fragments, so only the word half is surfaced to the agent.
        """
        x = self._word_vec.transform([query])
        idx, vals = x.indices, x.data
        if not len(idx):
            return []
        contrib = vals * self._word_coef[class_idx, idx]
        top = np.argsort(-contrib)[:k]
        return [self._word_names[idx[i]] for i in top if contrib[i] > 0]

    def decide(self, p: np.ndarray) -> Route:
        """Deterministic three-band policy for one probability row."""
        order = np.argsort(-p)
        c1, c2 = self.classes[order[0]], self.classes[order[1]]
        p1, p12 = p[order[0]], p[order[0]] + p[order[1]]
        probs = {self.classes[i]: round(float(p[i]), 4) for i in order}

        if p1 >= self.tau_hard:
            return Route((c1,), 1.0, "confident", "classifier", probs)
        if p12 >= self.tau_pair:
            return Route((c1, c2), 1.0, "ambiguous", "classifier", probs)
        return Route((), 0.0, "uncertain", "classifier", probs)


# ── Agent ────────────────────────────────────────────────────────────────────

_JSON_RE = re.compile(r"\{[^{}]*\}", re.DOTALL)
# Salvages the decision from a reply cut off mid-reason (the prompt asks for
# "categories" first precisely so a truncated reply still carries it).
_CATS_RE = re.compile(r'"categories"\s*:\s*\[([^\]]*)\]')


def _agent_user_message(query: str, clf: SectionClassifier, p: np.ndarray, band: str) -> tuple[str, list[str]]:
    order = np.argsort(-p)[:N_AGENT_CANDIDATES]
    candidates = [clf.classes[i] for i in order]
    lines = []
    for i in order:
        terms = clf.evidence_terms(query, i)
        lines.append(
            f'- "{clf.classes[i]}"  p={p[i]:.2f}  -> {_items_title(clf.classes[i])}'
            + (f'  | driven by: {", ".join(terms)}' if terms else "")
        )
    msg = (
        f"Question: {query}\n\n"
        f"Classifier confidence band: {band} "
        f"(top-1 needs p>={clf.tau_hard:.2f} to be trusted alone; "
        f"top-2 combined needs p>={clf.tau_pair:.2f})\n"
        f"Candidates:\n" + "\n".join(lines)
    )
    return msg, candidates


def _validate(cats, candidates: list[str]) -> tuple[str, ...] | None:
    chosen = [str(c).lower().strip() for c in cats]
    if any(c not in candidates for c in chosen):
        return None   # tried to route outside the candidate set
    # keep candidate order (probability order) and drop duplicates
    return tuple(c for c in candidates if c in chosen)


def _parse_agent(raw: str, candidates: list[str]) -> tuple[tuple[str, ...], str] | None:
    """(chosen categories, reason), () meaning abstain; None if the output is unusable."""
    for blob in reversed(_JSON_RE.findall(raw or "")):
        try:
            d = json.loads(blob)
        except json.JSONDecodeError:
            continue
        cats = d.get("categories")
        if not isinstance(cats, list):
            continue
        chosen = _validate(cats, candidates)
        return None if chosen is None else (chosen, str(d.get("reason", ""))[:300])

    # Qwen sometimes ends its turn one character early, without the closing brace.
    start = (raw or "").find("{")
    if start >= 0:
        try:
            d = json.loads(raw[start:].strip() + "}")
            if isinstance(d.get("categories"), list):
                chosen = _validate(d["categories"], candidates)
                return None if chosen is None else (chosen, str(d.get("reason", ""))[:300])
        except json.JSONDecodeError:
            pass

    m = _CATS_RE.search(raw or "")
    if m:
        try:
            cats = json.loads(f"[{m.group(1)}]")
        except json.JSONDecodeError:
            return None
        chosen = _validate(cats, candidates)
        return None if chosen is None else (chosen, "(reply truncated)")
    return None


def route_queries(
    queries:   list[str],
    clf:       SectionClassifier,
    model=None,
    tokenizer=None,
    completion_batch_size: int = 1,
    prefill_batch_size:    int = 1,
) -> dict[str, list[Route]]:
    """
    Route a batch of queries with both predicted routers.

    Returns {"classifier": [...], "agent": [...]}, one Route per query each.
    The agent's LLM calls (non-confident band only) go out as one batch. When no
    model is given -- and not on the API backend -- the agent column falls back to
    the classifier decisions, marked source="fallback".
    """
    import llm_backend
    from prompts import SECTION_ROUTER_PROMPT
    from evaluation.rag_functions import _batch_generate_texts, _chat_prompt

    P = clf.predict_proba(queries)
    clf_routes = [clf.decide(p) for p in P]
    agent_routes = list(clf_routes)

    pending = [i for i, r in enumerate(clf_routes) if r.band != "confident"]
    can_generate = llm_backend.USE_API or model is not None
    if not pending:
        return {"classifier": clf_routes, "agent": agent_routes}
    if not can_generate:
        for i in pending:
            agent_routes[i] = Route(**{**asdict(clf_routes[i]), "source": "fallback"})
        return {"classifier": clf_routes, "agent": agent_routes}

    tok    = None if llm_backend.USE_API else tokenizer
    system = SECTION_ROUTER_PROMPT.format(accuracy=clf.artifact["oof_accuracy"])
    prompts, cand_lists = [], []
    for i in pending:
        msg, cands = _agent_user_message(queries[i], clf, P[i], clf_routes[i].band)
        prompts.append(_chat_prompt(tok, system, msg))
        cand_lists.append(cands)

    texts = _batch_generate_texts(model, tokenizer, prompts, AGENT_MAX_TOKENS,
                                  completion_batch_size, prefill_batch_size)

    for i, raw, cands in zip(pending, texts, cand_lists):
        base   = clf_routes[i]
        parsed = _parse_agent(raw, cands)
        if parsed is None:
            print(f"  [router] unusable agent output, falling back: {raw[:120]!r}")
            agent_routes[i] = Route(**{**asdict(base), "source": "fallback"})
            continue
        chosen, reason = parsed
        agent_routes[i] = Route(chosen, 1.0 if chosen else 0.0, base.band, "llm", base.probs, reason)

    return {"classifier": clf_routes, "agent": agent_routes}


if __name__ == "__main__":
    from evaluation.build_evaluation_dataset import load_dataset, DATASET_PATH

    # Every question and every company in the eval dataset is held out of training.
    dataset, _ = load_dataset(DATASET_PATH)
    if dataset.empty:
        raise SystemExit(f"Eval dataset not found at {DATASET_PATH}; nothing to hold out.")
    train_classifier(
        exclude_ids=set(dataset["finder_id"]),
        exclude_tickers=set(dataset["ticker"]),
    )
