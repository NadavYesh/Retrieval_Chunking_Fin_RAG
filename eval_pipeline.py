#%%
import json
import random
import re
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import TypedDict, List, Optional
import pandas as pd
from langgraph.graph import StateGraph, END
from langchain_core.runnables import RunnableConfig

from search_engine import search_with_payload, generate_llm_answer, SYSTEM_PROMPT
from FinDER import run_finder
from db.database import get_qdrant_client
from mlx_lm import generate

client = get_qdrant_client()

# ─────────────────────────────────────────────────────────────────────────────
# Collection registry — add / remove levels here only
# ─────────────────────────────────────────────────────────────────────────────
COLLECTIONS = {
    "doc":      {"coll_name": "--level 0", "use_parent_fetch": False},
    "header":   {"coll_name": "--level 1", "use_parent_fetch": False},
    "child":    {"coll_name": "--level 2", "use_parent_fetch": True},
    "enriched": {"coll_name": "--level 3", "use_parent_fetch": True},
}
# this is the chunk given to generator. 
PARENT_COLL = "--level 1"   # parent_fetch always pulls header chunks


# ─────────────────────────────────────────────────────────────────────────────
# State
# ─────────────────────────────────────────────────────────────────────────────
class EvalState(TypedDict):
    raw_query:        str
    ticker:           Optional[str]
    year:             Optional[int]
    form_type:        Optional[str]
    level:            str
    use_parent_fetch: bool
    retrieved_points: Optional[List]   # ScoredPoint from vector search
    context_points:   Optional[List]   # what reaches the generator (may be parent chunks)
    answer:           Optional[str]
    ref_answer:       Optional[str]
    run_id:           str
    error:            Optional[str]


# ─────────────────────────────────────────────────────────────────────────────
# Metadata extraction — called ONCE per question, result shared across all levels
# ─────────────────────────────────────────────────────────────────────────────
def extract_metadata(raw_query: str, model, tokenizer) -> dict:
    """
    Uses the same SYSTEM_PROMPT as search_engine to extract ticker, year,
    form_type, and an embedding-optimized query from a natural language question.
    """
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": raw_query},
    ]
    prompt   = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    response = generate(model, tokenizer, prompt=prompt, verbose=False).lower()

    try:
        match = re.search(r'\{.*\}', response, re.DOTALL)
        if not match:
            print("  [extract_metadata] WARNING: no JSON found in LLM response; using raw query")
            return {"raw_query": raw_query, "ticker": None, "year": None, "form_type": None}
        data = json.loads(match.group())
        result = {
            "raw_query": data.pop("raw_prompt", raw_query),
            "ticker":    data.get("ticker"),
            "year":      data.get("year"),
            "form_type": data.get("form_type"),
        }
        return result
    except Exception as e:
        print(f"  [extract_metadata] WARNING: parse error ({e}); using raw query")
        return {"raw_query": raw_query, "ticker": None, "year": None, "form_type": None}


# ─────────────────────────────────────────────────────────────────────────────
# Nodes
# ─────────────────────────────────────────────────────────────────────────────
def retrieve_node(state: EvalState, config: RunnableConfig) -> dict:
    """Embed query and fetch top-k chunks from the configured Qdrant collection."""
    cfg         = config["configurable"]
    embed_model = cfg["embed_model"]
    coll_name   = cfg["coll_name"]

    filters = {k: state[k] for k in ("ticker", "year", "form_type") if state.get(k)}
    print(f"  [retrieve:{state['level']}] collection='{coll_name}' filters={filters}")

    try:
        query_vec = embed_model.encode(state["raw_query"], prompt_name="query").tolist()
        results   = search_with_payload(coll_name, query_vec, payload_must=filters)
        points    = results.points if hasattr(results, "points") else []
        scores    = [round(p.score, 3) for p in points] if points else []
        print(f"  [retrieve:{state['level']}] {len(points)} hits | scores={scores}")
    except Exception as e:
        print(f"  [retrieve:{state['level']}] ERROR: {e}")
        return {"error": str(e), "retrieved_points": [], "context_points": []}

    if not points:
        print(f"  [retrieve:{state['level']}] WARNING: 0 results — check filters or collection name")

    return {"retrieved_points": points, "context_points": points, "error": None}


def parent_fetch_node(state: EvalState, config: RunnableConfig) -> dict:
    """
    For child/enriched levels: swap retrieved child chunks for their parent
    header chunks so the generator sees richer, complete-section context.
    Falls back to child chunks if parent IDs are missing or fetch fails.
    """
    parent_coll = config["configurable"].get("parent_coll_name", PARENT_COLL)

    parent_ids = list({
        p.payload.get("parent_id")
        for p in (state.get("retrieved_points") or [])
        if p.payload.get("parent_id")
    })

    if not parent_ids:
        print(f"  [parent_fetch:{state['level']}] WARNING: no parent_id found in retrieved chunks; passing child chunks to generator")
        return {}

    try:
        parent_records = client.retrieve(parent_coll, ids=parent_ids, with_payload=True)
        ctx_tokens_est = sum(len(r.payload.get("text", "").split()) for r in parent_records)
        print(f"  [parent_fetch:{state['level']}] {len(parent_ids)} child → {len(parent_records)} header chunks (~{ctx_tokens_est} words passed to generator)")
        return {"context_points": parent_records}
    except Exception as e:
        print(f"  [parent_fetch:{state['level']}] ERROR: {e}; falling back to child chunks")
        return {}


def generate_node(state: EvalState, config: RunnableConfig) -> dict:
    """Generate an answer from context_points using the generation LLM."""
    cfg       = config["configurable"]
    model     = cfg["gen_model"]
    tokenizer = cfg["gen_tokenizer"]

    n_ctx = len(state.get("context_points") or [])
    print(f"  [generate:{state['level']}] {n_ctx} context chunk(s) → LLM")

    if not n_ctx:
        print(f"  [generate:{state['level']}] WARNING: empty context; returning fallback answer")
        return {"answer": "No relevant context retrieved."}

    wrapped   = SimpleNamespace(points=state["context_points"])
    answer, _ = generate_llm_answer(state["raw_query"], wrapped, model, tokenizer)
    print(f"  [generate:{state['level']}] answer ({len(answer)} chars): {answer[:120].strip()}{'...' if len(answer) > 120 else ''}")
    return {"answer": answer}


def collect_node(state: EvalState, config: RunnableConfig) -> dict:
    """Append a result row to the shared results list in config."""
    config["configurable"]["results_list"].append({
        "run_id":          state["run_id"],
        "raw_query":       state["raw_query"],
        "ref_answer":      state.get("ref_answer", ""),
        "ticker":          state.get("ticker"),
        "year":            state.get("year"),
        "level":           state["level"],
        "answer":          state.get("answer", ""),
        "error":           state.get("error"),
        "retrieved_ids":   [p.id for p in (state.get("retrieved_points") or [])],
        "context_ids":     [p.id for p in (state.get("context_points") or [])],
        "context_text":    " ||| ".join(
            p.payload.get("text", "")[:400]
            for p in (state.get("context_points") or [])
        ),
    })
    return {}


# ─────────────────────────────────────────────────────────────────────────────
# Routing
# ─────────────────────────────────────────────────────────────────────────────
def route_after_retrieve(state: EvalState) -> str:
    return "parent_fetch" if state.get("use_parent_fetch") else "generate"


# ─────────────────────────────────────────────────────────────────────────────
# Graph factory
# ─────────────────────────────────────────────────────────────────────────────
def create_eval_graph():
    """
    retrieve → [parent_fetch?] → generate → collect → END

    parent_fetch is bypassed for doc/header levels via conditional edge.
    Add or remove nodes here without touching the evaluation loop.
    """
    wf = StateGraph(EvalState)
    wf.add_node("retrieve",     retrieve_node)
    wf.add_node("parent_fetch", parent_fetch_node)
    wf.add_node("generate",     generate_node)
    wf.add_node("collect",      collect_node)

    wf.set_entry_point("retrieve")
    wf.add_conditional_edges(
        "retrieve", route_after_retrieve,
        {"parent_fetch": "parent_fetch", "generate": "generate"},
    )
    wf.add_edge("parent_fetch", "generate")
    wf.add_edge("generate",     "collect")
    wf.add_edge("collect",      END)
    return wf.compile()


# ─────────────────────────────────────────────────────────────────────────────
# Outer evaluation loop
# ─────────────────────────────────────────────────────────────────────────────
def run_evaluation(
    gen_model,
    gen_tokenizer,
    embed_model,
    finder_df,
    output_dir: str = "/Users/nadavsmacbookair/Desktop/Thesis/data/eval_results",
    top_k: int = 5,
    levels: list = None,
) -> pd.DataFrame:
    """
    Runs all FinDER questions against all (or selected) granularity levels.

    Args:
        levels: subset of COLLECTIONS keys to run; None = all four levels.

    Returns:
        DataFrame with one row per (question × level).
        Also saved to CSV + pickle under output_dir.
    """
    if levels is None:
        levels = list(COLLECTIONS.keys())

    print("Loading FinDER questions...")
    
    if finder_df.empty:
        print("No FinDER questions found.")
        return pd.DataFrame()
    print(f"FinDER: {len(finder_df)} questions after company filter.")
    print(f"Levels to evaluate: {levels}")
    print(f"Total graph runs: {len(finder_df) * len(levels)}")

    results_list = []
    graph        = create_eval_graph()

    base_config = {
        "embed_model":      embed_model,
        "gen_model":        gen_model,
        "gen_tokenizer":    gen_tokenizer,
        "parent_coll_name": PARENT_COLL,
        "top_k":            top_k,
        "results_list":     results_list,
    }

    for q_idx, (_, row) in enumerate(finder_df.iterrows()):
        raw_query = row["text"]
        print(f"\n[Q {q_idx+1}/{len(finder_df)}] {raw_query[:80]}...")

        # Extract once — same ticker/year/prompt used for every level
        meta = extract_metadata(raw_query, gen_model, gen_tokenizer)
        print(f"  extracted → ticker={meta['ticker']} year={meta['year']} form_type={meta['form_type']}")
        print(f"  optimized → {meta['raw_query'][:100]}{'...' if len(meta['raw_query']) > 100 else ''}")

        for level in levels:
            level_cfg = COLLECTIONS[level]
            run_id    = f"{meta.get('ticker','?')}_{meta.get('year','?')}_{level}_{q_idx}"
            print(f"\n  ── level: {level} (run {q_idx * len(levels) + levels.index(level) + 1}/{len(finder_df) * len(levels)}) ──")

            initial_state: EvalState = {
                "raw_query":        meta["raw_query"],
                "ticker":           meta.get("ticker"),
                "year":             meta.get("year"),
                "form_type":        meta.get("form_type"),
                "level":            level,
                "use_parent_fetch": level_cfg["use_parent_fetch"],
                "run_id":           run_id,
                "retrieved_points": None,
                "context_points":   None,
                "answer":           None,
                "ref_answer":       str(row.get("answer", "")) if hasattr(row, "get") else str(row["answer"]) if "answer" in row.index else "",
                "error":            None,
            }
            thread_cfg = {"configurable": {
                **base_config,
                "coll_name": level_cfg["coll_name"],
            }}

            try:
                graph.invoke(initial_state, config=thread_cfg)
            except Exception as e:
                print(f"  [ERROR] {level}: {e}")
                results_list.append({
                    "run_id": run_id, "raw_query": raw_query,
                    "raw_query": meta["raw_query"],
                    "ticker": meta.get("ticker"), "year": meta.get("year"),
                    "level": level, "answer": "", "error": str(e),
                    "retrieved_ids": [], "context_ids": [], "context_text": "",
                })

    # ── Save ──────────────────────────────────────────────────────────────────
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    ts         = datetime.now().strftime("%Y%m%d_%H%M")
    results_df = pd.DataFrame(results_list)

    n_ok    = results_df["error"].isna().sum()
    n_err   = results_df["error"].notna().sum()
    n_empty = (results_df["answer"] == "No relevant context retrieved.").sum()
    print(f"\n── Evaluation complete ──────────────────────────────")
    print(f"  Total runs : {len(results_df)}")
    print(f"  Successful : {n_ok}")
    print(f"  Empty ctx  : {n_empty}")
    print(f"  Errors     : {n_err}")

    results_df.to_csv(f"{output_dir}/eval_{ts}.csv",  index=False)
    results_df.to_pickle(f"{output_dir}/eval_{ts}.pkl")
    print(f"  Saved      → {output_dir}/eval_{ts}.{{csv,pkl}}")
    return results_df


# ─────────────────────────────────────────────────────────────────────────────
# Pairwise LLM-judge evaluation
# ─────────────────────────────────────────────────────────────────────────────
JUDGE_PROMPT = """\
[System]
Please act as an impartial judge and evaluate the quality of the responses provided by two \
AI assistants to the user question displayed below. Your evaluation should consider \
correctness and helpfulness. You will be given a reference answer, assistant A's answer, \
and assistant B's answer. Your job is to evaluate which assistant's answer is better. \
Begin your evaluation by comparing both assistants' answers with the reference answer. \
Identify and correct any mistakes. Avoid any position biases and ensure that the order in \
which the responses were presented does not influence your decision. Do not allow the \
length of the responses to influence your evaluation. Do not favor certain names of the \
assistants. Be as objective as possible. After providing your explanation, output your \
final verdict by strictly following this format: "[[A]]" if assistant A is better, "[[B]]" \
if assistant B is better, and "[[C]]" for a tie.
[User Question]
{question}
[The Start of Reference Answer]
{answer_ref}
[The End of Reference Answer]
[The Start of Assistant A's Answer]
{answer_a}
[The End of Assistant A's Answer]
[The Start of Assistant B's Answer]
{answer_b}
[The End of Assistant B's Answer]\
"""


def _parse_verdict(text: str) -> str:
    """Extract A, B, or C from [[X]] pattern. Returns '?' if unparseable."""
    m = re.search(r'\[\[([ABC])\]\]', text, re.IGNORECASE)
    return m.group(1).upper() if m else "?"


def judge_pair(
    question:   str,
    answer_ref: str,
    answer_a:   str,
    answer_b:   str,
    model,
    tokenizer,
    max_tokens: int = 512,
) -> tuple[str, str]:
    """
    Ask the judge LLM to compare answer_a vs answer_b given answer_ref.
    Returns (verdict, raw_response) where verdict is 'A', 'B', 'C', or '?'.
    """
    content  = JUDGE_PROMPT.format(
        question=question, answer_ref=answer_ref,
        answer_a=answer_a, answer_b=answer_b,
    )
    prompt   = tokenizer.apply_chat_template(
        [{"role": "user", "content": content}],
        tokenize=False, add_generation_prompt=True,
    )
    raw = generate(model, tokenizer, prompt=prompt, max_tokens=max_tokens, verbose=False)
    return _parse_verdict(raw), raw


def run_pairwise_eval(
    results_df:      pd.DataFrame,
    finder_df:       pd.DataFrame,
    judge_model,
    judge_tokenizer,
    baseline_level:  str = "doc",
    ref_col:         str = "answer",
    output_dir:      str = "/Users/nadavsmacbookair/Desktop/Thesis/data/eval_results",
    seed:            int = 42,
) -> pd.DataFrame:
    """
    LLM-judge pairwise comparison: baseline_level vs every other level in results_df.

    For each (question, comparison_level) pair the A/B order is randomized to
    mitigate position bias; the verdict is mapped back to the actual winning level.

    Args:
        results_df:     Output of run_evaluation() — one row per (question × level).
        finder_df:      FinDER DataFrame; must contain 'text' and ref_col columns.
        baseline_level: Level treated as the reference system (default "doc").
        ref_col:        Column in finder_df with gold reference answers.

    Returns:
        DataFrame with one row per (question × comparison_level).
        Also saves a win-rate summary and the full comparison table to output_dir.
    """
    random.seed(seed)

    if ref_col not in finder_df.columns:
        raise ValueError(
            f"Column '{ref_col}' not found in finder_df. Available: {list(finder_df.columns)}"
        )

    ref_lookup        = dict(zip(finder_df["text"], finder_df[ref_col]))
    comparison_levels = [l for l in results_df["level"].unique() if l != baseline_level]

    if not comparison_levels:
        raise ValueError(f"No levels to compare against baseline '{baseline_level}'.")

    n_questions    = len(results_df["raw_query"].unique())
    n_comparisons  = n_questions * len(comparison_levels)
    print(f"\n── Pairwise judge evaluation ────────────────────────")
    print(f"  Baseline      : {baseline_level}")
    print(f"  Comparing vs  : {comparison_levels}")
    print(f"  Questions     : {n_questions}")
    print(f"  Total pairs   : {n_comparisons}")
    print(f"  Position swap : randomized (seed={seed})")

    rows      = []
    questions = results_df["raw_query"].unique()

    for q_idx, raw_query in enumerate(questions):
        q_rows     = results_df[results_df["raw_query"] == raw_query]
        ref_answer = ref_lookup.get(raw_query, "")

        baseline_row = q_rows[q_rows["level"] == baseline_level]
        if baseline_row.empty:
            print(f"  [Judge Q {q_idx+1}] SKIP: no baseline answer found")
            continue
        baseline_answer = baseline_row.iloc[0]["answer"]

        has_ref = bool(ref_answer.strip())
        print(f"\n[Judge Q {q_idx+1}/{n_questions}] {raw_query[:70]}...")
        if not has_ref:
            print(f"  WARNING: no reference answer found for this question")

        for comp_level in comparison_levels:
            comp_row = q_rows[q_rows["level"] == comp_level]
            if comp_row.empty:
                continue
            comp_answer = comp_row.iloc[0]["answer"]

            # Randomize A/B order to reduce position bias
            swapped = random.random() < 0.5
            if swapped:
                answer_a, answer_b = comp_answer,     baseline_answer
                level_a,  level_b  = comp_level,      baseline_level
            else:
                answer_a, answer_b = baseline_answer, comp_answer
                level_a,  level_b  = baseline_level,  comp_level

            pair_num = q_idx * len(comparison_levels) + comparison_levels.index(comp_level) + 1
            print(f"  [{pair_num}/{n_comparisons}] {baseline_level} vs {comp_level}  A={level_a}  swapped={swapped}", end="  ")
            verdict, raw_response = judge_pair(
                raw_query, ref_answer, answer_a, answer_b,
                judge_model, judge_tokenizer,
            )
            winner_display = level_a if verdict == "A" else (level_b if verdict == "B" else "tie")
            print(f"→ [[{verdict}]] winner={winner_display}")

            winner = level_a if verdict == "A" else (level_b if verdict == "B" else "tie")

            rows.append({
                "raw_query":       raw_query,
                "ticker":          baseline_row.iloc[0].get("ticker"),
                "year":            baseline_row.iloc[0].get("year"),
                "baseline_level":  baseline_level,
                "comp_level":      comp_level,
                "level_a":         level_a,
                "level_b":         level_b,
                "answer_a":        answer_a,
                "answer_b":        answer_b,
                "ref_answer":      ref_answer,
                "swapped":         swapped,
                "verdict":         verdict,
                "winner":          winner,
                "judge_reasoning": raw_response,
            })

    pairwise_df = pd.DataFrame(rows)

    # ── Win-rate summary ──────────────────────────────────────────────────────
    if not pairwise_df.empty:
        n = pairwise_df.groupby("comp_level")["winner"].count().rename("n_questions")
        win_rates = (
            pairwise_df.groupby(["comp_level", "winner"])
            .size()
            .unstack(fill_value=0)
            .div(n, axis=0)
            .round(3)
        )
        print("\n── Win-rate summary (rows = comparison level) ──")
        print(win_rates.to_string())

    # ── Save ──────────────────────────────────────────────────────────────────
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M")
    print("Saving...")
    pairwise_df.to_csv(f"{output_dir}/pairwise_{ts}.csv",    index=False)
    pairwise_df.to_pickle(f"{output_dir}/pairwise_{ts}.pkl")
    print(f"\nSaved → {output_dir}/pairwise_{ts}.{{csv,pkl}}")
    return pairwise_df


# ─────────────────────────────────────────────────────────────────────────────
#%%
if __name__ == "__main__":
    from mlx_lm import load
    from models import MLXEmbedder
    tickers = ["nvda"]
    finder_df = run_finder(tickers=tickers).head(2)
    print("Loading generation model...")
    gen_model, gen_tokenizer = load("mlx-community/Llama-3.2-3B-Instruct-4bit")

    print("Loading embedding model...")
    embed_model = MLXEmbedder("mlx-community/embeddinggemma-300m-bf16")

    # ── Stage 1: generate answers for all levels ──────────────────────────────
    results = run_evaluation(
        gen_model=gen_model,
        gen_tokenizer=gen_tokenizer,
        embed_model=embed_model,
        finder_df=finder_df, #temp
        levels=[#"doc", 
                "header", "child", "enriched"],
    )

    # ── Stage 2: pairwise judge comparison ───────────────────────────────────
    if not results.empty:
        print(f"\nDone generating. {len(results)} total runs.")
        pairwise = run_pairwise_eval(
            results_df=results,
            finder_df=finder_df,
            judge_model=gen_model,        # reuse; swap for a stronger judge if available
            judge_tokenizer=gen_tokenizer,
            baseline_level="doc",
        )
