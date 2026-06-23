#%%
import json
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
PARENT_COLL = "--level 1"   # parent_fetch always pulls header chunks


# ─────────────────────────────────────────────────────────────────────────────
# State
# ─────────────────────────────────────────────────────────────────────────────
class EvalState(TypedDict):
    raw_query:        str
    optimized_query:  str
    ticker:           Optional[str]
    year:             Optional[int]
    form_type:        Optional[str]
    level:            str
    use_parent_fetch: bool
    retrieved_points: Optional[List]   # ScoredPoint from vector search
    context_points:   Optional[List]   # what reaches the generator (may be parent chunks)
    answer:           Optional[str]
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
            return {"optimized_query": raw_query, "ticker": None, "year": None, "form_type": None}
        data = json.loads(match.group())
        return {
            "optimized_query": data.pop("optimized_prompt", raw_query),
            "ticker":    data.get("ticker"),
            "year":      data.get("year"),
            "form_type": data.get("form_type"),
        }
    except Exception:
        return {"optimized_query": raw_query, "ticker": None, "year": None, "form_type": None}


# ─────────────────────────────────────────────────────────────────────────────
# Nodes
# ─────────────────────────────────────────────────────────────────────────────
def retrieve_node(state: EvalState, config: RunnableConfig) -> dict:
    """Embed query and fetch top-k chunks from the configured Qdrant collection."""
    cfg         = config["configurable"]
    embed_model = cfg["embed_model"]
    coll_name   = cfg["coll_name"]

    print(f"  [retrieve] level={state['level']} | ticker={state.get('ticker')} year={state.get('year')}")

    filters = {k: state[k] for k in ("ticker", "year", "form_type") if state.get(k)}

    try:
        query_vec = embed_model.encode(state["optimized_query"], prompt_name="query").tolist()
        results   = search_with_payload(coll_name, query_vec, payload_must=filters)
        points    = results.points if hasattr(results, "points") else []
    except Exception as e:
        return {"error": str(e), "retrieved_points": [], "context_points": []}

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
        return {}   # keep context_points unchanged

    try:
        parent_records = client.retrieve(parent_coll, ids=parent_ids, with_payload=True)
        print(f"  [parent_fetch] {len(parent_ids)} child → {len(parent_records)} header chunks")
        return {"context_points": parent_records}
    except Exception as e:
        print(f"  [parent_fetch] failed ({e}); using child chunks as context")
        return {}


def generate_node(state: EvalState, config: RunnableConfig) -> dict:
    """Generate an answer from context_points using the generation LLM."""
    cfg       = config["configurable"]
    model     = cfg["gen_model"]
    tokenizer = cfg["gen_tokenizer"]

    if not state.get("context_points"):
        return {"answer": "No relevant context retrieved."}

    # generate_llm_answer expects an object with .points; SimpleNamespace bridges
    # ScoredPoint (from retrieve) and Record (from parent_fetch) transparently
    wrapped   = SimpleNamespace(points=state["context_points"])
    answer, _ = generate_llm_answer(state["raw_query"], wrapped, model, tokenizer)
    return {"answer": answer}


def collect_node(state: EvalState, config: RunnableConfig) -> dict:
    """Append a result row to the shared results list in config."""
    config["configurable"]["results_list"].append({
        "run_id":          state["run_id"],
        "raw_query":       state["raw_query"],
        "optimized_query": state["optimized_query"],
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
    finder_df = run_finder()
    if finder_df.empty:
        print("No FinDER questions found.")
        return pd.DataFrame()
    print(f"{len(finder_df)} questions after company filter.")

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
        print(f"  ticker={meta['ticker']} year={meta['year']}")

        for level in levels:
            level_cfg = COLLECTIONS[level]
            run_id    = f"{meta.get('ticker','?')}_{meta.get('year','?')}_{level}_{q_idx}"
            print(f"  → {level}")

            initial_state: EvalState = {
                "raw_query":        raw_query,
                "optimized_query":  meta["optimized_query"],
                "ticker":           meta.get("ticker"),
                "year":             meta.get("year"),
                "form_type":        meta.get("form_type"),
                "level":            level,
                "use_parent_fetch": level_cfg["use_parent_fetch"],
                "run_id":           run_id,
                "retrieved_points": None,
                "context_points":   None,
                "answer":           None,
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
                    "optimized_query": meta["optimized_query"],
                    "ticker": meta.get("ticker"), "year": meta.get("year"),
                    "level": level, "answer": "", "error": str(e),
                    "retrieved_ids": [], "context_ids": [], "context_text": "",
                })

    # ── Save ──────────────────────────────────────────────────────────────────
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    ts         = datetime.now().strftime("%Y%m%d_%H%M")
    results_df = pd.DataFrame(results_list)
    results_df.to_csv(f"{output_dir}/eval_{ts}.csv",  index=False)
    results_df.to_pickle(f"{output_dir}/eval_{ts}.pkl")
    print(f"\nSaved → {output_dir}/eval_{ts}.{{csv,pkl}}")
    return results_df


# ─────────────────────────────────────────────────────────────────────────────
#%%
if __name__ == "__main__":
    from mlx_lm import load
    from models import MLXEmbedder

    print("Loading generation model...")
    gen_model, gen_tokenizer = load("mlx-community/Llama-3.2-3B-Instruct-4bit")

    print("Loading embedding model...")
    embed_model = MLXEmbedder("mlx-community/embeddinggemma-300m-bf16")

    results = run_evaluation(
        gen_model=gen_model,
        gen_tokenizer=gen_tokenizer,
        embed_model=embed_model,
        levels=["doc", "header", "child", "enriched"],
    )

    if not results.empty:
        print(f"\nDone. {len(results)} total runs.")
        print(results[["level", "ticker", "year", "answer"]].head(20).to_string())
