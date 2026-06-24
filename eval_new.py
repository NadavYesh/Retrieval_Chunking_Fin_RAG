#%%
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
import pandas as pd

from search_engine import search_with_payload, generate_llm_answer
from prompts import SYSTEM_PROMPT
from FinDER import run_finder
from db.database import get_qdrant_client
from mlx_lm import generate, load
import mlx.core
from mlx_embeddings.utils import load as emb_load
from utils import parse_metadata_response

client = get_qdrant_client()

COLLECTIONS_1 = {
    "doc":      {"coll_name": "--level 0", "use_parent_fetch": False},
    "header":   {"coll_name": "--level 1", "use_parent_fetch": False},
    "child":    {"coll_name": "--level 2", "use_parent_fetch": True},
    "enriched": {"coll_name": "--level 3", "use_parent_fetch": True},
}

COLLECTIONS_2 = {
    "header":   {"coll_name": "NVDA --level 1", "use_parent_fetch": False},
}
PARENT_COLL = "NVDA --level 1"


def extract_metadata(query: str, model, tokenizer) -> dict:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": query},
    ]
    prompt   = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    response = generate(model, tokenizer, prompt=prompt, verbose=False)
    return parse_metadata_response(response, fallback_query=query)


def run_evaluation(
    finder_df:  pd.DataFrame,
    gen_model = None,
    gen_tokenizer = None,
    embed_model = None,
    embed_tokenizer = None,
    output_dir: str  = "/Users/nadavsmacbookair/Desktop/Thesis/data/eval_results",
    top_k:      int  = 5,
    levels:     list = None,
    ) -> pd.DataFrame:
    if levels is None:
        levels = list(COLLECTIONS_2.keys())
    if isinstance(levels, str):
        levels = [levels]

    if finder_df.empty:
        print("No FinDER questions found.")
        return pd.DataFrame()

    print(f"Questions: {len(finder_df)} | Levels: {levels} | Total runs: {len(finder_df) * len(levels)}")

    results_list = []

    for q_idx, (_, row) in enumerate(finder_df.iterrows()):
        query        = row.get("query","")
        truth_answer = row.get("truth_answer", "")
        truth_ref    = row.get("truth_ref", "")
        print(f"\n[Q {q_idx+1}/{len(finder_df)}] {query[:80]}...")

        meta = extract_metadata(query, gen_model, gen_tokenizer)
        print(f"  ticker={meta['ticker']} year={meta['year']} form_type={meta['form_type']}")
        #print(f"  optimized → {meta['optimized_query'][:100]}...")

        filters = {k: meta[k] for k in ("ticker", "year", "form_type") if meta.get(k)}

        for level in levels:
            level_cfg        = COLLECTIONS_2[level] # this is where we address the collection
            coll_name        = level_cfg["coll_name"]
            use_parent_fetch = level_cfg["use_parent_fetch"]
            run_id  = f"{meta.get('ticker','?')}_{meta.get('year','?')}_{level}_{q_idx}"
            run_num = q_idx * len(levels) + levels.index(level) + 1
            print(f"\n  ── level: {level} ({run_num}/{len(finder_df) * len(levels)}) ──")

            error          = None
            context_points = []
            rag_answer     = ""

            try:
                # retrieve
                print(f"  [retrieve] collection='{coll_name}' filters={filters}")
                ######### GEMMA SPECIFIC ##############
                formatted_query = f"task: search result | query: {query}"
                query_tokens     = embed_tokenizer.encode(formatted_query, return_tensors="mlx")
                embed_outputs_    = embed_model(query_tokens)
                # if I dont flatten, then we get an mlx object, not suitable for qdrant. 
                query_vec = embed_outputs_.text_embeds.tolist()[0]  # flatten to plain list for Qdrant
                results          = search_with_payload(coll_name, query_vec, payload_must=filters, top_k=top_k)
                retrieved_points = results.points if hasattr(results, "points") else []
                context_points   = retrieved_points
                print(f"  [retrieve] {len(retrieved_points)} hits | scores={[round(p.score, 3) for p in retrieved_points]}")

                # parent fetch
                if use_parent_fetch:
                    parent_ids = list({
                        p.payload.get("parent_id")
                        for p in retrieved_points
                        if p.payload.get("parent_id")
                    })
                    if parent_ids:
                        parent_records = client.retrieve(PARENT_COLL, ids=parent_ids, with_payload=True)
                        word_count     = sum(len(r.payload.get("text", "").split()) for r in parent_records)
                        print(f"  [parent_fetch] {len(parent_ids)} child → {len(parent_records)} header chunks (~{word_count} words)")
                        context_points = parent_records
                    else:
                        print(f"  [parent_fetch] WARNING: no parent_ids found; using child chunks")

                # generate
                print(f"  [generate] {len(context_points)} context chunk(s) → LLM")
                if not context_points:
                    rag_answer = "No relevant context retrieved."
                else:
                    wrapped    = SimpleNamespace(points=context_points)
                    print(context_points)
                    rag_answer, _ = generate_llm_answer(meta["optimized_query"], wrapped, gen_model, gen_tokenizer)
                    print(f"  [generate] {len(rag_answer)} chars: {rag_answer[:120].strip()}{'...' if len(rag_answer) > 120 else ''}")

            except Exception as e:
                print(f"  [ERROR] {level}: {e}")
                error = str(e)
            rag_ret=[]
            for p_n,p in enumerate(context_points):
                rag_ret.append(f"======================\nSource Number {p_n}\n {p}")
            rag_ret = "".join(rag_ret)
            
            results_list.append({
                "run_id":        run_id,
                "level":         level,
                "query":         query,
                "truth_answer":  truth_answer,
                "truth_ref":     truth_ref,
                "rag_answer":    rag_answer,
                "rag_retrieved": rag_ret
                #"error":         error,
            })

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    ts         = datetime.now().strftime("%Y%m%d_%H%M")
    results_df = pd.DataFrame(results_list)


    n_empty = (results_df["rag_answer"] == "No relevant context retrieved.").sum()
    print(f"\n── Evaluation complete ──")
    print(f"  Total: {len(results_df)} | ")

    results_df.to_csv(f"{output_dir}/eval_{ts}.csv",   index=False)
    results_df.to_pickle(f"{output_dir}/eval_{ts}.pkl")
    results_df.to_json(f"{output_dir}/eval_{ts}.json")
    
    print(f"  Saved → {output_dir}/eval_{ts}.{{csv,pkl}}")
    return results_df


#%%
if __name__ == "__main__":
    tickers   = ["nvda"]
    finder_df = run_finder(tickers=tickers).head(5)
    print("Loading generation model...")
    gen_model, gen_tokenizer = load("mlx-community/Llama-3.2-3B-Instruct-4bit")

    print("Loading embedding model...")
    embed_model, embed_tokenizer = emb_load("mlx-community/embeddinggemma-300m-bf16")
    results = run_evaluation(
        finder_df=finder_df,        
        gen_model=gen_model,
        gen_tokenizer=gen_tokenizer,
        embed_model=embed_model,
        embed_tokenizer = embed_tokenizer,
        top_k=6,        
        levels=["header"],
    )
    print(results)

# %%
