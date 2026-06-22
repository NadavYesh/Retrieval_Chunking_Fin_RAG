import json
import re
from typing import TypedDict, List, Optional, Dict, Any
from langgraph.graph import StateGraph, END
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import MemorySaver
from FinDER import run_finder


# Import existing logic from your search engine
from search_engine import search_with_payload, generate_llm_answer, SYSTEM_PROMPT
from mlx_lm import generate

class GraphState(TypedDict):
    """
    Represents the state of the RAG pipeline.
    This allows for high explainability as we can inspect the state at any node.
    """
    user_query: str
    optimized_query: Optional[str]
    ticker: Optional[str]
    year: Optional[int]
    filters: Optional[Dict[str, Any]]
    retrieved_results: Optional[Any]
    answer: Optional[str]
    sources: List[str]
    error: Optional[str]
    retries: int

def query_analyzer_node(state: GraphState, config: RunnableConfig):
    """
    Node 1: Analyzes the natural language query to extract metadata (ticker, year)
    and produces a high-density financial search prompt.
    """    
    configurable = config.get("configurable", {})
    model = configurable.get("eval_model")
    tokenizer = configurable.get("eval_tokenizer")

    print(f"\n--- NODE: query_analyzer_node ---")
    print(f"Goal: Extracting financial metadata and optimizing the prompt for vector search.")
    
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": state["user_query"]}
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
    response_text = generate(model, tokenizer, prompt=prompt, verbose=False).lower()
    
    try:
        json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
        if not json_match:
            return {"error": "Failed to parse query analysis JSON", "optimized_query": state["user_query"]}
            
        data = json.loads(json_match.group())
        opt_query = data.pop("optimized_prompt", state["user_query"])
        
        # Only allow specific keys for filtering to avoid Qdrant JSON path errors (e.g. spaces in keys)
        allowed_filters = ["ticker", "year", "form_type"]
        filters = {k: v for k, v in data.items() if k in allowed_filters}
        
        print(f"Extracted Ticker: {filters.get('ticker')} | Year: {filters.get('year')}")
        print(f"Optimized Query: {opt_query[:100]}...")

        return {
            "optimized_query": opt_query,
            "ticker": filters.get("ticker"),
            "year": filters.get("year"),
            "filters": filters,
            "retries": state.get("retries", 0)
        }
    except Exception as e:
        return {"error": str(e), "optimized_query": state["user_query"], "filters": {}}

def retriever_node(state: GraphState, config: RunnableConfig):
    """
    Node 2: Executes the search against Qdrant using the extracted filters
    and the optimized query.
    """
    if state.get("error") and not state.get("optimized_query"):
        return {"error": "Skipping retrieval due to prior error"}

    configurable = config.get("configurable", {})
    embed_model = configurable.get("embed_model")
    coll_name = configurable.get("coll_name")
    
    print(f"\n--- NODE: retriever_node ---")
    print(f"Goal: Fetching top-5 relevant chunks from Qdrant collection: {coll_name}")

    # Embed the optimized prompt
    query_vec = embed_model.encode(state["optimized_query"],
                                   prompt_name="Retrieval-query").tolist()

    # Use existing search_with_payload logic
    results = search_with_payload(
        coll_name=coll_name,
        query_vec=query_vec,
        payload_must=state["filters"]
    )
    
    # results is typically a QueryResponse object; iterate over its points
    if hasattr(results, "points"):
        print(f"Retrieved {len(results.points)} points from the database.")
        for res in results.points:
            print(res)

    return {"retrieved_results": results}

def document_grader_node(state: GraphState, config: RunnableConfig):
    """
    Node 2.5: Determines if the retrieved documents are relevant to the query.
    This prevents the LLM from hallucinating based on irrelevant context.
    """
    if not state.get("retrieved_results") or not state["retrieved_results"].points:
        print("--- GRADER: No results found in search results ---")
        return {"error": "no_results"}

    configurable = config.get("configurable", {})
    model = configurable.get("model")
    tokenizer = configurable.get("tokenizer")

    # Simple relevance check logic
    context_snippet = " ".join([p.payload.get("text", "") for p in state["retrieved_results"].points])[:1000]
    
    messages = [
        {"role": "system", "content": "You are a grader evaluating the relevance of a retrieved document to a user question. Respond ONLY with 'YES' if the document is relevant, or 'NO' if it is not."},
        {"role": "user", "content": f"Question: {state['user_query']}\n\nDocument Context: {context_snippet}"}
    ]
    grading_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    print(f"--- GRADER: Evaluating relevance for query: {state['user_query'][:60]}... ---")
    response = generate(model, tokenizer, prompt=grading_prompt, verbose=False).upper()
    print(f"--- GRADER: Raw LLM response: {response.strip()} ---")
    
    if "YES" in response:
        print("--- GRADER: Document deemed RELEVANT ---")
        return {"error": None} # Successfully cleared error
    else:
        print("--- GRADER: Document deemed NOT RELEVANT ---")
        return {"error": "irrelevant_context", "retries": state.get("retries", 0) + 1}

def decide_to_generate(state: GraphState):
    """
    Conditional Edge: Decides whether to generate an answer or retry/stop.
    """
    if state.get("error") == "irrelevant_context" and state.get("retries", 0) < 2:
        print(f"--- RELEVANCE CHECK FAILED (Retry {state['retries']}) ---")
        return "rewrite_query"
    elif state.get("error"):
        return END
    return "generate"

def rewrite_query_node(state: GraphState, config: RunnableConfig):
    """Node to slightly modify the search query if the first attempt failed."""
    return {
        "optimized_query": f"detailed financial breakdown and SEC filing data for {state['user_query']}",
        "error": None # Clear the error so the next retrieval node isn't skipped
    }

def generator_node(state: GraphState, config: RunnableConfig):
    """
    Node 3: Generates the final answer using the retrieved context.
    """
    if not state.get("retrieved_results") or not state["retrieved_results"].points:
        return {"answer": "No relevant financial data found for the specified parameters."}

    print(f"\n--- NODE: generator_node ---")
    print(f"Goal: Synthesizing a final answer using the retrieved context.")

    configurable = config.get("configurable", {})
    model = configurable.get("model")
    tokenizer = configurable.get("tokenizer")
    
    # Reuse the existing generation logic
    answer, chunk_sources = generate_llm_answer(
        state["user_query"], 
        state["retrieved_results"], 
        model, 
        tokenizer
    )
    
    return {"answer": answer, "sources": chunk_sources}


# =====================================================================
def create_advanced_financial_rag_graph():
    """
    Builds the full state machine including grading, rewriting, and generation.
    """
    workflow = StateGraph(GraphState)
    workflow.add_node("analyze_query", query_analyzer_node)
    workflow.add_node("retrieve", retriever_node)
    workflow.add_node("grade_documents", document_grader_node)
    workflow.add_node("rewrite_query", rewrite_query_node)
    workflow.add_node("generate", generator_node)

    workflow.set_entry_point("analyze_query")
    workflow.add_edge("analyze_query", "retrieve")
    workflow.add_edge("retrieve", "grade_documents")
    workflow.add_conditional_edges(
        "grade_documents",
        decide_to_generate,
        {"rewrite_query": "rewrite_query", "generate": "generate", END: END}
    )
    workflow.add_edge("rewrite_query", "retrieve")
    workflow.add_edge("generate", END)
    return workflow.compile(checkpointer=MemorySaver())

def create_simple_financial_rag_graph():
    """
    Builds a streamlined graph: Start -> Analyze -> Retrieve -> End.
    Useful for evaluating retrieval performance.
    """
    workflow = StateGraph(GraphState)
    workflow.add_node("analyze_query", query_analyzer_node)
    workflow.add_node("retrieve", retriever_node)

    workflow.set_entry_point("analyze_query")
    workflow.add_edge("analyze_query", "retrieve")
    workflow.add_edge("retrieve", END)
    return workflow.compile(checkpointer=MemorySaver())

# --- Example Usage ---
if __name__ == "__main__":
    from mlx_lm import load
    from sentence_transformers import SentenceTransformer, models
    
    print("Loading models...")
    model, tokenizer, *_ = load("mlx-community/Llama-3.2-3B-Instruct-4bit")
    eval_model, eval_tokenizer, *__ = load("mlx-community/Qwen2.5-7B-Instruct-4bit")
    
    word_embedding_model = models.Transformer("mlx-community/embeddinggemma-300m-bf16")
    pooling_model = models.Pooling(
        word_embedding_model.get_word_embedding_dimension(), # pass the output parameters.
        pooling_mode_mean_tokens=True
    )

    embed_model = SentenceTransformer(modules=[word_embedding_model, pooling_model])


    #%%
    # 1. Fetch data from FinDER and take first 10 points
    finder_df = run_finder()
    #%%
    if finder_df.empty:
        print("No records found in FinDER.")
        exit()
    
    # choose size of test data
    test_data = finder_df.head(5)

    # 2. Use the simplified graph
    # app = create_simple_financial_rag_graph()
    app = create_advanced_financial_rag_graph()

    pipeline_config = {
        "model": model, 
        "tokenizer": tokenizer, 
        "eval_model": eval_model,
        "eval_tokenizer": eval_tokenizer,
        "embed_model": embed_model,
        "coll_name": "--embedding embeddinggemma-300M --chunking-split headers"
    }

    # 3. Execution loop:
    for i, (idx, row) in enumerate(test_data.iterrows()): #idx is the row id from the df
        user_query = row["text"]
        print(f"\n{'='*50}")
        print(f"STARTING EVALUATION: FinDER Query {i+1}") #just a print
        print(f"User Request: {user_query}")
        
        initial_state = {"user_query": user_query, "retries": 0}
        thread_config = {"configurable": {**pipeline_config, "thread_id": f"finder_eval_{i}"}}

        # reak pipeline
        final_output = app.invoke(initial_state, config=thread_config)

        print("\n--- FINAL EVALUATION SUMMARY ---")
        print("Below are the top snippets used for this retrieval:")
        results = final_output.get("retrieved_results")
        if results and results.points:
            for p in results.points:
                print(f"ID: {p.id} | Score: {p.score:.4f}")
                print(f"Metadata: Ticker={p.payload.get('ticker')}, Year={p.payload.get('fiscal_year_end')}")
                print(f"Snippet: {p.payload.get('text', '')[:250]}...")
                print("-" * 15)
        else:
            print("No relevant points retrieved for this query.")
