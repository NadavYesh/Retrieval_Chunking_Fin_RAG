import json
import re
from typing import TypedDict, List, Optional, Dict, Any
from langgraph.graph import StateGraph, END
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import MemorySaver

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
    model = configurable.get("model")
    tokenizer = configurable.get("tokenizer")
    
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
        
        return {
            "optimized_query": opt_query,
            "ticker": data.get("ticker"),
            "year": data.get("year"),
            "filters": data,
            "retries": state.get("retries", 0)
        }
    except Exception as e:
        return {"error": str(e), "optimized_query": state["user_query"]}

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
    
    # Embed the optimized prompt
    query_vec = embed_model.encode(state["optimized_query"]).tolist()
    
    # Use existing search_with_payload logic
    results = search_with_payload(
        coll_name=coll_name,
        query_vec=query_vec,
        payload_must=state["filters"]
    )
    for res in result:
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
    context = " ".join([p.payload.get("text", "") for p in state["retrieved_results"].points])
    grading_prompt = f"System: Is this context relevant to the question: '{state['user_query']}'? Respond with YES or NO.\nContext: {context[:500]}"
    
    print(f"--- GRADER: Evaluating relevance for query: {state['user_query'][:60]}... ---")
    # In a real app, use a specific grading prompt template
    response = generate(model, tokenizer, prompt=grading_prompt, verbose=False).upper()
    print(f"--- GRADER: Raw LLM response: {response.strip()} ---")
    
    if "YES" in response:
        print("--- GRADER: Document deemed RELEVANT ---")
        return {"error": None}
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
    return {"optimized_query": f"detailed financial breakdown of {state['user_query']}"}

def generator_node(state: GraphState, config: RunnableConfig):
    """
    Node 3: Generates the final answer using the retrieved context.
    """
    if not state.get("retrieved_results") or not state["retrieved_results"].points:
        return {"answer": "No relevant financial data found for the specified parameters."}

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

def create_financial_rag_graph():
    """
    Builds the state machine.
    """
    workflow = StateGraph(GraphState)

    # Add Nodes
    workflow.add_node("analyze_query", query_analyzer_node)
    workflow.add_node("retrieve", retriever_node)
    workflow.add_node("grade_documents", document_grader_node)
    workflow.add_node("rewrite_query", rewrite_query_node)
    workflow.add_node("generate", generator_node)

    # Define Edges (The Flow)
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

    # We add interrupt_before to pause the graph before retrieval
    # This allows a human to inspect 'ticker' and 'year' in the state.
    return workflow.compile(
        checkpointer=MemorySaver(),
        interrupt_before=["retrieve"])

# --- Example Usage ---
if __name__ == "__main__":
    from mlx_lm import load
    from sentence_transformers import SentenceTransformer
    from extract_financebench_data import get_fb_points
    
    # Setup (simplified version of run_rag_pipeline.py config)
    print("Loading models...")
    model, tokenizer, *_ = load("mlx-community/Llama-3.2-3B-Instruct-4bit")
    embed_model = SentenceTransformer("google/embeddinggemma-300M", device="mps")
    
    # 1. Fetch the same data used in run_rag_pipeline.py
    company = "bestbuy"
    year = 2023
    records = get_fb_points(company, year)
    
    if not records:
        print(f"No records found for {company} {year}")
        exit()

    # Take the first question to demonstrate the pipeline
    test_record = records[0]
    user_query = test_record["question"]
    ref_answer = test_record["answer"]

    print(f"\nTarget Question: {user_query}")
    print(f"Reference Answer: {ref_answer}")

    app = create_financial_rag_graph()
    
    pipeline_config = {
        "model": model, 
        "tokenizer": tokenizer, 
        "embed_model": embed_model,
        "coll_name": "--split headers,chars --embeddings text,meta"
    }
    
    initial_state = {"user_query": user_query, "retries": 0}
    
    # Thread ID allows the checkpointer to save this specific conversation
    thread_config = {"configurable": {**pipeline_config, "thread_id": "eval_1"}}
    
    print("\n--- Phase 1: Query Analysis ---")
    # Run the graph until the first breakpoint
    app.invoke(initial_state, config=thread_config)
    
    # Fetch the state at the breakpoint
    snapshot = app.get_state(thread_config)
    extracted_ticker = snapshot.values.get("ticker")
    extracted_year = snapshot.values.get("year")

    print(f"\n[HUMAN CHECK] Extracted Ticker: {extracted_ticker}, Year: {extracted_year}")
    confirm = input("Does this look correct? (y/n): ")

    if confirm.lower() != 'y':
        print("Aborting pipeline.")
        exit()

    print("\n--- Phase 2: Retrieval and Generation ---")
    # Passing None as the first argument tells LangGraph to resume from the checkpoint
    final_output = app.invoke(None, config=thread_config)

    print("\n--- FINAL ANSWER ---")
    print(final_output.get("answer"))
    
    print("\n--- METADATA EXTRACTED ---")
    print(f"Ticker: {final_output.get('ticker')}, Year: {final_output.get('year')}")