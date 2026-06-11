#%% 
import pandas as pd
from qdrant_client import QdrantClient, models
client = QdrantClient(url="http://localhost:6333", check_compatibility=True)
import uuid
from qdrant_client.models import Distance, VectorParams
from search import search_with_payload, coll_name

#%% Payload Extraction 
'''
implement an LLM that given a query, extracts the correct payloads:form_type,ticker,fiscal_year_end
The LLM will call search_with_payload.
'''
from mlx_lm import load, generate
from sentence_transformers import SentenceTransformer
import json
import re
from datetime import datetime
from transformers import AutoTokenizer, AutoModel

# Load LLM and Embedding Model
model, tokenizer, *extra = load("mlx-community/Llama-3.2-3B-Instruct-4bit")
embed_model = SentenceTransformer("google/embeddinggemma-300M", device="mps")
#%% Prompt rewriting agent
SYSTEM_PROMPT = """
You are a financial analysis expert specializing in SEC 10-K filings. Your task is to transform a user's natural language request into a structured search object.

### Instructions:
1. **Identify the Company**: The user might mention a company name instead of a ticker. You MUST identify the correct stock ticker symbol in LOWERCASE (e.g., "Apple" -> "aapl", "Microsoft" -> "msft", "3M" -> "mmm").
2. **Handle Fiscal Year**: If the user mentions a year (e.g., "fiscal 2022"), extract the year as an INTEGER (e.g., 2022).
3. **optimized_prompt**: Rewrite the user's request into a high-density financial query. Use professional terminology like 'amortization', 'revenue recognition', 'liquidity risk', 'EBITDA', 'segment reporting', and 'capital expenditures' to help a vector database find the most relevant chunks of text.
4. **payload**: 
   - "form_type": Always "10-k".
   - "ticker": The stock ticker symbol in LOWERCASE.
   - "year": The fiscal year as an INTEGER.

Return ONLY a valid JSON object.
"""

def search_agent(user_query):
    """
    1. Enhances the user query for vector search using an LLM.
    2. Extracts payload filters (ticker, year, form_type).
    3. Embeds the optimized prompt.
    4. Calls search_with_payload to get results.
    """
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_query}
    ]
    
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    
    # Generate LLM response
    response = generate(model, tokenizer, prompt=prompt, verbose=False)
    response = response.lower()
    try:
        # Extract JSON from response
        json_match = re.search(r'\{.*\}', response, re.DOTALL)
        if not json_match:
            print(f"No JSON found. Raw response: {response}")
            return None
        data = json.loads(json_match.group())
        opt_retr_query = data.pop("optimized_prompt", user_query)
        
        payload_filters = data
        print(f"Optimized Retrieval Query: {opt_retr_query}")
        print(f"Payload Filters: {payload_filters}")
        
        # Embed the optimized prompt
        query_vec = embed_model.encode(opt_retr_query).tolist()
        # Call the search function
        results = search_with_payload(coll_name, query_vec, payload_must=payload_filters)
        return opt_retr_query,results

    except Exception as e:
        print(f"Error in search_agent: {e}")
        return None


def response(user_query, search_results):
    """
    Given a user query and search results, this function returns an LLM generated answer.
    Follows RAG best practices:
    1. Extracts text from retrieved points.
    2. Constructs a prompt with context and instructions.
    3. Generates a grounded response.
    """
    if not search_results or not search_results.points:
        return "No relevant information found in the database to answer your query."

    # Extract and format context from search results
    context_chunks = []
    for i, point in enumerate(search_results.points):
        text = point.payload.get("text", "No text content available.")
        ticker = point.payload.get("ticker", "n/a")
        year = point.payload.get("fiscal_year_end", "n/a")
        
        # Robust year extraction for display
        display_year = "n/a"
        try:
            if hasattr(year, 'year'):
                display_year = str(year.year)
            elif isinstance(year, str):
                display_year = year[:4]
            else:
                display_year = str(year)
        except:
            pass
            
        context_chunks.append(f"--- Source {i+1} (Ticker: {ticker.upper()}, Year: {display_year}) ---\n{text}")

    context_text = "\n\n".join(context_chunks)

    RAG_SYSTEM_PROMPT = """
You are a financial assistant expert in SEC filings. Use the provided context from 10-K filings to answer the user's question.
Guidelines:
1. Base your answer ONLY on the provided context.
2. If the context doesn't contain the answer, state that you don't have enough information.
3. Reference specific sources (e.g., Source 1, Source 2) when citing numbers or facts.
4. Keep the response professional and structured.
"""

    messages = [
        {"role": "system", "content": RAG_SYSTEM_PROMPT},
        {"role": "user", "content": f"Context:\n{context_text}\n\nQuestion: {user_query}"}
    ]

    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    # Generate response
    generated_text = generate(model, tokenizer, prompt=prompt, verbose=False)
    
    return generated_text.strip(), context_chunks


# %%
if __name__ == "__main__":
    from trulens_eval import evaluate_trulens_response
    
    # Path to the dataset
    DATASET_PATH = "/Users/nadavsmacbookair/Desktop/Thesis/Sources/data/financebench_open_source.jsonl"
    
    # 1. Load and filter dataset
    filtered_pairs = []
    try:
        with open(DATASET_PATH, "r") as f:
            for line in f:
                record = json.loads(line)
                company = record.get("company", "").lower()
                doc_name = record.get("doc_name", "").lower()
                
                # Check for 3M and 2022
                if ("3m" in company or "mmm" in company) and ("2022" in doc_name):
                    filtered_pairs.append({
                        "question": record["question"],
                        "reference_answer": record["answer"]
                    })
    except Exception as e:
        print(f"Error loading dataset: {e}")
        
    print(f"Found {len(filtered_pairs)} relevant question-answer pairs for 3M in 2022.")
    
    # 2. Iterate and run the pipeline
    for i, pair in enumerate(filtered_pairs, 1):
        query = pair["question"]
        ref_answer = pair["reference_answer"]
        print(f"\n{'='*50}")
        print(f"Query {i}/ out of {len(filtered_pairs)}: {query}")
        print(f"Reference Answer: {ref_answer}")
        print(f"{'='*50}")
        
        # Search and Retrieve
        import time
        try:
            opt_ret_query, results = search_agent(query)
            
            if results and results.points:
                # Generate Answer
                answer, chunk_sources = response(query, results) #use original query, NOT retrieval query
                print("\n--- LLM Answer ---")
                print(answer)
                
                # Evaluate with TRULENS
                print("\n--- Running TRULENS Evaluation ---")
                # Add a sleep to prevent connection overload on the local LLM proxy
                time.sleep(20)
                scores = evaluate_trulens_response(query, answer, results)
                print("\n--- trulens Scores [0-1] ---")
                for metric, score in scores.items():
                    if metric == "error":
                        print(f"Error: {score}")
                        continue
                    if metric not in ["question", "answer", "contexts"]: 
                        if isinstance(score, dict):
                            val = list(score.values())[0] if score else 0
                            numeric_score = float(score.get("score", val))
                        else:
                            numeric_score = float(score)

                        print(f"{metric.capitalize()}: {numeric_score:.4f}")
            else:
                print("Search failed or returned no results.")
        except Exception as e:
            print(f"Pipeline failed for query {i}: {e}")

# %%
