#%% 
import pandas as pd
from qdrant_client import QdrantClient, models
client = QdrantClient(url="http://localhost:6333", check_compatibility=True)
import uuid
from qdrant_client.models import Distance, VectorParams
from search import search_with_payload

#%%  
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
#%%
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
        optimized_query = data.pop("optimized_prompt", user_query)
        
        payload_filters = data
        print(f"Optimized Prompt: {optimized_query}")
        print(f"Payload Filters: {payload_filters}")
        
        # Embed the optimized prompt
        query_vec = embed_model.encode(optimized_query).tolist()
        # Call the search function
        results = search_with_payload(query_vec, payload_must=payload_filters)
        return optimized_query,results

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
    from trulens_self import evaluate_trulens_response
    
    query = "what was the 3m (mmm) revenue for the fiscal year ending 2022?"
    print(f"\nUser Query: {query}")
    
    # 1. Search and Retrieve
    opt_query,results = search_agent(query)
    
    if results:
        # 2. Generate Answer
        # Note: We use the oORIGINAL user query for evaluation 
        # For evaluation, RAGAS usually takes the original user query.
        answer, chunk_sources = response(query, results)
        print("\n=== LLM Answer ===")
        print(answer)
        
        # 3. Evaluate with TRULENS
        print("\n=== Running TRULENS Evaluation ===")
        # try:
            
        scores = evaluate_trulens_response(query, answer, results)
        print("\n=== trulens Scores ===")
        for metric, score in scores.items():
            if metric == "error":
                print(f"Error: {score}")
                continue
            if metric not in ["question", "answer", "contexts"]:
                if isinstance(score, dict):
                    numeric_score = float(score.get("score", list(score.values())[0]))
                else:
                    numeric_score = float(score)

                print(f"{metric.capitalize()}: {numeric_score:.4f}")
        # except Exception as e:
        #     print(f"error")
    else:
        print("Search failed or returned no results.")

# %%
