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

# Load LLM and Embedding Model
model, tokenizer, *extra = load("mlx-community/Llama-3.2-3B-Instruct-4bit")
embed_model = SentenceTransformer("google/embeddinggemma-300M", device="mps")

SYSTEM_PROMPT = """
You are a financial analysis expert specializing in SEC 10-K filings. Your task is to transform a user's natural language request into a structured search object.

### Instructions:
1. **Identify the Company**: The user might mention a company name instead of a ticker. You MUST identify the correct stock ticker symbol in LOWERCASE (e.g., "Apple" -> "aapl", "Microsoft" -> "msft", "3M" -> "mmm").
2. **Handle Fiscal Dates**: If the user mentions a year (e.g., "fiscal 2022"), output the date strictly as "DD-MM-YY" (e.g., 12-31-22).
3. **optimized_prompt**: Rewrite the user's request into a high-density financial query. Use professional terminology like 'amortization', 'revenue recognition', 'liquidity risk', 'EBITDA', 'segment reporting', and 'capital expenditures' to help a vector database find the most relevant chunks of text.
4. **payload**: 
   - "form_type": Always "10-k".
   - "ticker": The stock ticker symbol in LOWERCASE.
   - "fiscal_year_end": The fiscal year end date strictly in "DD-MM-YY" format.

Return ONLY a valid JSON object.
"""

def search_agent(user_query):
    """
    1. Enhances the user query for vector search using an LLM.
    2. Extracts payload filters (ticker, date, form_type).
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
        return results

    except Exception as e:
        print(f"Error in search_agent: {e}")
        return None


def response(user_query,search_results):
    '''
    Given a user query and search results, this function returns an LLM generated answer to the user query.
    The given chunks are handled with reducing importance. The LLM is instructed to only use relevant chunks.
    args: 
        user_query: str
        search_results: QueryResponse (The search results from Qdrant [five chunks]).
    Returns:
        response: str
    '''
    response=""

    return(response)


if __name__ == "__main__":
    # Example usage
    test_query = "What were the main risk factors for 3M in year 2022?"
    print(f"User Query: {test_query}")
    test_results = search_agent(test_query)
    
    if test_results and hasattr(test_results, 'points'):
        print(f"\nFound {len(test_results.points)} results:")
        for point in test_results.points:
            print(f"Score: {point.score:.4f}")
            print(f"Meta: {point.payload.get("ticker"), point.payload.get("company_name"),point.payload.get("fiscal_year_end")}")
            print(f"Text: {point.payload.get('text')[:200]}...")
            print("-" * 20)
    else:
        print("No results returned.")

    # print("\n==============================")
    # print("LLM generated answer\n")
    # response_text = response(test_query,test_results)
