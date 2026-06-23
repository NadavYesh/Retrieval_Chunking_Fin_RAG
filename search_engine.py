import pandas as pd
from qdrant_client import models
from qdrant_client.models import PointStruct
import uuid
import json
import re
import time
from datetime import datetime
from mlx_lm import load, generate
from db.database import get_qdrant_client

client = get_qdrant_client()

def search_with_payload(coll_name, query_vec, payload_must=None, payload_must_not=None, payload_should=None):
    """
    Performs a vector search with optional payload filtering.
    """
    must = []
    if payload_must:
        items = payload_must.items() if isinstance(payload_must, dict) else payload_must
        for k, v in items:
            if k in ["fiscal_year_end", "year"]:
                year_vals = v if isinstance(v, list) else [v]
                year_conditions = []
                for val in year_vals:
                    try:
                        if hasattr(val, 'year'):
                            yr = val.year
                        elif isinstance(val, str):
                            yr = int(val[:4])
                        else:
                            yr = int(val)
                        
                        year_conditions.append(
                            models.FieldCondition(
                                key="fiscal_year_end",
                                range=models.DatetimeRange(
                                    gte=f"{yr}-01-01T00:00:00Z",
                                    lte=f"{yr}-12-31T23:59:59Z"
                                )
                            )
                        )
                    except (ValueError, TypeError) as e:
                        print(f"Warning: Could not parse year from {val}: {e}")
                
                if len(year_conditions) == 1:
                    must.append(year_conditions[0])
                elif len(year_conditions) > 1:
                    must.append(models.Filter(should=year_conditions))
            else:
                must.append(models.FieldCondition(key=k, match=models.MatchValue(value=v)))
    
    must_not = []    
    if payload_must_not:
        items = payload_must_not.items() if isinstance(payload_must_not, dict) else payload_must_not
        for k, v in items:
            must_not.append(models.FieldCondition(key=k, match=models.MatchValue(value=v)))
            
    should = []
    if payload_should:    
        items = payload_should.items() if isinstance(payload_should, dict) else payload_should
        for k, v in items:
            should.append(models.FieldCondition(key=k, match=models.MatchValue(value=v)))

    results_ = client.query_points(
        collection_name=coll_name,
        query=query_vec,
        limit=5,
        with_payload=True,
        query_filter=models.Filter(
            must=must if must else None,
            must_not=must_not if must_not else None,
            should=should if should else None
        ),
        search_params=models.SearchParams(hnsw_ef=128, exact=False),
        )
    return results_

def get_all_chunks_for_payload(coll_name, payload_must=None):
    """
    Retrieves all chunks matching the payload filter using scroll.
    """
    must = []
    if payload_must:
        items = payload_must.items() if isinstance(payload_must, dict) else payload_must
        for k, v in items:
            if k in ["fiscal_year_end", "year"]:
                year_vals = v if isinstance(v, list) else [v]
                year_conditions = []
                for val in year_vals:
                    try:
                        if hasattr(val, 'year'):
                            yr = val.year
                        elif isinstance(val, str):
                            yr = int(val[:4])
                        else:
                            yr = int(val)
                        
                        year_conditions.append(
                            models.FieldCondition(
                                key="fiscal_year_end",
                                range=models.DatetimeRange(
                                    gte=f"{yr}-01-01T00:00:00Z",
                                    lte=f"{yr}-12-31T23:59:59Z"
                                )
                            )
                        )
                    except (ValueError, TypeError):
                        pass
                
                if len(year_conditions) == 1:
                    must.append(year_conditions[0])
                elif len(year_conditions) > 1:
                    must.append(models.Filter(should=year_conditions))
            else:
                must.append(models.FieldCondition(key=k, match=models.MatchValue(value=v)))
    
    all_points = []
    next_page = None
    while True:
        res, next_page = client.scroll(
            collection_name=coll_name,
            scroll_filter=models.Filter(must=must) if must else None,
            limit=100,
            with_payload=True,
            with_vectors=False,
            offset=next_page
        )
        all_points.extend(res)
        if not next_page:
            break
    return all_points

# LLM Search Agent Logic
SYSTEM_PROMPT = """
You are a financial analysis expert specializing in SEC 10-K filings. Your task is to transform a user's natural language request into a structured search object.

### Instructions:
1. **Identify the Company**: The user might mention a company name instead of a ticker. You MUST identify the correct stock ticker symbol in LOWERCASE (e.g., "Apple" -> "aapl", "Microsoft" -> "msft", "3M" -> "mmm").
2. **Handle Fiscal Year**: The user could mention a year of interest. Extract fiscal years of interest. Use an array if multiple years are specified.
3. **optimized_prompt**: Rewrite the user's request into a high-density financial query. Use professional terminology like 'amortization', 'revenue recognition', 'liquidity risk', 'EBITDA', 'segment reporting', and 'capital expenditures' to help a vector database find the most relevant chunks of text.
4. **payload**: 
   - "form_type": Always "10-k".
   - "ticker": The stock ticker symbol in LOWERCASE.
   - "year": The fiscal year(s) as an INTEGER or an ARRAY of INTEGERs.

Return ONLY a valid JSON object.
"""

def search_agent(user_query, model, tokenizer, embed_model, coll_name, ENAHNCE_QUERY=True, BOTH = True):
    """
    1. Enhances the user query for vector search using an LLM. [depends on the boolean]
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
    response_text = generate(model, tokenizer, prompt=prompt, verbose=False)
    response_text = response_text.lower()
    try:
        # Extract JSON from response
        json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
        if not json_match:
            print(f"No JSON found. Raw response: {response_text}")
            return None, None
        data = json.loads(json_match.group())
        opt_retr_query = data.pop("optimized_prompt", user_query)
        
        payload_filters = data
        
        # Embed the optimized prompt
        
        query_vec_enhanced = embed_model.encode(opt_retr_query, 
                                                prompt_name="Retrieval-query").tolist()
        
        query_vec_raw = embed_model.encode(user_query,
                                           prompt_name="Retrieval-query").tolist()
        
        # Call the search function
        if ENAHNCE_QUERY and BOTH:
            results_enhanced = search_with_payload(coll_name, query_vec_enhanced, payload_must=payload_filters)
            results_raw = search_with_payload(coll_name, query_vec_raw, payload_must=payload_filters)
            return opt_retr_query, results_enhanced, results_raw
        elif ENAHNCE_QUERY:
            results_enhanced = search_with_payload(coll_name, query_vec_enhanced, payload_must=payload_filters)
            return opt_retr_query, results_enhanced, None
        else:
            results_raw = search_with_payload(coll_name, query_vec_raw, payload_must=payload_filters)
            return None,None,results_raw
            
    except Exception as e:
        print(f"Error in search_agent: {e}")
        return None, None

def generate_llm_answer(user_query, search_results, model, tokenizer):
    """
    Given a user query and search results, this function returns an LLM generated answer.
    """
    if not search_results or not search_results.points:
        return "No relevant information found in the database to answer your query.", []

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
        2. Be concise. No full sentence or paragraphs are needed.
        3. A good unswer contains numbers, percentages, and facts in a complementary fashion.
        4. If the context doesn't contain the answer, state that you don't have the right information.

    """

    messages = [
        {"role": "system", "content": RAG_SYSTEM_PROMPT},
        {"role": "user", "content": f"Context:\n{context_text}\n\n Question: {user_query}"}
    ]

    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    # Generate response
    generated_text = generate(model, tokenizer, prompt=prompt, verbose=False)
    
    return generated_text.strip(), context_chunks
