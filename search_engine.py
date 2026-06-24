import pandas as pd
from qdrant_client import models
from qdrant_client.models import PointStruct
import uuid
import time
from datetime import datetime
from mlx_lm import load, generate
from db.database import get_qdrant_client
from prompts import RAG_SYSTEM_PROMPT, SYSTEM_PROMPT
from utils import parse_metadata_response



#%%
client = get_qdrant_client()

from qdrant_client import models

def search_with_payload(coll_name, query_vec, payload_must=None, payload_must_not=None, payload_should=None, top_k=5):
    """
    Performs a vector search with optional payload filtering.
    Ignores keys where the value is None. Supports lists for 'MatchAny' filtering.
    """
    must = []
    if payload_must:
        items = payload_must.items() if isinstance(payload_must, dict) else payload_must
        for k, v in items:
            # Skip filtering if the value is explicitly None
            if v is None:
                continue
                
            if k in ["fiscal_year_end", "year"]:
                year_vals = v if isinstance(v, list) else [v]
                year_conditions = []
                for val in year_vals:
                    if val is None:
                        continue
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
                # Handle lists with MatchAny, scalars with MatchValue
                if isinstance(v, list):
                    must.append(models.FieldCondition(key=k, match=models.MatchAny(any=v)))
                else:
                    must.append(models.FieldCondition(key=k, match=models.MatchValue(value=v)))
    
    must_not = []    
    if payload_must_not:
        items = payload_must_not.items() if isinstance(payload_must_not, dict) else payload_must_not
        for k, v in items:
            if v is None:
                continue
            if isinstance(v, list):
                must_not.append(models.FieldCondition(key=k, match=models.MatchAny(any=v)))
            else:
                must_not.append(models.FieldCondition(key=k, match=models.MatchValue(value=v)))
            
    should = []
    if payload_should:    
        items = payload_should.items() if isinstance(payload_should, dict) else payload_should
        for k, v in items:
            if v is None:
                continue
            if isinstance(v, list):
                should.append(models.FieldCondition(key=k, match=models.MatchAny(any=v)))
            else:
                should.append(models.FieldCondition(key=k, match=models.MatchValue(value=v)))

    # Only create a Filter object if there is actually something to filter
    query_filter = None
    if must or must_not or should:
        query_filter = models.Filter(
            must=must if must else None,
            must_not=must_not if must_not else None,
            should=should if should else None
        )

    results_ = client.query_points(
        collection_name=coll_name,
        query=query_vec,
        limit=top_k,
        with_payload=True,
        query_filter=query_filter,
        search_params=models.SearchParams(hnsw_ef=128, exact=False)
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
    try:
        meta           = parse_metadata_response(response_text, fallback_query=user_query)
        opt_retr_query = meta["optimized_query"]
        payload_filters = {k: meta[k] for k in ("ticker", "year", "form_type") if meta.get(k)}
        
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
