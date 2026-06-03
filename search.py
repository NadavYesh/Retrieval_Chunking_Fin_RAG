#%% import pandas as pd
from qdrant_client import QdrantClient, models
client = QdrantClient(url="http://localhost:6333", check_compatibility=True)
import uuid
from qdrant_client.models import Distance, VectorParams
from datetime import datetime, timezone

#%% Scroll: basic search based on payload items. 
scroll_results = client.scroll(
    collection_name="quant",
    scroll_filter=models.Filter(
        must=[
            models.FieldCondition(key="form_type", match=models.MatchValue(value='10-k')),
            models.FieldCondition(key="company_name", match=models.MatchValue(value='3m company')),
            models.FieldCondition(
                key="fiscal_year_end", 
                range=models.DatetimeRange(
                    gte="2022-01-01T00:00:00Z",
                    lte="2022-12-31T23:59:59Z"
                )
            ),
        ]
    ),
    limit=10,
    with_payload=True,
    with_vectors=False,
)

#%% Apply vector search on top
client.query_points(
    collection_name="quant",
    query=[0.2, 0.1, 0.9, 0.7]*192, # RANDOM VECTOR
    query_filter=models.Filter(
        must=[
            models.FieldCondition(key="form_type", match=models.MatchValue(value='10-k')),
            
        ]
    ),
    search_params=models.SearchParams(hnsw_ef=128, exact=False),
    limit=3,
)

#%%

def search_with_payload(query_vec, payload_must=None, payload_must_not=None, payload_should=None):
    """
    Performs a vector search in the 'quant' collection with optional payload filtering.

    Args:
        query_vec (list[float]): The vector to search for.
        payload_must (dict | list[tuple], optional): Conditions that must be satisfied. Defaults to None.
        payload_must_not (dict | list[tuple], optional): Conditions that must not be satisfied. Defaults to None.
        payload_should (dict | list[tuple], optional): Conditions that should be satisfied (boosts score). Defaults to None.

    Returns:
        QueryResponse: The search results from Qdrant.
    """
    must = []
    if payload_must:
        items = payload_must.items() if isinstance(payload_must, dict) else payload_must
        for k, v in items:
            if k in ["fiscal_year_end", "year"]: # filter date only on year
                # Robustly extract year from int, str (e.g. "2022" or "2022-12-31"), or datetime object
                try:
                    if hasattr(v, 'year'):
                        year = v.year
                    elif isinstance(v, str):
                        year = int(v[:4])
                    else:
                        year = int(v)
                    
                    must.append(
                        models.FieldCondition(
                            key="fiscal_year_end",
                            range=models.DatetimeRange(
                                gte=f"{year}-01-01T00:00:00Z",
                                lte=f"{year}-12-31T23:59:59Z"
                            )
                        )
                    )
                except (ValueError, TypeError) as e:
                    print(f"Warning: Could not parse year from {k}={v}: {e}")
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
        collection_name="quant",
        query=query_vec,
        query_filter=models.Filter(
            must=must if must else None,
            must_not=must_not if must_not else None,
            should=should if should else None
        ),
        search_params=models.SearchParams(hnsw_ef=128, exact=False),
        limit=5,
        with_payload=True,)
    return results_
