#%% import pandas as pd
from qdrant_client import QdrantClient, models
from qdrant_client.models import PointStruct
client = QdrantClient(url="http://localhost:6333", check_compatibility=True)
import uuid
from qdrant_client.models import Distance, VectorParams
from datetime import datetime, timezone

#%% Scroll: basic search based on payload items. 
coll_name = "--split headers --embeddings text,meta"

client.scroll(coll_name,with_payload=True,
    scroll_filter=models.Filter(must=[
        models.FieldCondition(key="company_name", match=models.MatchValue(value='3m company')),
        models.FieldCondition(key="form_type", match=models.MatchValue(value='10-k')),
        models.FieldCondition(key="text",match = models.MatchValue(value='to income taxes, the related deduction from taxes payable is based on the')),
    ]))

#%% the differences between filtering and searchin. 
client.query_points(
    collection_name=coll_name,
    query=[0.2, 0.1, 0.9, 0.7]*192, # RANDOM VECTOR
    # models.Filter — What to search [these are pre filters ON SEARCH]
    # Filters operate on payload fields (metadata). They narrow the candidate set before or during scoring by including/excluding points based on conditions.
    query_filter=models.Filter(
        must=[
            models.FieldCondition(key="form_type", match=models.MatchValue(value='10-k')),
            
        ]
    ),
    # These configures THE WAY in which the NNSH algorithm performs search, given the available points. 
    search_params=models.SearchParams(hnsw_ef=128, 
                                      # Lowering hnsw_ef speeds up the search but may reduce accuracy since fewer candidate vectors are considered. A typical range is 50–200+ depending on latency targets.
                                      exact=False # doesnt matter in python, exact =True always
                                      indexed_only = True # 
                                      ),
    limit=3,
    with_payload=True,
)

#%%

def search_with_payload(coll_name, query_vec, payload_must=None, payload_must_not=None, payload_should=None):
    """
    Performs a vector search with optional payload filtering.

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

# not useful
def get_all_chunks_for_payload(payload_must=None):
    """
    Retrieves all chunks matching the payload filter using scroll.
    """
    must = []
    if payload_must:
        items = payload_must.items() if isinstance(payload_must, dict) else payload_must
        for k, v in items:
            if k in ["fiscal_year_end", "year"]:
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
                except (ValueError, TypeError):
                    pass
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

