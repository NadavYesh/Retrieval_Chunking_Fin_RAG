#%% import pandas as pd
from qdrant_client import QdrantClient, models
client = QdrantClient(url="http://localhost:6333", check_compatibility=True)
import uuid
from qdrant_client.models import Distance, VectorParams

#%% Scroll: basic search based on payload items. 
scroll_results = client.scroll(
    collection_name="quant",
    scroll_filter=models.Filter(
        must=[
            models.FieldCondition(key="form_type", match=models.MatchValue(value='10-k')),
            models.FieldCondition(key="company_name", match=models.MatchValue(value='3m company')),
            models.FieldCondition(key="fiscal_year_end", match=models.MatchValue(value='12-31-22')),

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
'''
Now, I wanna build a function that applies smart search. 
It applies hard filtering based on: form_type, ticker, fiscal_year_end,
'''

def payload_search(query_vec, payload_must=None, payload_must_not=None, payload_should=None):
    must = []
    if payload_must:
        items = payload_must.items() if isinstance(payload_must, dict) else payload_must
        for k, v in items:
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

    return client.query_points(
        collection_name="quant",
        query=query_vec,
        query_filter=models.Filter(
            must=must,
            must_not=must_not,
            should=should
        ),
        search_params=models.SearchParams(hnsw_ef=128, exact=False),
        limit=5,
    )
# %%
payload_search(query_vec=[0.2, 0.1, 0.9, 0.7]*192, payload_must={'form_type':'10-k'})
