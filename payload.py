#%%
import pydantic
import pandas as pd
from qdrant_client import QdrantClient
client = QdrantClient(url="http://localhost:6333", check_compatibility=True)
import uuid
from qdrant_client import models
from qdrant_client.models import Distance, VectorParams

from typing import Optional
from datetime import datetime
#%%
class doc_payload(pydantic.BaseModel):
    '''
    This class insures data consistency. Not all metadata fields exist for all chunks. Therefore, they're defaulted to None
    '''
    form_type: Optional[str] = None
    company_name: Optional[str] = None
    ticker: Optional[str] = None
    fiscal_year_end: Optional[datetime] = None
    section: Optional[str] = None
    subsection: Optional[str] = None
    item: Optional[str] = None

    # Clean and normalize strings automatically
    @pydantic.field_validator('form_type', 'company_name', 'ticker', 'section', 'subsection', 'item', mode="before")
    @classmethod
    def lowercase_string(cls, v):
        if isinstance(v, str):
            return v.strip().lower()
        return v

coll_name = "--split headers,chars --embeddings text-only"




try:
    client.create_collection(
        collection_name=coll_name,
        vectors_config=VectorParams(size=768, distance=Distance.COSINE),
        on_disk_payload=True # keeps unindexed meta out of RAM
    )
except Exception as e:
    print("exists")
#%%
client.create_payload_index(
    collection_name="quant",
    field_name="form_type",
    field_schema=models.PayloadSchemaType.KEYWORD, 
)
client.create_payload_index(
collection_name="quant",
    field_name="company_name",
    field_schema=models.PayloadSchemaType.KEYWORD, 
)
client.create_payload_index(
    collection_name="quant",
    field_name="ticker",
    field_schema=models.PayloadSchemaType.KEYWORD, 
)    
client.create_payload_index(
    collection_name="quant",
    field_name="fiscal_year_end",
    field_schema=models.PayloadSchemaType.DATETIME, 
)    
client.create_payload_index(
    collection_name="quant",
    field_name="section",
    field_schema=models.PayloadSchemaType.TEXT, 
)
client.create_payload_index(
    collection_name="quant",
    field_name="subsection",
    field_schema=models.PayloadSchemaType.TEXT, 
)
client.create_payload_index(
    collection_name="quant",
    field_name="item",
    field_schema=models.PayloadSchemaType.TEXT, 
)


#%% upserting 
# coll_name = "headers_split"
import pickle
from sentence_transformers import SentenceTransformer
import torch
from qdrant_client.models import PointStruct


chunk_paths = [f"/Users/nadavsmacbookair/Desktop/Thesis/code/data/chunks/header_and_chars_split/{file}" for file in [
    "3M_2018_10K.pkl" 
    # "ADOBE_2016_10K.pkl",
    # "PAYPAL_2022_10K.pkl", 
    # "3M_2022_10K.pkl",
    # "ADOBE_2017_10K.pkl"
    ]]

# print number of chunks totally
count=0
for path in chunk_paths:
    with open(path, "rb") as f:
        doc = (pickle.load(f))
    print(doc.shape)
    count+=doc.shape[1]
#%%

# Iterate through paths directly in the processing loop
# (No need to load all into chunks_df first)

# Load embedding model
model = SentenceTransformer("google/embeddinggemma-300M", device="mps")

# Memory Optimization: Limit torch threads and use no_grad
torch.set_num_threads(1)

def get_batches(lst, n):
    """Yield successive n-sized batches from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

#%%
print("Starting upserting loop...")
batch_size = 32  # Smaller batch size to reduce peak memory

for path in chunk_paths:
    print(f"Processing file: {path}")
    with open(path, "rb") as f:
        chunk = pickle.load(f)
    
    texts = chunk['text'].tolist() if hasattr(chunk['text'], 'tolist') else chunk['text']
    metadatas = chunk['metadata'].tolist() if hasattr(chunk['metadata'], 'tolist') else chunk['metadata']
    breakpoint()
    
    # Pre-process metadatas (date conversion)
    for i in range(len(metadatas)):
        date_ = metadatas[i]["fiscal_year_end"]
        # Handle different date formats or ensure consistency
        if isinstance(date_, str) and len(date_) >= 8:
            if not date_[-4:-2].isdigit() or int(date_[-4:-2]) < 19: # Simple heuristic for 2-digit year
                 date_ = date_[:len(date_)-2] + "20" + date_[len(date_)-2:]
            metadatas[i]["fiscal_year_end"] = datetime.strptime(date_, "%m-%d-%Y").date()

    # Process in sub-batches
    total_chunks = len(texts)
    for batch_idx, (batch_texts, batch_metadatas) in enumerate(zip(get_batches(texts, batch_size), get_batches(metadatas, batch_size))):
        print(f"  Encoding batch {batch_idx + 1} ({len(batch_texts)} chunks)...")
        
        with torch.no_grad():
            embeddings = model.encode(batch_texts, show_progress_bar=False, batch_size=batch_size).tolist()

        points = []
        for idx, (metadata, text) in enumerate(zip(batch_metadatas, batch_texts)):
            try:
                # Use doc_payload for validation and normalization
                validated_payload = doc_payload(**metadata).model_dump()
                validated_payload["text"] = text
                payload = validated_payload
            except Exception as e:
                print(f"    Warning: Metadata validation failed for batch {batch_idx}, index {idx}: {e}")
                payload = metadata.copy()
                payload["text"] = text
                
            points.append(
                PointStruct(
                    id=str(uuid.uuid4()), 
                    vector=embeddings[idx], 
                    payload=payload
                )
            )
        
        print(f"  Upserting {len(points)} points to Qdrant...")
        client.upsert(
            collection_name=coll_name,
            wait=True,
            points=points
        )
        
        # Clear MPS cache if using Apple Silicon
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()

    # Clear memory after each file
    del chunk
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()


#%% Inspect Collection
print("Inspecting collection...")
inspect_results = client.scroll(coll_name, limit=1000000)
print(f"Retrieved {len(inspect_results[0])} points sample.")

# %%
# if __name__ == "__main__":
#     TEN_K_NAMES = ["3M_2022_10K","BESTBUY_2023_10K"]
#     # Adjust path to financebench if needed, using sample names for query subsetting
#     try:
#         Q_A_fb = pd.read_json("/Users/nadavsmacbookair/Documents/sec2md/financebench/data/financebench_open_source.jsonl",lines=True)
#         results_list = []
        
#         for name in TEN_K_NAMES:
#             # SEARCH
#             QA_subset = Q_A_fb[Q_A_fb['doc_name']==name]
#             questions = QA_subset["question"].tolist()
            
#             if not questions:
#                 print(f"No questions found for {name}")
#                 continue
                
#             print(f"Embedding {len(questions)} questions for {name}...")
#             question_embeddings = model.encode(questions)
            
#             print(f"Searching Qdrant for {name}...")
#             for i, query_vector in enumerate(question_embeddings):
#                 search_results = client.query_points(
#                     collection_name="quant",
#                     query=query_vector.tolist(),
#                     limit=5,
#                     with_payload=True,
#                 ).points
                
#                 for res in search_results:
#                     results_list.append({
#                         "doc_name": name,
#                         "question": questions[i],
#                         "score": res.score,
#                         "retrieved_text": res.payload.get("text", ""),
#                         "metadata": {k: v for k, v in res.payload.items() if k != "text"}
#                     })
        
#         results_df = pd.DataFrame(results_list)
#         print("Search Completed. Results DataFrame created.")
#         if not results_df.empty:
#             print(results_df.head())
#         else:
#             print("No search results found.")
            
#     except FileNotFoundError:
#         print("FinanceBench data file not found. Skipping search example.")
#     except Exception as e:
#         print(f"An error occurred during search: {e}")


