import pydantic

from typing import Optional

class doc_payload(pydantic.BaseModel):
    '''
    This class insures data consistency. Not all metadata fields exist for all chunks. Therefore, they're defaulted to None
    '''
    form_type: Optional[str] = None
    company_name: Optional[str] = None
    ticker: Optional[str] = None
    fiscal_year_end: Optional[str] = None
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

#%%
import pandas as pd
from qdrant_client import QdrantClient, models
client = QdrantClient(url="http://localhost:6333", check_compatibility=True)
import uuid
from qdrant_client.models import Distance, VectorParams

try:
    client.create_collection(
        collection_name="quant",
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
    field_schema=models.PayloadSchemaType.TEXT, 
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
import pickle
from sentence_transformers import SentenceTransformer
import torch
from qdrant_client.models import PointStruct

chunk_paths = ["/Users/nadavsmacbookair/Desktop/Thesis/code/data/BESTBUY_2023_10K-Chunks.pkl", "/Users/nadavsmacbookair/Desktop/Thesis/code/data/3M_2022_10K-Chunks.pkl"]
chunks_df=[]
for path in chunk_paths:
    with open(path, "rb") as f:
        chunks_df.append(pickle.load(f))

# Load embedding model
model = SentenceTransformer("google/embeddinggemma-300M", device="mps")

print("Starting upserting loop...")
for chunk in chunks_df:
    texts = chunk['text'].tolist() if hasattr(chunk['text'], 'tolist') else chunk['text']
    metadatas = chunk['metadata'].tolist() if hasattr(chunk['metadata'], 'tolist') else chunk['metadata']

    print(f"Encoding {len(texts)} chunks...")
    embeddings = model.encode(texts, show_progress_bar=True).tolist()

    points = []
    for idx, (metadata, text) in enumerate(zip(metadatas, texts)):
        try:
            # Use doc_payload for validation and normalization (lowercase, strip)
            # This handles the "mentioned metadata fields" requirement
            validated_payload = doc_payload(**metadata).model_dump()
            # Add the text field as metadata but it won't be indexed (no payload index created for 'text')
            validated_payload["text"] = text
            payload = validated_payload
        except Exception as e:
            # Fallback to original metadata if validation fails, adding text
            print(f"Warning: Metadata validation failed for index {idx}: {e}")
            payload = metadata.copy()
            payload["text"] = text
            
        points.append(
            PointStruct(
                id=str(uuid.uuid4()), 
                vector=embeddings[idx], 
                payload=payload
            )
        )
    
    print(f"Upserting {len(points)} points to Qdrant...")
    client.upsert(
        collection_name="quant",
        wait=True,
        points=points
    )


#%% Inspect Collection
print("Inspecting collection...")
inspect_results = client.scroll("quant", limit=10)
print(f"Retrieved {len(inspect_results[0])} points sample.")

# %%
if __name__ == "__main__":
    TEN_K_NAMES = ["3M_2022_10K","BESTBUY_2023_10K"]
    # Adjust path to financebench if needed, using sample names for query subsetting
    try:
        Q_A_fb = pd.read_json("/Users/nadavsmacbookair/Documents/sec2md/financebench/data/financebench_open_source.jsonl",lines=True)
        results_list = []
        
        for name in TEN_K_NAMES:
            # SEARCH
            QA_subset = Q_A_fb[Q_A_fb['doc_name']==name]
            questions = QA_subset["question"].tolist()
            
            if not questions:
                print(f"No questions found for {name}")
                continue
                
            print(f"Embedding {len(questions)} questions for {name}...")
            question_embeddings = model.encode(questions)
            
            print(f"Searching Qdrant for {name}...")
            for i, query_vector in enumerate(question_embeddings):
                search_results = client.query_points(
                    collection_name="quant",
                    query=query_vector.tolist(),
                    limit=5,
                    with_payload=True,
                ).points
                
                for res in search_results:
                    results_list.append({
                        "doc_name": name,
                        "question": questions[i],
                        "score": res.score,
                        "retrieved_text": res.payload.get("text", ""),
                        "metadata": {k: v for k, v in res.payload.items() if k != "text"}
                    })
        
        results_df = pd.DataFrame(results_list)
        print("Search Completed. Results DataFrame created.")
        if not results_df.empty:
            print(results_df.head())
        else:
            print("No search results found.")
            
    except FileNotFoundError:
        print("FinanceBench data file not found. Skipping search example.")
    except Exception as e:
        print(f"An error occurred during search: {e}")
