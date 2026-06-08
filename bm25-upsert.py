#%%
#from payload import doc_payload # זה מריץ את כל הקובץ!!!!!!!
from classes import doc_payload
import pydantic
import pandas as pd
from qdrant_client import QdrantClient
client = QdrantClient(url="http://localhost:6333", check_compatibility=True, cloud_inference=False)
import uuid
from qdrant_client import models
from qdrant_client.models import PointStruct
from qdrant_client.models import Distance, VectorParams

from typing import Optional
from datetime import datetime

import pickle
from sentence_transformers import SentenceTransformer
import torch

#%% updating a point with bm25 sparse vectors ==> impossible ==> create new collection
collection_name = "--split headers --bm25 text"
try:
    client.create_collection(collection_name,
                         sparse_vectors_config={
                            "text": models.SparseVectorParams(modifier=models.Modifier.IDF)
                        },
                        on_disk_payload=True
    )
except:
    print("exists")
embed_meta = False

#%%
chunk_paths = [f"/Users/nadavsmacbookair/Desktop/Thesis/code/data/chunks/header_and_chars_split/{file}" for file in [
    "3M_2018_10K.pkl",
    "ADOBE_2016_10K.pkl",
    "PAYPAL_2022_10K.pkl", 
    "3M_2022_10K.pkl",
    "ADOBE_2017_10K.pkl"
    ]]

# print number of chunks totally
count=0
for path in chunk_paths:
    with open(path, "rb") as f:
        doc = (pickle.load(f))
    print(doc.shape)
    count+=doc.shape[1]

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

    
    # Pre-process metadatas (date conversion)
    for i in range(len(metadatas)):
        date_ = metadatas[i]["fiscal_year_end"]
        # Handle different date formats or ensure consistency
        if isinstance(date_, str) and len(date_) >= 8:
            if not date_[-4:-2].isdigit() or int(date_[-4:-2]) < 19: # Simple heuristic for 2-digit year
                 date_ = date_[:len(date_)-2] + "20" + date_[len(date_)-2:]
            metadatas[i]["fiscal_year_end"] = datetime.strptime(date_, "%m-%d-%Y").date()
    
    if embed_meta: # concatenate the metadata with the text
        combined = [str(metas)[1:len(str(metas))-1] + ", 'text': " + txt for (metas,txt) in zip(metadatas,texts)]
    
    # Process in sub-batches
    total_chunks = len(texts)
    for batch_idx, (batch_texts, batch_metadatas) in enumerate(zip(get_batches(combined if embed_meta else texts, batch_size), get_batches(metadatas, batch_size))):
        print(f"  Encoding batch {batch_idx + 1} ({len(batch_texts)} chunks)...")
        points = []
        for idx, (metadata, text) in enumerate(zip(batch_metadatas, batch_texts)):
            try:
                # Use doc_payload for validation and normalization
                validated_payload = doc_payload(**metadata).model_dump()
                validated_payload["text"] = text[text.index("'text':")+8:] if embed_meta else text
                payload = validated_payload
            except Exception as e:
                print(f"    Warning: Metadata validation failed for batch {batch_idx}, index {idx}: {e}")
                payload = metadata.copy()
                payload["text"] = text
                
            points.append(
                PointStruct(
                    id=str(uuid.uuid4()), 
                    vector={ # upload a sparse lexical vector. 
                        "text": models.Document(
                            text=text,
                            model="qdrant/bm25",
                        )
            },
                    payload=payload
                )
            )
        
        print(f"  Upserting {len(points)} points to Qdrant...")
        client.upsert(
            collection_name=collection_name,
            wait=True,
            points=points
        )



# %%
