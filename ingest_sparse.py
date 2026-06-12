import uuid
import pickle
import pandas as pd
from datetime import datetime
from qdrant_client import models
from qdrant_client.models import PointStruct

from models import doc_payload
from database import get_qdrant_client
from utils import get_batches

client = get_qdrant_client()

# Configuration
COLLECTION_NAME = "--split headers --bm25 text"
EMBED_META = False
CHUNK_PATHS = [f"/Users/nadavsmacbookair/Desktop/Thesis/code/data/chunks/header_and_chars_split/{file}" for file in [
    "3M_2018_10K.pkl",
    "ADOBE_2016_10K.pkl",
    "PAYPAL_2022_10K.pkl", 
    "3M_2022_10K.pkl",
    "ADOBE_2017_10K.pkl"
]]

def init_collection():
    try:
        client.create_collection(
            collection_name=COLLECTION_NAME,
            sparse_vectors_config={
                "text": models.SparseVectorParams(modifier=models.Modifier.IDF)
            },
            on_disk_payload=True
        )
        print(f"Collection {COLLECTION_NAME} created.")
    except Exception:
        print(f"Collection {COLLECTION_NAME} already exists.")

def upsert_sparse_data():
    batch_size = 32
    print("Starting sparse upserting loop...")

    for path in CHUNK_PATHS:
        print(f"Processing file: {path}")
        with open(path, "rb") as f:
            chunk = pickle.load(f)
        
        texts = chunk['text'].tolist() if hasattr(chunk['text'], 'tolist') else chunk['text']
        metadatas = chunk['metadata'].tolist() if hasattr(chunk['metadata'], 'tolist') else chunk['metadata']

        # Pre-process metadatas
        for i in range(len(metadatas)):
            date_ = metadatas[i]["fiscal_year_end"]
            if isinstance(date_, str) and len(date_) >= 8:
                if not date_[-4:-2].isdigit() or int(date_[-4:-2]) < 19:
                     date_ = date_[:len(date_)-2] + "20" + date_[len(date_)-2:]
                metadatas[i]["fiscal_year_end"] = datetime.strptime(date_, "%m-%d-%Y").date()
        
        if EMBED_META:
            combined = [str(metas)[1:len(str(metas))-1] + ", 'text': " + txt for (metas,txt) in zip(metadatas,texts)]
        else:
            combined = texts
        
        for batch_idx, (batch_texts, batch_metadatas) in enumerate(zip(get_batches(combined, batch_size), get_batches(metadatas, batch_size))):
            print(f"  Processing batch {batch_idx + 1}...")
            points = []
            for idx, (metadata, text) in enumerate(zip(batch_metadatas, batch_texts)):
                try:
                    validated_payload = doc_payload(**metadata).model_dump()
                    validated_payload["text"] = text[text.index("'text':")+8:] if EMBED_META else text
                    payload = validated_payload
                except Exception as e:
                    print(f"    Warning: Metadata validation failed: {e}")
                    payload = metadata.copy()
                    payload["text"] = text
                    
                points.append(
                    PointStruct(
                        id=str(uuid.uuid4()), 
                        vector={
                            "text": models.Document(
                                text=text,
                                model="qdrant/bm25",
                            )
                        },
                        payload=payload
                    )
                )
            
            client.upsert(collection_name=COLLECTION_NAME, wait=True, points=points)

if __name__ == "__main__":
    # To run this script, call the functions explicitly:
    # init_collection()
    # upsert_sparse_data()
    print("Execution skipped. Call functions explicitly to run ingestion.")
