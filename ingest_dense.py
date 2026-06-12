import uuid
import pickle
import torch
import pandas as pd
from datetime import datetime
from sentence_transformers import SentenceTransformer
from qdrant_client import models
from qdrant_client.models import PointStruct, VectorParams, Distance

from models import doc_payload
from database import get_qdrant_client
from utils import get_batches

client = get_qdrant_client()

# Configuration
COLL_NAME = "--split headers,chars --embeddings text,meta"
EMBED_META = True
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
            collection_name=COLL_NAME,
            vectors_config=VectorParams(size=768, distance=Distance.COSINE),
            on_disk_payload=True
        )
        print(f"Collection {COLL_NAME} created.")
    except Exception:
        print(f"Collection {COLL_NAME} already exists.")

    # Create indices
    fields = ["form_type", "company_name", "ticker"]
    for field in fields:
        client.create_payload_index(COLL_NAME, field, models.PayloadSchemaType.KEYWORD)
    
    client.create_payload_index(COLL_NAME, "fiscal_year_end", models.PayloadSchemaType.DATETIME)
    
    for field in ["section", "subsection", "item"]:
        client.create_payload_index(COLL_NAME, field, models.PayloadSchemaType.TEXT)

    client.create_payload_index(
        collection_name=COLL_NAME,
        field_name="text",
        field_schema=models.TextIndexParams(
            type=models.TextIndexType.TEXT,
            tokenizer=models.TokenizerType.WORD,
            lowercase=True,
            phrase_matching=True,
        ),
    )

def upsert_data():
    model = SentenceTransformer("google/embeddinggemma-300M", device="mps")
    torch.set_num_threads(1)
    
    batch_size = 32
    print("Starting upserting loop...")

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
            print(f"  Encoding batch {batch_idx + 1}...")
            
            with torch.no_grad():
                embeddings = model.encode(batch_texts, show_progress_bar=False, batch_size=batch_size).tolist()
            
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
                        vector=embeddings[idx], 
                        payload=payload
                    )
                )
            
            client.upsert(collection_name=COLL_NAME, wait=True, points=points)
            
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()

if __name__ == "__main__":
    # To run this script, call the functions explicitly:
    # init_collection()
    # upsert_data()
    print("Execution skipped. Call functions explicitly to run ingestion.")
