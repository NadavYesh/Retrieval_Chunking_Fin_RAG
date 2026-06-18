#%%
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
import uuid
import pickle
import torch
import pandas as pd
from datetime import datetime
from sentence_transformers import SentenceTransformer
from qdrant_client import models
from qdrant_client.models import PointStruct, VectorParams, Distance
from models import doc_payload
from db.database import get_qdrant_client
from db.utils import get_batches

client = get_qdrant_client()
######################
# AMZN_10K_2024 failed on batch 3. remove all its ids and do again

######################
# Configuration
COLL_NAME = "--embedding embeddinggemma-300M --chunking-split headers"
EMBED_META = False
import os
files = os.listdir("/Users/nadavsmacbookair/Desktop/Thesis/data/financial_corpora/chunks/indexed-at-13-06-26/headers_split")
files = [f for f in files if f.endswith(".pkl")]
paths=[os.path.join("/Users/nadavsmacbookair/Desktop/Thesis/data/financial_corpora/chunks/indexed-at-13-06-26/headers_split",f)for f in files]

#%%
CHUNK_PATHS = paths
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
    print(f"WARNING: are you absolutuley sure you want to ingest data? Make sure you are not replicating.\nthis is collection {COLL_NAME}")
    confirm = input("Type 'y' to proceed with ingestion: ")
    if confirm.lower() != 'y':
        print("Ingestion aborted.")
        return

    model = SentenceTransformer("google/embeddinggemma-300M", device="mps")
    torch.set_num_threads(1)
    
    upsert_batch = 42
    encode_batch = 6

    print("Starting upserting loop...")

    for path in CHUNK_PATHS:
        print(f"Processing file: {path}")
        with open(path, "rb") as f:
            chunk = pickle.load(f)
        
        texts = chunk['text'].tolist() if hasattr(chunk['text'], 'tolist') else chunk['text']
        metadatas = chunk['metadata'].tolist() if hasattr(chunk['metadata'], 'tolist') else chunk['metadata']
        ids = chunk['id'].tolist() if 'id' in chunk.columns else [str(uuid.uuid4()) for _ in range(len(texts))]
        raw_texts = chunk['text'].tolist() if hasattr(chunk['text'], 'tolist') else chunk['text']

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
            # Gemma models often perform better with a document prefix
            combined = [txt for txt in texts]

        batches = zip(
            get_batches(combined, upsert_batch), 
            get_batches(metadatas, upsert_batch), 
            get_batches(ids, upsert_batch),
            get_batches(raw_texts, upsert_batch)
        )
        for batch_idx, (batch_texts, batch_metadatas, batch_ids, batch_raw) in enumerate(batches):
            print(f"  Encoding batch {batch_idx + 1}...")
            

            with torch.no_grad():
                embeddings = model.encode(batch_texts, 
                                          # for search, use Retrieval-query
                                          prompt_name="Retrieval-document",
                                          show_progress_bar=False, batch_size=encode_batch).tolist()
            
            points = []
            for idx, (metadata, text, p_id, raw_txt) in enumerate(zip(batch_metadatas, batch_texts, batch_ids, batch_raw)):
                try:
                    validated_payload = doc_payload(**metadata).model_dump()
                    validated_payload["text"] = raw_txt 
                    payload = validated_payload
                except Exception as e:
                    print(f"    Warning: Metadata validation failed: {e}")
                    payload = metadata.copy()
                    payload["text"] = texts[idx]
                    
                points.append(
                    PointStruct(
                        id=p_id, 
                        vector=embeddings[idx], 
                        payload=payload
                    )
                )
            try:
                client.upsert(collection_name=COLL_NAME, wait=True, points=points)
            except ValueError as e:
                print(pt.id for pt in points)
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()
        
def remove_points_from_pkl(pkl_path, collection_name=COLL_NAME):
    """
    Removes points from the Qdrant collection using IDs extracted from a pickle file.
    """
    print(f"Loading IDs for removal from: {pkl_path}")
    with open(pkl_path, "rb") as f:
        chunk = pickle.load(f)
    
    ids_to_remove = chunk["id"].tolist() if hasattr(chunk["id"], "tolist") else list(chunk["id"])
    
    try:
        client.delete(collection_name=collection_name, points_selector=models.PointIdsList(points=ids_to_remove))
        print(f"Successfully deleted {len(ids_to_remove)} points.")
    except Exception as e:
        print(f"Error during deletion: {e}")

#%%
if __name__ == "__main__":
    init_collection()
    upsert_data()
