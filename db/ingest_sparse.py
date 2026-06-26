PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
import uuid
import pickle
import pandas as pd
from datetime import datetime
from qdrant_client import models
from qdrant_client.models import PointStruct

from models import doc_payload
from db.database import get_qdrant_client
from db.utils import get_batches

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

    for field in ["form_type", "company_name", "ticker", "doc_id", "parent_id"]:
        client.create_payload_index(COLLECTION_NAME, field, models.PayloadSchemaType.KEYWORD)

    client.create_payload_index(COLLECTION_NAME, "fiscal_year_end", models.PayloadSchemaType.DATETIME)

    for field in ["section", "subsection", "item"]:
        client.create_payload_index(COLLECTION_NAME, field, models.PayloadSchemaType.TEXT)

def upsert_sparse_data():
    batch_size = 32
    print("Starting sparse upserting loop...")

    for path in CHUNK_PATHS:
        print(f"Processing file: {path}")
        with open(path, "rb") as f:
            chunk = pickle.load(f)
        
        texts = chunk['text'].tolist() if hasattr(chunk['text'], 'tolist') else chunk['text']
        metadatas = chunk['metadata'].tolist() if hasattr(chunk['metadata'], 'tolist') else chunk['metadata']
        ids = chunk['id'].tolist() if 'id' in chunk.columns else [str(uuid.uuid4()) for _ in range(len(texts))]

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
        
        batches = zip(
            get_batches(combined, batch_size), 
            get_batches(metadatas, batch_size), 
            get_batches(ids, batch_size)
        )
        for batch_idx, (batch_texts, batch_metadatas, batch_ids) in enumerate(batches):
            print(f"  Processing batch {batch_idx + 1}...")
            points = []
            for idx, (metadata, text, p_id) in enumerate(zip(batch_metadatas, batch_texts, batch_ids)):
                stored_text = text[text.index("'text':")+8:] if EMBED_META else text
                if isinstance(stored_text, str):
                    stored_text = stored_text.lower()
                try:
                    validated_payload = doc_payload(**metadata).model_dump()
                    validated_payload["text"] = stored_text
                    payload = validated_payload
                except Exception as e:
                    print(f"    Warning: Metadata validation failed: {e}")
                    payload = metadata.copy()
                    payload["text"] = stored_text

                text_lower = text.lower() if isinstance(text, str) else text
                points.append(
                    PointStruct(
                        id=p_id,
                        vector={
                            "text": models.Document(
                                text=text_lower,
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
