import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import os
import uuid
import pickle
from datetime import datetime

from qdrant_client import models
from qdrant_client.models import PointStruct

from models import doc_payload
from db.database import get_qdrant_client
from db.utils import get_batches

client = get_qdrant_client()

COLL_NAME_SPARSE = "--level 2 BM25"
BATCH_SIZE      = 64
path_ = "/Users/nadavsmacbookair/Desktop/Thesis/data/financial_corpora/chunks/hierarchical/indexed-at-30-06-26/child"
files = [f for f in os.listdir(path_) if f.endswith(".pkl")]
CHUNK_PATHS = [os.path.join(path_, f) for f in files]


def init_collection():
    try:
        client.create_collection(
            collection_name=COLL_NAME_SPARSE,
            sparse_vectors_config={
                "bm25": models.SparseVectorParams(modifier=models.Modifier.IDF)
            },
            on_disk_payload=True,
        )
        print(f"Collection '{COLL_NAME_SPARSE}' created.")
    except Exception:
        print(f"Collection '{COLL_NAME_SPARSE}' already exists — skipping creation.")

    for field in ["form_type", "company_name", "ticker", "doc_id", "parent_id"]:
        client.create_payload_index(COLL_NAME_SPARSE, field, models.PayloadSchemaType.KEYWORD)

    client.create_payload_index(COLL_NAME_SPARSE, "fiscal_year_end", models.PayloadSchemaType.DATETIME)

    for field in ["section", "subsection", "item"]:
        client.create_payload_index(COLL_NAME_SPARSE, field, models.PayloadSchemaType.TEXT)


def upsert_bm25_data():
    print(f"WARNING: ingesting into '{COLL_NAME_SPARSE}'. Make sure you are not duplicating.")
    confirm = input("Type 'y' to proceed: ")
    if confirm.lower() != "y":
        print("Ingestion aborted.")
        return

    total_points = 0
    for path in CHUNK_PATHS:
        print(f"\nProcessing: {path}")
        with open(path, "rb") as f:
            chunk = pickle.load(f)

        texts     = chunk["text"].tolist()
        metadatas = chunk["metadata"].tolist() if hasattr(chunk["metadata"], "tolist") else list(chunk["metadata"])
        ids       = chunk["id"].tolist() if "id" in chunk.columns else [str(uuid.uuid4()) for _ in range(len(texts))]

        # Same date preprocessing as ingest_dense.py
        for i, meta in enumerate(metadatas):
            date_ = meta.get("fiscal_year_end", "")
            if isinstance(date_, str) and len(date_) >= 8:
                if not date_[-4:-2].isdigit() or int(date_[-4:-2]) < 19:
                    date_ = date_[: len(date_) - 2] + "20" + date_[len(date_) - 2 :]
                metadatas[i]["fiscal_year_end"] = datetime.strptime(date_, "%m-%d-%Y").date()

        for batch_idx, (batch_texts, batch_metas, batch_ids) in enumerate(
            zip(get_batches(texts, BATCH_SIZE), get_batches(metadatas, BATCH_SIZE), get_batches(ids, BATCH_SIZE))
        ):
            points = []
            for text, metadata, p_id in zip(batch_texts, batch_metas, batch_ids):
                try:
                    payload = doc_payload(**metadata).model_dump()
                except Exception as e:
                    print(f"  Metadata validation failed: {e}")
                    payload = dict(metadata)
                text_lower = text.lower() if isinstance(text, str) else text
                payload["text"] = text_lower

                points.append(
                    PointStruct(
                        id=p_id,
                        vector={"bm25": models.Document(text=text_lower, model="qdrant/bm25")},
                        payload=payload,
                    )
                )

            client.upsert(collection_name=COLL_NAME_SPARSE, wait=True, points=points)
            total_points += len(points)
            print(f"  Batch {batch_idx + 1}: upserted {len(points)} points (total so far: {total_points})")

    print(f"\nDone. Total points upserted: {total_points}")


if __name__ == "__main__":
    init_collection()
    upsert_bm25_data()
