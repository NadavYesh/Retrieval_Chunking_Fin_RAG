#%%
import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
import uuid
import pickle
import pandas as pd
from datetime import datetime
from qdrant_client import models as qdrant_models
from qdrant_client.models import PointStruct, VectorParams, Distance
from models import doc_payload, MLXEmbedder
from db.database import get_qdrant_client
from db.utils import get_batches
from mlx_embeddings import load as emb_load
import mlx.core as mx


client = get_qdrant_client()
######################

######################
# Configuration
COLL_NAME = "--level 3"
EMBED_META = False
import os
path_ = "/Users/nadavsmacbookair/Desktop/Thesis/data/financial_corpora/chunks/hierarchical/indexed-at-22-06-26/enriched"
files = os.listdir(path_)
files = [f for f in files if f.endswith(".pkl")]
paths=[os.path.join(path_,f)for f in files]

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

    # Create indices by type. not everything must have a value
    fields = ["form_type", "company_name", "ticker"]
    for field in fields:
        client.create_payload_index(COLL_NAME, field, qdrant_models.PayloadSchemaType.KEYWORD)

    client.create_payload_index(COLL_NAME, "fiscal_year_end", qdrant_models.PayloadSchemaType.DATETIME)

    for field in ["section", "subsection", "item"]:
        client.create_payload_index(COLL_NAME, field, qdrant_models.PayloadSchemaType.TEXT)

    for field in ["doc_id", "parent_id"]:
        client.create_payload_index(COLL_NAME, field, qdrant_models.PayloadSchemaType.KEYWORD)

    # this will allow bm25
    client.create_payload_index(
        collection_name=COLL_NAME,
        field_name="text",
        field_schema=qdrant_models.TextIndexParams(
            type=qdrant_models.TextIndexType.TEXT,
            tokenizer=qdrant_models.TokenizerType.WORD,
            lowercase=True,
            phrase_matching=True,
        ),
    )
    # only for enriched
    client.create_payload_index(
        collection_name=COLL_NAME,
        field_name="description",
        field_schema=qdrant_models.TextIndexParams(
            type=qdrant_models.TextIndexType.TEXT,
            tokenizer=qdrant_models.TokenizerType.WORD,
            lowercase=True,
            phrase_matching=True,
        ),
    )

def get_embedding(texts, model, tokenizer):
    """ 
    Returns embeddings for batch processing.  
    """
    inputs = tokenizer.batch_encode_plus(texts, return_tensors="mlx", padding=True, truncation=True)
    outputs = model(
        inputs["input_ids"],
        attention_mask=inputs["attention_mask"]
    )
    return outputs.text_embeds.tolist()[0] # mean pooled and normalized embeddings

def upsert_data():
    print(f"WARNING: are you absolutuley sure you want to ingest data? Make sure you are not replicating.\nthis is collection {COLL_NAME}")
    confirm = input("Type 'y' to proceed with ingestion: ")
    if confirm.lower() != 'y':
        print("Ingestion aborted.")
        return
    

    embed_model, embed_tokenizer = emb_load("mlx-community/embeddinggemma-300m-bf16")

    model = MLXEmbedder("mlx-community/embeddinggemma-300m-bf16")

    upsert_batch = 42
    encode_batch = 6

    print("Starting upserting loop...")

    for path in CHUNK_PATHS:
        print(f"Processing file: {path}")
        with open(path, "rb") as f:
            chunk = pickle.load(f)
    
        # this is for captions. 
        #texts = (chunk['description'] + chunk['text']).tolist()

        texts = chunk['text'].tolist()
        
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
            doc_title = f"10-K filing for ticker: {batch_metadatas.get("ticker")}, fiscal year end: {batch_metadatas.get("fiscal_year_end")}"
            # formatted_docs contains the gemma-specific prompt style of "title: | text:"

            #f"title: {doc_title} | text: {batch_texts}"
            breakpoint()
            batch_texts_prompt          = [f" | text: {b}" for b in batch_texts]
            batch_metadatas_prompt      =  [f"title: {b}" for b in batch_texts]
            formatted_doc = [b_meta + b_txt for (b_meta,b_txt) in zip(batch_metadatas_prompt, batch_metadatas_prompt)]
            embeddings = get_embedding(formatted_doc, embed_model, embed_tokenizer)

            # embeddings = model.encode(
            #     batch_texts,
            #     prompt_name="document",
            #     show_progress_bar=False,
            #     batch_size=encode_batch,
            # ).tolist()
            
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
        
def remove_points_from_pkl(pkl_path, collection_name=COLL_NAME):
    """
    Removes points from the Qdrant collection using IDs extracted from a pickle file.
    """
    print(f"Loading IDs for removal from: {pkl_path}")
    with open(pkl_path, "rb") as f:
        chunk = pickle.load(f)
    
    ids_to_remove = chunk["id"].tolist() if hasattr(chunk["id"], "tolist") else list(chunk["id"])
    
    try:
        client.delete(collection_name=collection_name, points_selector=qdrant_models.PointIdsList(points=ids_to_remove))
        print(f"Successfully deleted {len(ids_to_remove)} points.")
    except Exception as e:
        print(f"Error during deletion: {e}")

#%%
if __name__ == "__main__":
    init_collection()
    upsert_data()
