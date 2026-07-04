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
from models import doc_payload
from db.database import get_qdrant_client
from db.utils import get_batches
from mlx_embeddings import load as emb_load
import mlx.core as mx


client = get_qdrant_client()
######################

######################
# Configuration
#coll_name_dense = "--limited --level 1 DENSE"

############### make false for not level 3
#level_3_enriched = False

import os
# path_ = "/Users/nadavsmacbookair/Desktop/Thesis/data/financial_corpora/chunks/hierarchical/indexed-at-30-06-26-limited-with-enriched/header"
# files = os.listdir(path_)
# files = [f for f in files if f.endswith(".pkl")]
# paths=[os.path.join(path_,f)for f in files]

#%%
def init_collection(coll_name_dense):
    try:
        client.create_collection(
            collection_name=coll_name_dense,
            vectors_config=VectorParams(size=768, distance=Distance.COSINE),
            on_disk_payload=True
        )
        print(f"Collection {coll_name_dense} created.")
    except Exception:
        print(f"Collection {coll_name_dense} already exists.")

    # Create indices by type. not everything must have a value
    fields = ["form_type", "company_name", "ticker"]
    for field in fields:
        client.create_payload_index(coll_name_dense, field, qdrant_models.PayloadSchemaType.KEYWORD)

    client.create_payload_index(coll_name_dense, "fiscal_year_end", qdrant_models.PayloadSchemaType.DATETIME)

    for field in ["section", "subsection", "item", "subitem", "run_header"]:
        client.create_payload_index(coll_name_dense, field, qdrant_models.PayloadSchemaType.TEXT)

    for field in ["doc_id", "parent_id"]:
        client.create_payload_index(coll_name_dense, field, qdrant_models.PayloadSchemaType.KEYWORD)


    client.create_payload_index(
        collection_name=coll_name_dense,
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
        collection_name=coll_name_dense,
        field_name="description",
        field_schema=qdrant_models.TextIndexParams(
            type=qdrant_models.TextIndexType.TEXT,
            tokenizer=qdrant_models.TokenizerType.WORD,
            lowercase=True,
            phrase_matching=True,
        ),
    )

def fmt_title(b):
    # 'section'/'subsection' are boilerplate SEC item titles (identical across
    # every 10-K ever filed) and add no discriminative signal to the embedding.
    # 'subitem' (deepest Markdown header, e.g. a table/line-item caption),
    # 'item' (the header directly above the content), and 'run_header' (a
    # child-level "*Label*" run-in subheading, e.g. "Americas", one level
    # deeper than the Markdown header hierarchy — see pack_prose_and_tables)
    # are the parts of the header hierarchy that actually describe what's in
    # the chunk, so those are what gets embedded — each is more specific than
    # the last when present, so all are kept together rather than one
    # replacing another.
    fields = [
        ("ticker", b.get("ticker")),
        ("item", b.get("item")),
        ("subitem", b.get("subitem")),
        ("run_header", b.get("run_header")),
    ]
    return "title: " + ", ".join(f"{k}: {v}" for k, v in fields if v)

def get_embedding(texts, model, tokenizer):
    """ 
    Returns embeddings for batch processing.  
    """
    inputs = tokenizer.batch_encode_plus(texts, return_tensors="mlx", padding=True, truncation=True)
    outputs = model(
        inputs["input_ids"],
        attention_mask=inputs["attention_mask"]
    )
    return outputs.text_embeds.tolist() # mean pooled and normalized embeddings

def upsert_data(chunk_paths, coll_name_dense, level_3_enriched=False):
    print(f"WARNING: are you absolutuley sure you want to ingest data? Make sure you are not replicating.\nthis is collection {coll_name_dense}")
    confirm = input("Type 'y' to proceed with ingestion: ")
    if confirm.lower() != 'y':
        print("Ingestion aborted.")
        return
    

    embed_model, embed_tokenizer = emb_load("mlx-community/embeddinggemma-300m-bf16")

    upsert_batch = 42
    encode_batch = 6

    print("Starting upserting loop...")

    for path in chunk_paths:
        print(f"Processing file: {path}")
        with open(path, "rb") as f:
            chunk = pickle.load(f)
    
        # this is for captions. 
        if level_3_enriched:
            texts = (chunk['description'] + chunk['text']).tolist()
        else:
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
        
        # Metadata is embedded via fmt_title() below — only the descriptive,
        # non-boilerplate fields (ticker/item/subitem), not the raw metadata dict.
        batches = zip(
            get_batches(texts, upsert_batch),
            get_batches(metadatas, upsert_batch),
            get_batches(ids, upsert_batch),
            get_batches(raw_texts, upsert_batch)
        )
        for batch_idx, (batch_texts, batch_metadatas, batch_ids, batch_raw) in enumerate(batches):

            print(f"  Encoding batch {batch_idx + 1}...")
            # formatted_docs contains the gemma-specific prompt style of "title: | text:"
            batch_texts_prompt          = [f" | text: {b}" for b in batch_texts]
            batch_metadatas_prompt      = [fmt_title(b) for b in batch_metadatas]
            formatted_doc = [b_meta + b_txt for (b_meta,b_txt) in zip(batch_metadatas_prompt, batch_texts_prompt)]
            embeddings = get_embedding(formatted_doc, embed_model, embed_tokenizer)

            
            points = []
            for idx, (metadata, text, p_id, raw_txt) in enumerate(zip(batch_metadatas, batch_texts, batch_ids, batch_raw)):
                try:
                    validated_payload = doc_payload(**metadata).model_dump()
                    validated_payload["text"] = raw_txt.lower() if isinstance(raw_txt, str) else raw_txt
                    payload = validated_payload
                except Exception as e:
                    print(f"    Warning: Metadata validation failed: {e}")
                    payload = metadata.copy()
                    payload["text"] = texts[idx].lower() if isinstance(texts[idx], str) else texts[idx]
                    
                points.append(
                    PointStruct(
                        id=p_id, 
                        vector=embeddings[idx], 
                        payload=payload
                    )
                )
            try:
                client.upsert(collection_name=coll_name_dense, wait=True, points=points)
            except ValueError as e:
                print(pt.id for pt in points)
        
def remove_points_from_pkl(pkl_path, collection_name):
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
    coll_name_dense = "--limited --level 3 DENSE"
    path_ = "/Users/nadavsmacbookair/Desktop/Thesis/data/financial_corpora/chunks/hierarchical/indexed-at-30-06-26-limited-with-enriched/enriched/batch_1"
    files = os.listdir(path_)
    files = [f for f in files if f.endswith(".pkl")]
    paths=[os.path.join(path_,f)for f in files]
    init_collection(coll_name_dense = coll_name_dense)
    upsert_data(chunk_paths=paths,
                coll_name_dense=coll_name_dense,
                level_3_enriched=True)
