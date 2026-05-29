#%% kickstart qdrant
import pandas as pd
from qdrant_client import QdrantClient
client = QdrantClient(url="http://localhost:6333", check_compatibility=True)
import uuid
from qdrant_client.models import Distance, VectorParams

#%%
# load chunks pkl
import pickle
chunk_paths = ["/Users/nadavsmacbookair/Desktop/Thesis/code/data/BESTBUY_2023_10K-Chunks.pkl", "/Users/nadavsmacbookair/Desktop/Thesis/code/data/3M_2022_10K-Chunks.pkl"]
chunks_df=[]
for path in chunk_paths:
    with open(path, "rb") as f:
        chunks_df.append(pickle.load(f))

from sentence_transformers import SentenceTransformer
import torch
model = SentenceTransformer("google/embeddinggemma-300M", device="mps")

for chunk in chunks_df:
    embeddings = []
    chunk_text = chunk['text']
    for text in chunk_text:
        emb = model.encode(text).tolist()
        embeddings.append(emb)
    from qdrant_client.models import PointStruct

    points = []
    for idx,(metadata,text) in enumerate(zip(chunk["metadata"],chunk["text"])):
        metadata["text"] = text #add text as metadata
        points.append(
            PointStruct(id= uuid.uuid4(),vector=embeddings[idx], payload=metadata)
        )
    model_space = len(embeddings[0])
    try:
        client.create_collection(
            collection_name="10_k_col",
            vectors_config=VectorParams(size=model_space, distance=Distance.COSINE),
    )
    except Exception as e:
        print(f"Collection likely already exists")
    operation_info = client.upsert(
        collection_name="10_k_col",
        wait=True,
        points=points
    )

#%% Inspect Collection
client.scroll("10_k_col",limit=300000)

# %%
if __name__ == "__main__":
    # Example usage
    #RAW_PATH = "/Users/nadavsmacbookair/Documents/one_table.html"
    TEN_K_NAMES = ["3M_2022_10K","BESTBUY_2023_10K"]
    Q_A_fb = pd.read_json("/Users/nadavsmacbookair/Documents/sec2md/financebench/data/financebench_open_source.jsonl",lines=True)
    results_list = []
    for name in (TEN_K_NAMES):
        # SEARCH
        QA_subset = Q_A_fb[Q_A_fb['doc_name']==name]
        questions = QA_subset["question"].tolist()
        
        if not questions:
            continue
            
        print(f"Embedding questions for {name}...")
        # embed questions using the same model loaded above
        question_embeddings = model.encode(questions)
        
        print(f"Searching Qdrant for {name}...")
        # search in the qdrant collection
        for i, query_vector in enumerate(question_embeddings):
            search_results = client.query_points(
                collection_name="10_k_col",
                query=query_vector.tolist(),
                limit=5,
                with_payload=True,
            ).points
    
    # create a results df showing per question the retrieved chunks's payloads
    results_df = pd.DataFrame(results_list)
    print("Search Completed. Results DataFrame created.")
    print(results_df.head())

# %%
