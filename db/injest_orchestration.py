from ingest_dense import init_collection, upsert_data
import ingest_bm25 as bm25

COLL_NAME_DENSE = ["--PILOT --level 1 DENSE",
                   "--PILOT --level 2 DENSE","--PILOT --level 3 DENSE"]
COLL_NAME_SPARSE = ["--PILOT --level 1 BM25",
                     "--PILOT --level 2 BM25","--PILOT --level 3 BM25"]
paths = ["/Users/nadavsmacbookair/Desktop/Thesis/data/financial_corpora/chunks/hierarchical/indexed-at-07-07-26/header/upsert_batch_2",
         "/Users/nadavsmacbookair/Desktop/Thesis/data/financial_corpora/chunks/hierarchical/indexed-at-07-07-26/child/upsert_batch_2",
         "/Users/nadavsmacbookair/Desktop/Thesis/data/financial_corpora/chunks/hierarchical/indexed-at-07-07-26/enriched/upsert_batch_2"
         ]
############### make false for not level 3

# EMBED_META = [False,False,True] # this is misleading, as the current file embed textual meta.
import os
LEVEL_3_ENRICHED = [False,
                    False,
                    True]

for (c_name,sparse_c_name,path_,l_three) in zip(COLL_NAME_DENSE,COLL_NAME_SPARSE,paths,LEVEL_3_ENRICHED):
    files = os.listdir(path_)
    files = [f for f in files if f.endswith(".pkl")]
    chunk_paths = [os.path.join(path_,f) for f in files]

    init_collection(c_name)
    upsert_data(chunk_paths=chunk_paths, level_3_enriched=l_three, coll_name_dense=c_name)

    bm25.COLL_NAME_SPARSE = sparse_c_name
    bm25.LEVEL_3_ENRICHED = l_three
    bm25.CHUNK_PATHS = chunk_paths
    bm25.init_collection()
    bm25.upsert_bm25_data()