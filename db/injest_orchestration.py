from ingest_dense import init_collection, upsert_data

COLL_NAME_DENSE = [#"--limited --level 1 DENSE",
                   "--limited --level 2 DENSE","--limited --level 3 DENSE"]
paths = [#"/Users/nadavsmacbookair/Desktop/Thesis/data/financial_corpora/chunks/hierarchical/indexed-at-30-06-26-limited-with-enriched/header/batch_1",
         "/Users/nadavsmacbookair/Desktop/Thesis/data/financial_corpora/chunks/hierarchical/indexed-at-30-06-26-limited-with-enriched/child/batch_1",
         "/Users/nadavsmacbookair/Desktop/Thesis/data/financial_corpora/chunks/hierarchical/indexed-at-30-06-26-limited-with-enriched/enriched/batch_1"
         ]
############### make false for not level 3

# EMBED_META = [False,False,True] # this is misleading, as the current file embed textual meta.
import os
LEVEL_3_ENRICHED = [#False,
                    False,
                    True]

for (c_name,path_,l_three) in zip (COLL_NAME_DENSE,paths,LEVEL_3_ENRICHED):
    init_collection(c_name)
    files = os.listdir(path_)
    files = [f for f in files if f.endswith(".pkl")]
    paths=[os.path.join(path_,f)for f in files]
    upsert_data(CHUNK_PATHS=paths, LEVEL_3_ENRICHED=l_three, COLL_NAME_DENSE=c_name)