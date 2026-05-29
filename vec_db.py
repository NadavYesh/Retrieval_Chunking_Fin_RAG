


# %% SEARCH
import numpy.random
# Query expects a 1D list/array, not a 2D array

vec_test = numpy.random.rand(model_space).tolist()


search_result = client.query_points(
    collection_name="10_k_col",  # Matched your previously created collection name
    query=vec_test,
    with_payload=False,
    limit=3
).points
search_idx = [point.id for point in search_result]
print(search_result)


#%% Enrich Chunks with doc level metadata
'''
HTML ==> MD ==> CHUNKS  ==> COMBINE
            ===>METADATA
'''
if __name__ == "__main__":
    import chunks_chk, pickle
    with open ("/Users/nadavsmacbookair/Desktop/Thesis/Code_old/data/markdown/3M_2022_10K.md", 'r') as f:
        md_doc = f.read()
    with open ("/Users/nadavsmacbookair/Desktop/Thesis/Code_old/data/Corpus/3M_2022_10K-Chunks.pkl", 'rb') as f:
        chunks = pickle.load(f)
    meta_dict = chunks_chk.get_meta_sec(content=md_doc)
    enr_chunks = chunks_chk.sec_metadata(chunks, meta_dict)
    enr_chunks.to_pickle("/Users/nadavsmacbookair/Desktop/Thesis/code/data/3M_2022_10K-Chunks.pkl")
    
    

