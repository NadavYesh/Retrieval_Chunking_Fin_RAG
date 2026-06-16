#%%
import os
import sys
from pathlib import Path
import uuid
import pandas as pd
import pickle
# PROJECT_ROOT = Path(__file__).resolve().parents[1]
# if str(PROJECT_ROOT) not in sys.path:
#     sys.path.insert(0, str(PROJECT_ROOT))
# breakpoint()
#from chunking.metadata_extractor import get_meta_sec, sec_metadata
#from chunking.sec_processing import sec_to_mk, sec_splitter_headers, enrich_md_text, search_sec_bm25, sec_splitter_header_chars
from metadata_extractor import get_meta_sec, sec_metadata
from sec_processing import sec_to_mk, sec_splitter_headers, enrich_md_text, search_sec_bm25, sec_splitter_chars, inject_header_placeholders, collapse_double_newlines




#%%

def sec_chunking_pipeline(html_path, corpus_path, MD_PATH, 
                          method="head"):
    """
    **** fixed: formatting giberish issue
    Main pipeline for processing SEC HTML filings into searchable chunks.
    
    Args:
        html_path (str): Path to the raw SEC HTML file.
        output_path (str): Path where the resulting chunks will be saved.
        method (str): "head" or "head_and_chars"
    """
    print(f"Loading SEC filing from: {html_path}")
    with open(html_path, 'r', encoding='utf-8') as f:
        html_content = f.read()

    # Convert to Markdown
    print("Converting to Markdown...")
    mk_file = sec_to_mk(html_content)
    
    # enrich with header

    mk_file = enrich_md_text(mk_file)

    # inject headers placeholders to not neglect consecutive headers
    mk_file = inject_header_placeholders(mk_file)
    

    with open(MD_PATH, 'w', encoding='utf-8') as f:
        f.write(mk_file)

    meta = get_meta_sec(mk_file)

    if method == "head":
        print("Splitting into chunks header...")
        chunks = sec_splitter_headers(mk_file)
    elif method == "head_and_chars":
        print("Splitting into headers and chars chunks...")
        header_chunks = sec_splitter_headers(mk_file)
        chunks = sec_splitter_chars(doc = header_chunks, chunk_size=400, chunk_overlap=40)

    # convert chunks from langchanin list of document objects into DataFrame
    chunks_df = pd.DataFrame([
        {"id": str(uuid.uuid4()), "metadata": c.metadata, "text": c.page_content}
        for c in chunks
    ])

    # Inject document-level metadata into the chunks
    chunks_df = sec_metadata(chunks_df, meta)

    # Save chunks to Pickle
    print(f"Saving chunks to: {corpus_path}")
    chunks_df.to_pickle(corpus_path)
    return chunks_df



#%%

if __name__ == "__main__":
    #RAW_FILES = os.listdir("/Users/nadavsmacbookair/Desktop/Thesis/data/html/indexed at 13-6-26")
    #RAW_FILES = [f for f in RAW_FILES if f.endswith(".html")]
    RAW_FILES = ["BBY_10K_2024_copy.html"]

    print(RAW_FILES)
    #path_names = [os.path.join("/Users/nadavsmacbookair/Desktop/Thesis/data/html/indexed at 13-6-26",f)for f in RAW_FILES]
    path_names = ["/Users/nadavsmacbookair/Desktop/Thesis/data/html/indexed at 13-6-26/BBY_10K_2024_copy.html"]
    for (path,name) in zip(path_names,RAW_FILES):
        name = name[:-5]
        

        # CORPUS_PATH_header_split = f"/Users/nadavsmacbookair/Desktop/Thesis/Code_old/data/Corpus/{name}-Chunks-headers_split.pkl"
        CORPUS_PATH_header_split = f"/Users/nadavsmacbookair/Desktop/Thesis/data/financial_corpora/chunks/indexed-at-16-06-26/headers_split/{name}.pkl"
        CORPUS_PATH_header_char_split = f"/Users/nadavsmacbookair/Desktop/Thesis/data/financial_corpora/chunks/indexed-at-16-06-26/headers_chars_split/{name}.pkl"
        MD_PATH = f"/Users/nadavsmacbookair/Desktop/Thesis/data/financial_corpora/md/indexed-at-16-06-26/{name}.md"
        corpus_header_split = sec_chunking_pipeline(path, CORPUS_PATH_header_split, MD_PATH, method = "head")
        corpus_header_char_split = sec_chunking_pipeline(path, CORPUS_PATH_header_char_split, MD_PATH, method="head_and_chars")



#%%

# import pickle
# CORPUS_PATH = f"/Users/nadavsmacbookair/Desktop/Thesis/Code/data/Corpus/BESTBUY_2023_10K-Chunks.pkl"

# if os.path.exists(CORPUS_PATH):
#     with open (CORPUS_PATH, 'rb') as f:
#         corpus = pickle.load(f)
#         print(corpus)
# # %%
# corpus.iloc[140].metadata

#%% look at NEM_10K_2023.pkl
# import pickle
# with open ("/Users/nadavsmacbookair/Desktop/Thesis/data/financial_corpora/chunks/indexed-at-16-06-26/headers_split/BBY_10K_2024_copy.pkl", 'rb') as f:
#     corpus_headers = pickle.load(f)
# with open ("/Users/nadavsmacbookair/Desktop/Thesis/data/financial_corpora/chunks/indexed-at-16-06-26/headers_chars_split/BBY_10K_2024_copy.pkl", 'rb') as f:
#     corpus_headers_chars = pickle.load(f)