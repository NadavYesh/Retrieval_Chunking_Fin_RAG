#%%
import os
import sys
from pathlib import Path
import uuid
import pandas as pd
import pickle
import tiktoken
# PROJECT_ROOT = Path(__file__).resolve().parents[1]
# if str(PROJECT_ROOT) not in sys.path:
#     sys.path.insert(0, str(PROJECT_ROOT))
# breakpoint()
#from chunking.metadata_extractor import get_meta_sec, sec_metadata
#from chunking.sec_processing import sec_to_mk, sec_splitter_headers, enrich_md_text, search_sec_bm25, sec_splitter_header_chars
from metadata_extractor import get_meta_sec, sec_metadata
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sec_processing import sec_to_mk, sec_splitter_headers, enrich_md_text, inject_header_placeholders,chunk_document




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
        chunks = chunk_document(budget = BUDGET,header_chunks = header_chunks, char_splitter=CHAR_SPLITTER,length_function=LENGTH_FUNC)
        

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



def sec_chunking_pipeline_hierarchical(
    html_path,
    doc_path,
    header_path,
    child_path,
    MD_PATH,
    budget: int,
    length_function,
    char_splitter,
):
    """
    Produces three granularity levels with parent-document linkage:

      doc (1 chunk)
       └── header chunks  (doc_id → doc)
             └── child chunks  (parent_id → header, doc_id → doc)

    Saves three separate pickle files — one per level.
    """
    print(f"Loading SEC filing from: {html_path}")
    with open(html_path, 'r', encoding='utf-8') as f:
        html_content = f.read()

    print("Converting to Markdown...")
    mk_file = sec_to_mk(html_content)
    mk_file = enrich_md_text(mk_file)
    mk_file = inject_header_placeholders(mk_file)

    with open(MD_PATH, 'w', encoding='utf-8') as f:
        f.write(mk_file)

    meta = get_meta_sec(mk_file)

    # ── Level 0: full document ────────────────────────────────────────────────
    doc_id = str(uuid.uuid4())
    doc_df = pd.DataFrame([{"id": doc_id, "metadata": dict(meta), "text": mk_file}])
    doc_df = sec_metadata(doc_df, meta)

    # ── Level 1: header chunks ────────────────────────────────────────────────
    header_chunks_raw = sec_splitter_headers(mk_file)

    header_rows = []
    for hc in header_chunks_raw:
        hid = str(uuid.uuid4())
        hc.metadata["_id"] = hid          # consumed by chunk_document; stripped below
        header_rows.append({
            "id": hid,
            "metadata": {**hc.metadata, "doc_id": doc_id},
            "text": hc.page_content,
        })
    header_df = pd.DataFrame(header_rows)
    # strip the internal marker from saved metadata
    header_df["metadata"] = header_df["metadata"].apply(
        lambda m: {k: v for k, v in m.items() if k != "_id"}
    )
    header_df = sec_metadata(header_df, meta)

    # ── Level 2: child chunks (header → char/table split) ────────────────────
    child_chunks = chunk_document(
        budget=budget,
        header_chunks=header_chunks_raw,   # _id already injected above
        char_splitter=char_splitter,
        length_function=length_function,
    )
    child_df = pd.DataFrame([
        {"id": str(uuid.uuid4()), "metadata": {**c.metadata, "doc_id": doc_id}, "text": c.page_content}
        for c in child_chunks
    ])
    child_df = sec_metadata(child_df, meta)

    # ── Persist ───────────────────────────────────────────────────────────────
    print(f"Saving doc chunk to:    {doc_path}")
    doc_df.to_pickle(doc_path)

    print(f"Saving header chunks to: {header_path}")
    header_df.to_pickle(header_path)

    print(f"Saving child chunks to:  {child_path}")
    child_df.to_pickle(child_path)

    return doc_df, header_df, child_df


#%%

if __name__ == "__main__":
    RAW_FILES = os.listdir("/Users/nadavsmacbookair/Desktop/Thesis/data/html/indexed at 13-6-26")
    RAW_FILES = [f for f in RAW_FILES if f.endswith(".html")]
    path_names = [os.path.join("/Users/nadavsmacbookair/Desktop/Thesis/data/html/indexed at 13-6-26/",f) for f in RAW_FILES]

    BUDGET = 400
    enc = tiktoken.encoding_for_model("text-embedding-3-small")
    LENGTH_FUNC = lambda text: len(enc.encode(text))
    CHAR_SPLITTER = RecursiveCharacterTextSplitter(
        chunk_size=BUDGET,
        chunk_overlap=40,
        separators=["\n\n","\n", " ", "\u200b", "\uff0c", "\u3001", "\uff0e", "\u3002", ""],
        length_function=LENGTH_FUNC,
    )

    BASE_CHUNKS = "/Users/nadavsmacbookair/Desktop/Thesis/data/financial_corpora/chunks/hierarchical"
    BASE_MD    = "/Users/nadavsmacbookair/Desktop/Thesis/data/financial_corpora/md/indexed-at-22-06-26"

    for (path, name) in zip(path_names, RAW_FILES):
        name = name[:-5]
        MD_PATH     = f"{BASE_MD}/{name}.md"
        doc_path    = f"{BASE_CHUNKS}/doc/{name}.pkl"
        header_path = f"{BASE_CHUNKS}/header/{name}.pkl"
        child_path  = f"{BASE_CHUNKS}/child/{name}.pkl"
        sec_chunking_pipeline_hierarchical(
            html_path=path,
            doc_path=doc_path,
            header_path=header_path,
            child_path=child_path,
            MD_PATH=MD_PATH,
            budget=BUDGET,
            length_function=LENGTH_FUNC,
            char_splitter=CHAR_SPLITTER,
        )
    
