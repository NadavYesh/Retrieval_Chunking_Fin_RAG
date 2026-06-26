#%%
import os
import sys
from pathlib import Path
import uuid
import pandas as pd
import pickle
import tiktoken
sys.path.insert(0, str(Path(__file__).parent))
from prompts import ENRICH_CHUNKS_PROMPT 
from metadata_extractor import get_meta_sec, sec_metadata
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sec_processing import sec_to_mk, sec_splitter_headers, enrich_md_text, inject_header_placeholders, chunk_document




# ═════════════════════════════════════════════════════════════════════════════
# Sub-functions — each level is independently callable
# ═════════════════════════════════════════════════════════════════════════════

def _html_to_md(html_path: str, MD_PATH: str) -> str:
    """HTML → enriched, placeholder-injected markdown. Saves .md file."""
    print(f"Loading SEC filing from: {html_path}")
    with open(html_path, 'r', encoding='utf-8') as f:
        html_content = f.read()
    print("Converting to Markdown...")
    mk_file = sec_to_mk(html_content)
    mk_file = enrich_md_text(mk_file)
    mk_file = inject_header_placeholders(mk_file)
    Path(MD_PATH).parent.mkdir(parents=True, exist_ok=True)
    with open(MD_PATH, 'w', encoding='utf-8') as f:
        f.write(mk_file)
    return mk_file


def build_doc_level(mk_file: str, meta: dict) -> tuple[str, pd.DataFrame]:
    """
    Level 0 — one chunk per filing: full markdown text + doc-level metadata.

    Returns:
        doc_id:  UUID string for this document (referenced by lower levels).
        doc_df:  Single-row DataFrame.
    """
    doc_id = str(uuid.uuid4())
    doc_df = pd.DataFrame([{"id": doc_id, "metadata": dict(meta), "text": mk_file}])
    doc_df = sec_metadata(doc_df, meta)
    return doc_id, doc_df


def build_header_level(mk_file: str, meta: dict, doc_id: str) -> tuple[pd.DataFrame, list]:
    """
    Level 1 — header-split chunks; each carries doc_id → Level 0.

    Returns:
        header_df:         DataFrame of header chunks (doc_id in metadata).
        header_chunks_raw: LangChain Document list with _id injected in metadata
                           (needed as input to build_child_level).
    """
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
    header_df["metadata"] = header_df["metadata"].apply(
        lambda m: {k: v for k, v in m.items() if k != "_id"}
    )
    header_df = sec_metadata(header_df, meta)
    return header_df, header_chunks_raw


def build_child_level(
    header_chunks_raw: list,
    meta: dict,
    doc_id: str,
    budget: int,
    length_function,
    char_splitter,
) -> pd.DataFrame:
    """
    Level 2 — dense child chunks; each carries parent_id → Level 1 and doc_id → Level 0.

    header_chunks_raw must have _id set in metadata (output of build_header_level).
    """
    child_chunks = chunk_document(
        budget=budget,
        header_chunks=header_chunks_raw,
        char_splitter=char_splitter,
        length_function=length_function,
    )
    child_df = pd.DataFrame([
        {
            "id": str(uuid.uuid4()),
            "metadata": {**c.metadata, "doc_id": doc_id},
            "text": c.page_content,
        }
        for c in child_chunks
    ])
    child_df = sec_metadata(child_df, meta)
    return child_df


def build_enriched_level(
    child_df: pd.DataFrame,
    caption_=None,
) -> pd.DataFrame:
    """
    Level 3 — child chunks augmented with an LLM-generated description.

    The description captures the semantic meaning of the chunk including the
    nature of any tables, reducing embedding-model dependence on tabular layout.
    At ingest time, embed 'description' instead of 'text'.

    Args:
        child_df: Output of build_child_level.
        caption_:     Optional pre-loaded (model, tokenizer) tuple from mlx_lm.load().
                  Pass it when processing multiple files to avoid reloading per file.
                  If None, the model is loaded automatically.

    Returns:
        enriched_df: copy of child_df with a top-level 'description' column
                     and 'description' injected into each row's metadata dict.
    """
    from mlx_lm import load, generate as mlx_generate

    if caption_ is  None:
        print("Loading phi-4 for chunk enrichment...")
        model, tokenizer = load("mlx-community/phi-4-4bit")
    # else:
    #     model, tokenizer = caption_

    n = len(child_df)
    descriptions = []

    for i, (_, row) in enumerate(child_df.iterrows()):
        messages = [
            {"role": "system", "content": ENRICH_CHUNKS_PROMPT},
            {"role": "user",   "content": row["text"]},
        ]
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        desc = mlx_generate(model, tokenizer, prompt=prompt, max_tokens=150, verbose=False)
        descriptions.append(desc.strip())
        if (i + 1) % 20 == 0 or (i + 1) == n:
            print(f"  Enriched {i + 1}/{n} chunks")

    enriched_df = child_df.copy()
    enriched_df["description"] = descriptions
    enriched_df["metadata"] = enriched_df.apply(
        lambda r: {**r["metadata"], "description": r["description"]}, axis=1
    )
    return enriched_df



# ═════════════════════════════════════════════════════════════════════════════
# Hierarchical pipeline — orchestrates all levels
# ═════════════════════════════════════════════════════════════════════════════

def sec_chunking_pipeline_hierarchical(
    html_path: str,
    doc_path: str,
    header_path: str,
    child_path: str,
    MD_PATH: str,
    budget: int,
    length_function,
    char_splitter,
    enriched_path: str = None,
    caption_=None,
):
    """
    Runs all granularity levels and links them via parent_id / doc_id:

      Level 0  doc      (1 chunk)   — full filing text
      Level 1  header   (N chunks)  — header-split; doc_id → Level 0
      Level 2  child    (M chunks)  — dense ~400-tok; parent_id → Level 1, doc_id → Level 0
      Level 3  enriched (M chunks)  — child + LLM description; optional, only if enriched_path given

    Each level is saved as a separate pickle.
    Pass caption_=(model, tokenizer) when processing multiple files to load once.
    """
    mk_file = _html_to_md(html_path, MD_PATH)
    meta    = get_meta_sec(mk_file)

    doc_id  = str(uuid.uuid4())
    doc_df  = None
    if doc_path is not None:
        doc_id, doc_df = build_doc_level(mk_file, meta)

    header_df, header_chunks_raw = build_header_level(mk_file, meta, doc_id)

    child_df = None
    if child_path is not None or enriched_path is not None:
        child_df = build_child_level(
            header_chunks_raw, meta, doc_id, budget, length_function, char_splitter
        )

    enriched_df = None
    if enriched_path is not None:
        enriched_df = build_enriched_level(child_df, caption_=caption_)

    for p, df, label in [
        (doc_path,      doc_df,      "doc chunk    "),
        (header_path,   header_df,   "header chunks"),
        (child_path,    child_df,    "child chunks "),
        (enriched_path, enriched_df, "enriched chunks"),
    ]:
        if p is None:
            continue
        Path(p).parent.mkdir(parents=True, exist_ok=True)
        print(f"Saving {label} → {p}")
        df.to_pickle(p)

    return doc_df, header_df, child_df, enriched_df


# ─────────────────────────────────────────────────────────────────────────────
#%%
if __name__ == "__main__":
    RAW_DIR  = "/Users/nadavsmacbookair/Desktop/Thesis/data/html/indexed at 26-6-26/batch_2"
    RAW_FILES = [f for f in os.listdir(RAW_DIR) if f.endswith(".html")]
    path_names = [os.path.join(RAW_DIR, f) for f in RAW_FILES]

    BUDGET = 400
    enc = tiktoken.encoding_for_model("text-embedding-3-small")
    LENGTH_FUNC = lambda text: len(enc.encode(text))
    CHAR_SPLITTER = RecursiveCharacterTextSplitter(
        chunk_size=BUDGET,
        chunk_overlap=40,
        separators=["\n\n", "\n", " ", "​", "，", "、", "．", "。", ""],
        length_function=LENGTH_FUNC,
    )

    BASE_CHUNKS = "/Users/nadavsmacbookair/Desktop/Thesis/data/financial_corpora/chunks/hierarchical/indexed-at-26-06-26/header/batch_2"
    BASE_MD     = "/Users/nadavsmacbookair/Desktop/Thesis/data/financial_corpora/md/indexed-at-26-06-26/batch_2"

    # Load phi-4 once for the whole batch (expensive — skip if not running level 3)
    RUN_ENRICHED = False
    caption_ = None
    if RUN_ENRICHED:
        from mlx_lm import load
        print("Loading Llama...")
        caption_ = load("mlx-community/Llama-3.2-3B-Instruct-4bit")

    for (path, name) in zip(path_names, RAW_FILES):
        name        = name[:-5]
        MD_PATH     = f"{BASE_MD}/{name}.md"
        #doc_path    = f"{BASE_CHUNKS}/doc/{name}.pkl"
        header_path = f"{BASE_CHUNKS}/header/{name}.pkl"
        #child_path  = f"{BASE_CHUNKS}/child/{name}.pkl"
        #enriched_path = f"{BASE_CHUNKS}/enriched/{name}.pkl" if RUN_ENRICHED else None

        sec_chunking_pipeline_hierarchical(
            html_path=path,
            doc_path=None,
            header_path=header_path,
            child_path=None,
            MD_PATH=MD_PATH,
            budget=BUDGET,
            length_function=LENGTH_FUNC,
            char_splitter=CHAR_SPLITTER,
            #enriched_path=enriched_path,
            #caption_=caption_,
        )


# for (path, name) in zip(path_names, RAW_FILES):
#         name = name[:-5]
#         CHILD_path = f"{BASE_CHUNKS}/child/{name}.pkl"
#         MD_PATH     = f"{BASE_MD}/{name}.md"
#         enriched_path = f"{BASE_CHUNKS}/enriched/{name}.pkl" if RUN_ENRICHED else None
#         import pickle
#         with open(CHILD_path, "rb") as f:
#             child_df = pickle.load(f)
#         print(f"starting enriching doc {name}")
#         enriched_df = build_enriched_level(child_df, caption_=caption_)
#         print(f"Saving enriched chunks → {enriched_path}")
#         enriched_df.to_pickle(enriched_path)
