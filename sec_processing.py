#%%
import re
import sys

# import local sec2md 
from pathlib import Path
sys.path.append("/Users/nadavsmacbookair/Documents/sec2md/src")
import sec2md 
#%%
from langchain_text_splitters import RecursiveCharacterTextSplitter, MarkdownHeaderTextSplitter

############# SECTION: HTML TO MARKDOWN ##################################
def sec_to_mk(html_content):
    """
    Converts SEC HTML filing content to Markdown format.

    Args:
        html_content (str): The raw HTML content of the SEC filing.

    Returns:
        str: The converted Markdown text.
    """
    return sec2md.convert_to_markdown(html_content)


def enrich_md_text(md_text):
    """
    Enrich SEC-derived Markdown by converting emphasized section markers to headings.

    Applies heading normalization to the body text line-by-line:
    - **PART I|II|III|IV** -> # PART ...
    - **Item X...** -> ## Item ...
    - ***text*** -> #### text
    - **text** -> ### text

    Args:
        md_text (str): Raw Markdown text.

    Returns:
        str: Markdown text with standardized heading levels.
    """
    toc_match = re.search(r"(?im)^\s*(table of contents|contents)\s*$", md_text)
    if toc_match:
        split_idx = md_text.find("\n", toc_match.end())
        if split_idx == -1:
            return md_text
        prefix = md_text[:split_idx + 1]
        body = md_text[split_idx + 1:]
    else:
        prefix = ""
        body = md_text

    part_re = re.compile(r"^\*\*\s*(PART\s+(?:I|II|III|IV))\s*\*\*\s*$", re.IGNORECASE)
    item_re = re.compile(r"^\*\*\s*(Item\s+\d+[A-Za-z]?(?:\.[^*]+)?)\s*\*\*\s*$", re.IGNORECASE)
    triple_bold_re = re.compile(r"^\*\*\*\s*(.+?)\s*\*\*\*$")
    double_bold_re = re.compile(r"^\*\*\s*(.+?)\s*\*\*$")

    out_lines = []
    for line in body.splitlines():
        stripped = line.strip()
        m_part = part_re.match(stripped)
        if m_part:
            out_lines.append(f"# {m_part.group(1).upper()}")
            continue

        m_item = item_re.match(stripped)
        if m_item:
            out_lines.append(f"## {m_item.group(1)}")
            continue

        m_triple = triple_bold_re.match(stripped)
        if m_triple:
            out_lines.append(f"#### {m_triple.group(1)}")
            continue

        m_double = double_bold_re.match(stripped)
        if m_double:
            out_lines.append(f"### {m_double.group(1)}")
            continue

        out_lines.append(line)

    return prefix + "\n".join(out_lines)



############# SECTION: MARKDOWN TO CHUNKS ##################################
def sec_splitter_headers(doc):
    """
    Performs a hierarchical split on SEC Markdown documents.

    1. Header Splitting: Segments by Markdown headers (#, ##, ###).

    Args:
        doc (str): The Markdown-formatted SEC filing text.


    Returns:
        list[Document]: A list of LangChain Document objects with metadata.
    """
    # making sure tables are not split- never
    # table = starts with "|" and ends with "/n/n" 

    header_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=[
            ("#", "section"),
            ("##", "subsection"),
            ("###", "item"),
            ("####", "subitem")
            
        ]
    )
    header_chunks = header_splitter.split_text(doc)
    return(header_chunks)

def sec_splitter_header_chars(doc, chunk_size=800, chunk_overlap=200):
    """
    Performs a hierarchical split on SEC Markdown documents.

    1. Header Splitting: Segments by Markdown headers (#, ##, ###).
    2. Character Splitting: Sub-divides large sections into smaller chunks.

    Args:
        doc (str): The Markdown-formatted SEC filing text.
        chunk_size (int): Max characters per chunk.
        chunk_overlap (int): Character overlap between chunks.

    Returns:
        list[Document]: A list of LangChain Document objects with metadata.
    """
    # making sure tables are not split- never
    # table = starts with "|" and ends with "/n/n" 

    header_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=[
            ("#", "section"),
            ("##", "subsection"),
            ("###", "item"),
            ("####", "subitem")
            
        ]
    )
    header_chunks = header_splitter.split_text(doc)
    char_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=[
            "\n\n",   # Double newline (paragraph break)
            "\n",     # Single newline
            " ",      # Space
            "\u200b", # Zero-width space
            "\uff0c", # Fullwidth comma ，
            "\u3001", # Ideographic comma 、
            "\uff0e", # Fullwidth full stop ．
            "\u3002", # Ideographic full stop 。
            "",       # Empty string (character-level fallback)
        ],
        length_function = len)
    final_chunks = char_splitter.split_documents(header_chunks)
    return final_chunks


def search_sec_bm25(query, corpus_df, k=5):
    """
    Performs a simple BM25 search on the SEC corpus.
    
    Args:
        query (str): The search query.
        corpus_df (pd.DataFrame): DataFrame containing 'text' column.
        k (int): Number of results to return.
    """
    import bm25s
    # only sending the text. 
    retriever = bm25s.BM25(corpus=corpus_df["text"])
    retriever.index(bm25s.tokenize(corpus_df["text"]))
    
    results, scores = retriever.retrieve(bm25s.tokenize(query), k=k)
    return results, scores

# %%
