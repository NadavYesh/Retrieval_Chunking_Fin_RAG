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
    Enrich SEC-derived Markdown by converting section markers and emphasized text to headings.

    Handles:
    - PART I|II|III|IV (Standalone or in tables)
    - Item X... (Standalone or in tables)
    - **Bold standalone lines** -> ### Heading
    - Short, Title Case standalone lines -> ### Heading
    - **Run-in headers.** Text -> #### Run-in header
    """
    if not md_text:
        return ""

    # Normalize line endings
    md_text = md_text.replace('\r\n', '\n').replace('\r', '\n')
    
    # Identify the end of the Table of Contents to avoid enriching TOC entries
    # Use a more robust search that doesn't strictly require newlines for the split
    toc_match = re.search(r"(?im)^\s*(table of contents|contents)\s*$", md_text)
    if toc_match:
        split_idx = md_text.find("\n", toc_match.end())
        if split_idx == -1:
            # If no newline after TOC, try to split at a reasonable distance or just take the whole thing
            prefix = md_text[:toc_match.end()]
            body = md_text[toc_match.end():]
        else:
            prefix = md_text[:split_idx + 1]
            body = md_text[split_idx + 1:]
    else:
        prefix = ""
        body = md_text

    # Regex definitions
    # Standalone Part/Item (Optional bolding, optional trailing punctuation)
    part_re = re.compile(r"^\s*(?:\*\*)?\s*(PART\s+(?:I|II|III|IV))\b\s*(.*?)(?:\*\*)?\s*$", re.I)
    item_re = re.compile(r"^\s*(?:\*\*)?\s*(ITEM\s+\d+[A-Z]?)(?:\.|\b)\s*(.*?)(?:\*\*)?\s*$", re.I)
    
    # Table-based Part/Item: | Item 1 | Business | or | PART I | |
    table_part_re = re.compile(r"^\s*\|\s*(PART\s+(?:I|II|III|IV))\b\s*\|\s*([^|]*?)\s*(?:\||$)", re.I)
    table_item_re = re.compile(r"^\s*\|\s*(ITEM\s+\d+[A-Z]?)(?:\.|\b)\s*\|\s*([^|]*?)\s*(?:\||$)", re.I)
    
    # Other header markers
    triple_bold_re = re.compile(r"^\s*\*\*\*\s*(.+?)\s*\*\*\*\s*$", re.I)
    double_bold_re = re.compile(r"^\s*\*\*\s*(.+?)\s*\*\*\s*$", re.I)
    
    # Run-in header heuristic: **Subheader.** Some text...
    runin_re = re.compile(r"^\s*\*\*\s*([^.*]{3,60}?)\.?\s*\*\*\s+(.+)$")

    lines = body.splitlines()
    out_lines = []
    
    for i, line in enumerate(lines):
        stripped = line.strip()
        if not stripped:
            out_lines.append(line)
            continue
            
        # Skip if already a header
        if stripped.startswith("#"):
            out_lines.append(line)
            continue

        # 1. Check Table-based Markers (Common in Intel)
        m_tp = table_part_re.match(stripped)
        if m_tp:
            out_lines.append(f"# {m_tp.group(1).upper()} {m_tp.group(2).strip()}")
            continue
        m_ti = table_item_re.match(stripped)
        if m_ti:
            out_lines.append(f"## {m_ti.group(1).upper()} {m_ti.group(2).strip()}")
            continue

        # 2. Check Standalone Part/Item (Common in BBY/AMZN)
        m_p = part_re.match(stripped)
        if m_p:
            out_lines.append(f"# {m_p.group(1).upper()} {m_p.group(2).strip()}")
            continue
        m_i = item_re.match(stripped)
        if m_i:
            out_lines.append(f"## {m_i.group(1).upper()} {m_i.group(2).strip()}")
            continue

        # 3. Triple Bold Standalone
        m_triple = triple_bold_re.match(stripped)
        if m_triple:
            out_lines.append(f"### {m_triple.group(1)}")
            continue

        # 4. Double Bold Standalone
        m_double = double_bold_re.match(stripped)
        if m_double:
            # Check if it's a known non-header (e.g. signature names)
            content = m_double.group(1)
            if len(content) < 100:
                out_lines.append(f"### {content}")
                continue

        # 5. Run-in header (#### Level)
        m_runin = runin_re.match(stripped)
        if m_runin:
            out_lines.append(f"#### {m_runin.group(1)}")
            out_lines.append("")
            out_lines.append(m_runin.group(2))
            continue

        # 6. Floating Header Heuristic (Plain text, short, Title Case, standalone)
        # Check context
        prev_empty = (i == 0 or not lines[i-1].strip())
        next_empty = (i == len(lines)-1 or not lines[i+1].strip())
        if prev_empty and next_empty and 3 < len(stripped) < 80 and stripped[0].isupper() and not stripped.endswith('.'):
            # Verify Title Case (majority of words start with upper)
            words = [w for w in stripped.split() if w.isalpha()]
            if words:
                upper_words = [w for w in words if w[0].isupper()]
                if len(upper_words) / len(words) > 0.6:
                    out_lines.append(f"### {stripped}")
                    continue

        out_lines.append(line)

    # Rejoin and fix double-empty lines that might have been introduced
    result = prefix + "\n".join(out_lines)
    return re.sub(r'\n{3,}', '\n\n', result)



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
