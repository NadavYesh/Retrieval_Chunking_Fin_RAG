#%%
import re
import sys
# import local sec2md 
from pathlib import Path
sys.path.append("/Users/nadavsmacbookair/Documents/sec2md/src")
import sec2md 
from typing import Optional
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter, MarkdownHeaderTextSplitter


#%%

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



def inject_header_placeholders(markdown: str) -> str:
    """
    ########## Solution to ignored headers problem ################
    Inject a placeholder line after any header that has no immediate content
    (i.e. next non-empty line is another header).
    """
    lines = markdown.split("\n")
    result = []
    header_pattern = re.compile(r"^(#{1,4}) (.+)")
    i = 0

    while i < len(lines):
        line = lines[i]
        match = header_pattern.match(line)

        if match:
            level = match.group(1)   # e.g. "###"
            title = match.group(2)   # e.g. "Thesis Project"

            # Collect consecutive same-level headers
            titles = [title]
            j = i + 1
            while j < len(lines):
                next_line = lines[j]
                if not next_line.strip():
                    j += 1
                    continue  # skip blank lines between headers
                next_match = header_pattern.match(next_line)
                if next_match and next_match.group(1) == level:
                    titles.append(next_match.group(2))
                    j += 1
                else:
                    break

            # Emit a single concatenated header
            concatenated = f"{level} {' | '.join(titles)}"
            result.append(concatenated)
            i = j  # skip past all consumed headers

            # Check if this concatenated header also has no content
            next_content = next(
                (lines[k] for k in range(i, len(lines)) if lines[k].strip()),
                None
            )
            if next_content is None or header_pattern.match(next_content):
                result.append("<!-- header-only -->")

        else:
            result.append(line)
            i += 1

    return "\n".join(result)


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

# Matches a full markdown table: one or more pipe rows, then a separator row, then data rows
pipe_row = r'(?:\|.+\|\n)'
separator_row = r'\|[-| :]+\|\n'
data_rows = r'(?:\|.+\|\n|\n)*'   # pipe line OR bare newline (blank line)

grp = rf'{pipe_row}+{separator_row}{data_rows}'

TABLE_PATTERN = re.compile(
    rf'(?P<table>{grp})',
    re.MULTILINE
)


def _count_tokens(text: str, length_function) -> int:
    ''' Returns the length according to the length function of the given text.'''
    return length_function(text)


def _get_caption(text_before_table: str) -> str:
    """
    Extract the last non-empty line before the table as the caption.
    """
    lines = text_before_table.rstrip("\n").split("\n") ###################
    for line in reversed(lines):
        stripped = line.strip()
        if stripped:
            return stripped
    return ""


def _parse_table(table_text: str) -> tuple[list[str], list[list[str]]]:
    """
    Parse a markdown table into (header_rows, data_rows).
    Header rows are all rows before and including the separator row (|---|...).
    Returns raw line strings to preserve original formatting.
    """
    lines = [l for l in table_text.strip().split("\n") if l.strip()] # l.strip removes trailing spaces (and the if condition check that it's a line)
    
    separator_idx = None
    for i, line in enumerate(lines):
        # | `^` | Start of string (or line with `re.MULTILINE`) |
        # | `$` | End of string (or line with `re.MULTILINE`) | 
        #  \| is a special regex character for a tube 
        # [-| :]+ any of contained chars, one or more of them!!

        # Detect separator row
        if re.match(r'^\|[-| :]+\|$', line.strip()):
            separator_idx = i
            break
    ## end for
    if separator_idx is None:
        # No separator found, treat entire table as header (won't be split)
        return lines, []
    
    header_rows = lines[:separator_idx + 1]  # includes the --- row
    data_rows = lines[separator_idx + 1:]
    return header_rows, data_rows


def _build_table_text(header_rows: list[str], data_rows: list[str]) -> str:
    return "\n".join(header_rows + data_rows) + "\n"


def split_large_table(
    table_text: str,
    caption: str,
    metadata: dict,
    budget: int, # allowance length
    length_function,
) -> list[Document]:
    """
    Split a markdown table that exceeds `budget` by grouping rows into
    sub-chunks, each prefixed with the caption and column headers.

    Each sub-chunk looks like:
        <caption>
        | col1 | col2 | ...
        | ---  | ---  | ...
        | row1 | ...
        | row2 | ...
    """
    header_rows, data_rows = _parse_table(table_text)
    header_block = "\n".join(header_rows) + "\n"
    caption_line = caption + "\n" if caption else ""
    prefix = caption_line + header_block  # prepended to every sub-chunk
    prefix_tokens = _count_tokens(prefix, length_function)

    chunks: list[Document] = []
    current_rows: list[str] = []
    current_tokens = prefix_tokens

    for row in data_rows:
        row_line = row + "\n"
        row_tokens = _count_tokens(row_line, length_function)

        # If a single row is already over budget, emit it alone (can't split further)
        if prefix_tokens + row_tokens > budget:
            if current_rows:
                chunk_text = prefix + "".join(r + "\n" for r in current_rows)
                chunks.append(Document(page_content=chunk_text, metadata=dict(metadata)))
                current_rows = []
                current_tokens = prefix_tokens
            # Emit the oversized row on its own
            chunk_text = prefix + row_line
            chunks.append(Document(page_content=chunk_text, metadata=dict(metadata)))
            continue

        if current_tokens + row_tokens > budget:
            # Flush current batch
            chunk_text = prefix + "".join(r + "\n" for r in current_rows)
            chunks.append(Document(page_content=chunk_text, metadata=dict(metadata)))
            current_rows = [row]
            current_tokens = prefix_tokens + row_tokens
        else:
            current_rows.append(row)
            current_tokens += row_tokens

    if current_rows:
        chunk_text = prefix + "".join(r + "\n" for r in current_rows)
        chunks.append(Document(page_content=chunk_text, metadata=dict(metadata)))

    return chunks


def split_chunk_with_table_awareness(
    doc: Document,
    budget: int,
    length_function,
    char_splitter,
) -> list[Document]:
    """
    For a single header-split chunk (Document), detect markdown tables and:
      - Keep tables that fit within `budget` as atomic chunks (with caption prepended).
      - Row-split tables that exceed `budget`, repeating headers on each sub-chunk.
      - Split non-table text segments normally via `char_splitter`.

    Args:
        doc:              A Document from MarkdownHeaderTextSplitter.
        budget:           Max tokens (or chars) per chunk — same unit as length_function.
        length_function:  Callable(str) -> int, e.g. token counter or len.
        char_splitter:    Your RecursiveCharacterTextSplitter instance (for prose segments).

    Returns:
        List of Documents, all inheriting `doc.metadata`.
    """
    text = doc.page_content #chunk.page_content is the text. 
    metadata = doc.metadata
    result: list[Document] = [] # doc list

    last_end = 0
    # finiter is a re object
    for match in TABLE_PATTERN.finditer(text): # when table detected
        start, end = match.start(), match.end() # match is re-type object
        # --- Prose segment before the table ---
        prose_before = text[last_end:start] #this is from chunk start to start of table, and then from end of last detected table to start of new
        if prose_before.strip(): #
            sub_docs = char_splitter.create_documents(
                [prose_before], metadatas=[metadata]
            )
            result.extend(sub_docs) # append would give nested list if a document is split into more than 1 chunk

        # --- Caption: last non-empty line before this table ---
        caption = _get_caption(text[last_end:start])

        # --- Table itself ---
        table_text = match.group("table")
        full_table_with_caption = (caption + "\n" if caption else "") + table_text
        table_tokens = _count_tokens(full_table_with_caption, length_function) # evaluate length

        if table_tokens <= budget:
            # Fits whole: emit as single atomic chunk
            result.append(
                Document(page_content=full_table_with_caption, metadata=dict(metadata))
            )
        else:
            # Too large: row-split with header repetition
            sub_docs = split_large_table(
                table_text=table_text,
                caption=caption,
                metadata=metadata,
                budget=budget,
                length_function=length_function,
            )
            result.extend(sub_docs)

        last_end = end

    # --- Remaining prose after the last table ---
    prose_after = text[last_end:]
    if prose_after.strip():
        sub_docs = char_splitter.create_documents(
            [prose_after], metadatas=[metadata]
        )
        result.extend(sub_docs)

    return result


def chunk_document(
    header_chunks,
    char_splitter,
    budget: int,
    length_function,
) -> list[Document]:
    """
    Full pipeline: header split → table-aware char split.
    """
    all_chunks: list[Document] = []
    
    for chunk in header_chunks:
        chunk_tokens = _count_tokens(chunk.page_content, length_function)
        IS_TABLE = TABLE_PATTERN.search(chunk.page_content)
        if chunk_tokens <= budget and not IS_TABLE:
            # Small chunk, no table: pass through as-is
            all_chunks.append(chunk)
        else:
            # May contain a table or is too large: go through table-aware splitter
            sub_chunks = split_chunk_with_table_awareness(
                doc=chunk,
                budget=budget,
                length_function=length_function,
                char_splitter=char_splitter,
            )
            all_chunks.extend(sub_chunks)

    return all_chunks



########################################
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
