import re
from typing import Optional
from langchain_core.documents import Document


# Matches a full markdown table: one or more pipe rows, then a separator row, then data rows
TABLE_PATTERN = re.compile(
    r'(?P<table>(?:\|.+\|\n)+\|[-| :]+\|\n(?:\|.+\|\n)*)',
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

        # Separator row: only dashes, pipes, colons, spaces
        if re.match(r'^\|[-| :]+\|$', line.strip()):
            separator_idx = i
            break
    
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
    budget: int,
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
    text = doc.page_content
    metadata = doc.metadata
    result: list[Document] = []

    last_end = 0
    for match in TABLE_PATTERN.finditer(text): # when table detected
        start, end = match.start(), match.end()

        # --- Prose segment before the table ---
        prose_before = text[last_end:start]
        if prose_before.strip():
            sub_docs = char_splitter.create_documents(
                [prose_before], metadatas=[metadata]
            )
            result.extend(sub_docs)

        # --- Caption: last non-empty line before this table ---
        caption = _get_caption(text[last_end:start])

        # --- Table itself ---
        table_text = match.group("table")
        full_table_with_caption = (caption + "\n" if caption else "") + table_text
        table_tokens = _count_tokens(full_table_with_caption, length_function)

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
    doc_text: str,
    header_splitter,
    char_splitter,
    budget: int,
    length_function,
) -> list[Document]:
    """
    Full pipeline: header split → table-aware char split.
    """
    header_chunks = header_splitter.split_text(doc_text)
    all_chunks: list[Document] = []

    for chunk in header_chunks:
        chunk_tokens = _count_tokens(chunk.page_content, length_function)

        if chunk_tokens <= budget and not TABLE_PATTERN.search(chunk.page_content):
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