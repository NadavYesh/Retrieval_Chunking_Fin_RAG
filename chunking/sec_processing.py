#%%
import re
import sys
# import local sec2md 
from pathlib import Path
sys.path.append("/Users/nadavsmacbookair/Documents/sec2md/src")
import sec2md 
from typing import NamedTuple, Optional
from langchain_core.documents import Document
from langchain_text_splitters import MarkdownHeaderTextSplitter


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



TOC_SEGMENT_RE = re.compile(r"^\*{0,2}\s*table of contents\s*\*{0,2}$", re.I)


def strip_toc_running_headers(markdown: str) -> str:
    """
    Drop the "Table of Contents" running-page-header artifact from header
    lines. SEC filings repeat this text at the top of every page, and
    sec2md/enrich_md_text end up emitting it as its own header segment
    (standalone, or piped alongside the page's real title, e.g.
    "### Table of Contents | Our Strategy"). Left in place, it pollutes the
    'item'/'subitem' metadata used downstream for embedding with a string
    that recurs on nearly every page, exactly the kind of non-discriminative
    boilerplate that metadata is meant to avoid.

    Segments are filtered individually (splitting on the same " | " that
    inject_header_placeholders joins consecutive headers with), so this
    catches both the standalone and the piped-with-a-real-title cases.
    A header left with no remaining title after filtering is dropped
    entirely, so its content simply continues under the previous section
    instead of anchoring a spuriously titled chunk.
    """
    header_re = re.compile(r"^(#{1,4})\s+(.*)$")
    out_lines = []
    for line in markdown.split("\n"):
        match = header_re.match(line)
        if not match:
            out_lines.append(line)
            continue

        level, title = match.group(1), match.group(2)
        kept = [seg.strip() for seg in title.split("|") if not TOC_SEGMENT_RE.match(seg.strip())]
        if kept:
            out_lines.append(f"{level} {' | '.join(kept)}")
        # else: drop the line entirely — pure running-header noise.

    return "\n".join(out_lines)


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

    # MarkdownHeaderTextSplitter strips blank lines and rejoins the surrounding
    # lines with a Markdown hard-break ("  \n") wherever a blank line used to be.
    # This corrupts table rows immediately followed by prose (the blank line
    # that used to separate them becomes part of the row's own line ending),
    # breaking TABLE_PATTERN's `\|...\|\n` matches downstream. Restore the
    # original blank-line paragraph/table boundaries before returning.
    #
    # It also never adds a trailing newline to the section's last line, so a
    # table that ends a header-chunk has an unterminated final row — which
    # also fails TABLE_PATTERN's `\|...\|\n` row match. Ensure every chunk
    # ends with a newline to close out that last row.
    for chunk in header_chunks:
        chunk.page_content = re.sub(r' {2,}\n', '\n\n', chunk.page_content)
        if chunk.page_content and not chunk.page_content.endswith('\n'):
            chunk.page_content += '\n'

    return _merge_header_only_chunks(header_chunks)


HEADER_ONLY_MARKER = "<!-- header-only -->"


def _merge_header_metadata(orphan_meta: dict, next_meta: dict) -> dict:
    """
    Fold an orphaned (header-only) chunk's metadata onto the metadata of the
    chunk that follows it: shared keys with differing values are pipe-joined
    (orphan's title first, since it comes first in the document), keys unique
    to the orphan are carried over as-is, and identical values are left alone.
    """
    merged = dict(next_meta)
    for key, value in orphan_meta.items():
        if key not in next_meta:
            merged[key] = value
        elif next_meta[key] != value:
            merged[key] = f"{value} | {next_meta[key]}"
    return merged


def _merge_header_only_chunks(header_chunks: list) -> list:
    """
    inject_header_placeholders leaves a `<!-- header-only -->` marker chunk
    wherever a header has no content before the next header — needed so the
    header's title survives the split at all, but it ships downstream as a
    chunk with no real text. Instead of emitting it as its own empty chunk,
    fold its metadata forward onto the next (real) chunk and drop it. A run
    of several header-only chunks in a row accumulates in document order
    before being folded onto the first chunk with real content; a run that
    reaches the end of the document with no real chunk to attach to is
    dropped, since there is nothing left to attach its title to.
    """
    merged: list = []
    pending_meta: Optional[dict] = None

    for chunk in header_chunks:
        if chunk.page_content.strip() == HEADER_ONLY_MARKER:
            pending_meta = (
                chunk.metadata if pending_meta is None
                else _merge_header_metadata(pending_meta, chunk.metadata)
            )
            continue

        if pending_meta is not None:
            chunk.metadata = _merge_header_metadata(pending_meta, chunk.metadata)
            pending_meta = None
        merged.append(chunk)

    return merged

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
    Currently unused: tables are now always kept atomic (see `_pack_segments`),
    never row-split, even when they exceed `budget`. Kept unreferenced for a
    possible future ablation comparing row-split vs. atomic table chunking.

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


class _Segment(NamedTuple):
    """
    An ordered unit of a header chunk's content, produced by `_segment_header_chunk`.

    kind:           "prose", "table", or "run_header".
    text:           The raw text of this unit. A "table" unit is never split
                     internally, no matter its size. For "run_header", this is
                     just the label itself (asterisks stripped), not a full
                     paragraph.
    caption_source: Only meaningful for kind == "table" — the raw, unsplit prose
                     that immediately preceded this table in the original text
                     (same input `_get_caption` uses). "" if none preceded it.
    """
    kind: str
    text: str
    caption_source: str = ""


def _split_into_paragraphs(text: str) -> list[str]:
    """Split prose text into paragraph-level units on blank lines."""
    return [p for p in re.split(r'\n{2,}', text.strip('\n')) if p.strip()]


# A standalone `*Label*` line, e.g. "*Americas*" or "*iPhone*" — SEC filers use
# this single-asterisk (italic) pattern as a lightweight run-in subheading
# beneath a `###`/`####` header, one level deeper than the Markdown header
# hierarchy MarkdownHeaderTextSplitter already captures (e.g. per-geography or
# per-product breakdowns under a single "Segment Operating Performance" item).
RUN_HEADER_PATTERN = re.compile(r'^\*(?!\*)([^\n*]{1,80})\*$')


def _is_run_header_candidate(title: str) -> bool:
    """
    The same single-asterisk line pattern is also used for footnotes (e.g.
    "*See accompanying notes.*", "*1 Some disclaimer.*"), which read as full
    sentences rather than titles. Filter those out: real run-in subheadings
    in this corpus are short labels with no leading digit and no
    sentence-ending period; footnotes reliably have one or both.

    Also reject the recurring "Table of Contents" running-page-header
    artifact (TOC_SEGMENT_RE — see strip_toc_running_headers above). That
    should already be stripped out of `#`-level headers before this stage
    ever runs, but this guards against it also surfacing as a standalone
    `*Table of Contents*` paragraph in chunk body text (e.g. from a stale
    corpus snapshot predating that fix) and getting embedded as if it were
    a genuine, discriminative subheading.
    """
    title = title.strip()
    if not title or title[0].isdigit() or title.endswith('.'):
        return False
    return not TOC_SEGMENT_RE.match(title)


def _paragraph_to_segment(para: str) -> _Segment:
    match = RUN_HEADER_PATTERN.match(para.strip())
    if match and _is_run_header_candidate(match.group(1)):
        return _Segment("run_header", match.group(1).strip())
    return _Segment("prose", para)


def _segment_header_chunk(text: str) -> list[_Segment]:
    """
    Break a header chunk's text into an ordered sequence of paragraph-level
    prose segments, atomic table segments (via TABLE_PATTERN), and run-header
    segments (via RUN_HEADER_PATTERN), preserving document order so they can
    be greedily packed back together.
    """
    segments: list[_Segment] = []
    last_end = 0

    for match in TABLE_PATTERN.finditer(text):
        start, end = match.start(), match.end()
        prose_before = text[last_end:start]
        for para in _split_into_paragraphs(prose_before):
            segments.append(_paragraph_to_segment(para))
        segments.append(_Segment("table", match.group("table"), caption_source=prose_before))
        last_end = end

    for para in _split_into_paragraphs(text[last_end:]):
        segments.append(_paragraph_to_segment(para))

    return segments


def _pack_segments(
    segments: list[_Segment],
    metadata: dict,
    budget: int,
    length_function,
    char_splitter,
) -> list[Document]:
    """
    Greedily pack an ordered sequence of prose/table segments into chunks up
    to `budget`, merging prose and tables from the same header chunk whenever
    they fit together (instead of always isolating every table), and never
    splitting a table internally — even one that exceeds `budget` alone.

    A "run_header" segment (e.g. "*Americas*") never merges with what came
    before it — it always starts a fresh chunk — and its label is attached as
    metadata["run_header"] (not duplicated into the chunk text, since it's
    already embedded via fmt_title) to every chunk produced from that point
    onward, until the next run-header segment replaces it. This keeps a
    token-budget boundary from silently blending two unrelated subsections
    together (e.g. the tail of an "Americas" breakdown with the start of
    "Europe"'s) purely because they happened to fit together, and gives each
    resulting chunk the specific, contextual label instead of just the
    section's overall header.
    """
    result: list[Document] = []
    buffer: list[str] = []
    current_metadata = dict(metadata)

    def flush():
        if buffer:
            result.append(Document(page_content="\n\n".join(buffer), metadata=dict(current_metadata)))
            buffer.clear()

    for seg in segments:
        if seg.kind == "run_header":
            # Flush what came before, then only update the label going
            # forward — don't seed the buffer with it. It's already carried
            # as metadata["run_header"] (embedded via fmt_title), so leaving
            # it out of the text avoids duplicating it there too.
            flush()
            current_metadata = {**metadata, "run_header": seg.text}
            continue

        bare_tokens = _count_tokens(seg.text, length_function)

        if bare_tokens > budget:
            # Doesn't fit even alone -> can't fit merged either.
            flush()
            if seg.kind == "table":
                caption = _get_caption(seg.caption_source) if seg.caption_source else ""
                final_text = f"{caption}\n{seg.text}" if caption else seg.text
                result.append(Document(page_content=final_text, metadata=dict(current_metadata)))
            else:
                result.extend(char_splitter.create_documents([seg.text], metadatas=[current_metadata]))
            continue

        if buffer:
            candidate = "\n\n".join(buffer + [seg.text])
            if _count_tokens(candidate, length_function) <= budget:
                buffer.append(seg.text)
                continue
            flush()

        # Buffer is empty here (either started empty, or a merge attempt just
        # failed and got flushed) -> seed a fresh buffer with this segment.
        seed = seg.text
        if seg.kind == "table" and seg.caption_source:
            caption = _get_caption(seg.caption_source)
            if caption:
                captioned = f"{caption}\n{seg.text}"
                if _count_tokens(captioned, length_function) <= budget:
                    seed = captioned
        buffer.append(seed)

    flush()
    return result


def pack_prose_and_tables(
    doc: Document,
    budget: int,
    length_function,
    char_splitter,
) -> list[Document]:
    """
    For a single header-split chunk (Document), pack its paragraphs and tables
    into child chunks up to `budget`, keeping adjacent prose context (lead-in
    sentence, trailing explanation) together with a table whenever it fits,
    instead of always carving every table out into its own isolated chunk.
    A table is never split internally, even if it alone exceeds `budget`.

    Args:
        doc:              A Document from MarkdownHeaderTextSplitter.
        budget:           Max tokens (or chars) per chunk — same unit as length_function.
        length_function:  Callable(str) -> int, e.g. token counter or len.
        char_splitter:    Your RecursiveCharacterTextSplitter instance (for oversized prose).

    Returns:
        List of Documents, all inheriting `doc.metadata`.
    """
    segments = _segment_header_chunk(doc.page_content)
    return _pack_segments(segments, doc.metadata, budget, length_function, char_splitter)


def chunk_document(
    header_chunks,
    char_splitter,
    budget: int,
    length_function,
) -> list[Document]:
    """
    Full pipeline: header split → table-aware char split.

    If a header chunk has metadata["_id"] set (by the hierarchical pipeline),
    that value is propagated as metadata["parent_id"] on every child chunk.
    """
    all_chunks: list[Document] = []

    for chunk in header_chunks:
        parent_id = chunk.metadata.get("_id")
        chunk_tokens = _count_tokens(chunk.page_content, length_function)
        if chunk_tokens <= budget:
            sub_chunks = [chunk]
        else:
            sub_chunks = pack_prose_and_tables(
                doc=chunk,
                budget=budget,
                length_function=length_function,
                char_splitter=char_splitter,
            )

        if parent_id:
            for sc in sub_chunks:
                sc.metadata["parent_id"] = parent_id
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
