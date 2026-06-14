#%%
import re
import os
import pickle
import pandas as pd

def get_meta_sec(content):
    """
    Parses a markdown SEC filing to extract meta fields.
    'content' can be the raw string content or a path to a .md/.pkl file.
    """
    if isinstance(content, str) and os.path.isfile(content):
        if content.endswith(".pkl"):
            with open(content, "rb") as f:
                content = pickle.load(f)
        else:
            with open(content, "r", encoding="utf-8") as f:
                content = f.read(100000) # Increased to read more
    
    # If it's a DataFrame (common for our chunked files), join the first 100 chunks
    if isinstance(content, pd.DataFrame):
        if 'text' in content.columns:
            content = "\n".join(content['text'].iloc[:100].astype(str))
        else:
            content = str(content)
    elif not isinstance(content, str):
        content = str(content)
    
    content = content[:200000] # Increased limit for regex

    results = {
        "form_type": None,
        "company_name": None,
        "ticker": None,
        "fiscal_year_end": None
    }

    # 1. Form Type
    form_match = re.search(r'FORM\s+(10-K(?:/A)?)', content, re.IGNORECASE)
    if form_match:
        results["form_type"] = form_match.group(1).upper()

    # 2. Fiscal Year End
    fiscal_match = re.search(r'fiscal year[- ]ende?d?(?:\s+of)?\s+([A-Za-z]+)\s+(\d{1,2})\s*,\s*(\d{4})', content, re.IGNORECASE)
    if fiscal_match:
        month_name = fiscal_match.group(1).lower()
        day = int(fiscal_match.group(2))
        year = fiscal_match.group(3)[2:] # Get last 2 digits of year
        months = {
            'january': 1, 'february': 2, 'march': 3, 'april': 4,
            'may': 5, 'june': 6, 'july': 7, 'august': 8,
            'september': 9, 'october': 10, 'november': 11, 'december': 12
        }
        month_num = months.get(month_name)
        if month_num:
            results["fiscal_year_end"] = f"{month_num:02d}-{day:02d}-{year}"

    # 3. Company Name
    # Priority 1: Look for name before "Exact name" or "State of Incorporation"
    name_pattern = r'(?:\*\*|###)?\s*([^\n#\*]{3,100})\s*(?:\*\*|###)?\s*\n+(?:\*\*|###)?\s*(?:\(?Exact name|State of Incorporation|I\.R\.S\.)'
    name_match = re.search(name_pattern, content, re.IGNORECASE)
    if name_match:
        name = name_match.group(1).strip()
        # Validation: If it looks like a form title, it's likely not the company name
        if not any(word in name.upper() for word in ["REPORT", "PURSUANT", "SECTION", "ACT OF", "FORM"]):
            results["company_name"] = name

    # Priority 2: Fallback for "was incorporated in" (common in 3M files)
    if not results["company_name"]:
        inc_match = re.search(r'([^\n#\*]{3,100}?)\s+(?:was )?incorporated in\s+\d{4}', content)
        if inc_match:
            results["company_name"] = inc_match.group(1).strip()

    # 4. Ticker
    # Look in the table: | Title of each class | Trading Symbol |
    ticker_pattern = re.compile(r'Trading Symbol[^\n]+\n\|(?:\s*:?---:?\s*\|)+\n\|[^|]+\|\s*([A-Za-z0-9]+)\s*\|', re.IGNORECASE)
    ticker_match = ticker_pattern.search(content)
    if ticker_match:
        results["ticker"] = ticker_match.group(1).strip()
    
    # Fallback for ticker
    if not results["ticker"]:
        # Fallback 1: "ticker symbol is MMM"
        ticker_fb1 = re.search(r"ticker symbol is\s+([A-Z0-9]+)", content, re.IGNORECASE)
        if ticker_fb1:
            results["ticker"] = ticker_fb1.group(1).strip()
        
        # Fallback 2: "symbol “ADBE”" or "symbol ADBE" or "under the symbol PYPL"
        if not results["ticker"]:
            ticker_fb2 = re.search(r"(?:under the\s+)?symbol\s+[“\"]?([A-Z0-9]+)[”\"]?", content, re.IGNORECASE)
            if ticker_fb2:
                results["ticker"] = ticker_fb2.group(1).strip()

    # Clean and normalize company name
    if results["company_name"]:
        name = results["company_name"]
        # Remove legal suffixes and punctuation
        # This handles: , Inc. , Corp. , L.P. , LLC, /DE, etc.
        name = re.sub(r'[,.\/]+', ' ', name) # Replace punctuation/slashes with space
        # Remove common legal entities and "COMPANY"
        name = re.sub(r'\b(?:INC|CORP|CORPORATION|LIMITED|LTD|HOLDINGS|SYSTEMS|INCORPORATED|LLC|LP|PLC|COMPANY)\b.*', '', name, flags=re.IGNORECASE)
        name = re.sub(r'\s+', ' ', name).strip() # Collapse whitespace
        results["company_name"] = name

    # Ensure all metadata fields are lowercase
    for key in results:
        if isinstance(results[key], str):
            results[key] = results[key].lower()

    return results


def sec_metadata(chunks_input, doc_metadata):
    """
    Injects document-level metadata to the beginning of all chunk metadata attributes.
    'chunks_input' can be a pandas DataFrame or a path to a .pkl file.
    """
    if isinstance(chunks_input, str) and os.path.isfile(chunks_input):
        with open(chunks_input, "rb") as f:
            chunks_df = pickle.load(f)
    elif isinstance(chunks_input, pd.DataFrame):
        chunks_df = chunks_input.copy()
    else:
        chunks_df = chunks_input # Fallback for other types

    def merge_meta(chunk_meta):
        new_meta = doc_metadata.copy()
        if isinstance(chunk_meta, dict):
            new_meta.update(chunk_meta)
        return new_meta

    chunks_df['metadata'] = chunks_df['metadata'].apply(merge_meta)
    return chunks_df
# %%
