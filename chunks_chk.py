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
                content = f.read(10000)
    # If it was a file, 'content' is now the string. 
    # If not a file, 'content' remains the input string/data.

    # Ensure we have a string for regex
    if not isinstance(content, str):
        content = str(content)
    
    content = content[:10000]

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
    fiscal_match = re.search(r'fiscal year ended\s+([A-Za-z]+)\s+(\d{1,2})\s*,\s*(\d{4})', content, re.IGNORECASE)
    if fiscal_match:
        month_name = fiscal_match.group(1).lower()
        day = int(fiscal_match.group(2))
        year = fiscal_match.group(3)[2:] # Get last 2 digits of year
        months = {'january': '01', 'february': '02', 'march': '03', 'april': '04', 'may': '05', 'june': '06', 
                  'july': '07', 'august': '08', 'september': '09', 'october': '10', 'november': '11', 'december': '12'}
        month_num = months.get(month_name)
        if month_num:
            results["fiscal_year_end"] = f"{month_num}-{day:02d}-{year}"

    # 3. Company Name
    # Look for name right before address or incorporate/organization
    # Example: ### BEST BUY CO., INC.\n(Exact name of registrant...)
    # Example: ### 3M COMPANY\nState of Incorporation
    name_match = re.search(r'###\s+([^\n#]+)\n+(?:\(Exact name|State of Incorporation|I\.R\.S\.)', content, re.IGNORECASE)
    if name_match:
        results["company_name"] = name_match.group(1).strip()

    # 4. Ticker
    # Look in the table: | Title of each class | Trading Symbol |
    # Optimized to handle various table layouts in markdown
    ticker_pattern = re.compile(r'Trading Symbol[^\n]+\n\|(?:\s*:?---:?\s*\|)+\n\|[^|]+\|\s*([A-Za-z0-9]+)\s*\|', re.IGNORECASE)
    ticker_match = ticker_pattern.search(content)
    if ticker_match:
        results["ticker"] = ticker_match.group(1).strip()
    
    # Fallback for ticker: "The Company's ticker symbol is MMM."
    if not results["ticker"]:
        ticker_fb = re.search(r"ticker symbol is\s+([A-Z0-9]+)", content, re.IGNORECASE)
        if ticker_fb:
            results["ticker"] = ticker_fb.group(1).strip()

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