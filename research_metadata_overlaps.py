import pickle
import pandas as pd
import os
import re
from collections import Counter

def get_canonical_key(text):
    """Extracts 'part i' or 'item 7' from a string for lenient matching."""
    if not text or not isinstance(text, str):
        return None
    text = text.lower().strip()
    
    # Match "item 1a", "item 7", etc.
    item_match = re.match(r"(item\s+\d+[a-z]?)", text)
    if item_match:
        return item_match.group(1)
        
    # Match "part i", "part ii", etc.
    part_match = re.match(r"(part\s+[ivx]+)", text)
    if part_match:
        return part_match.group(1)
    return text[:20] # Fallback to first 20 chars

def analyze_metadata_overlaps(file_paths):
    all_rows = []
    
    print(f"--- STARTING METADATA EXTRACTION ---")
    print(f"Targeting {len(file_paths)} files.\n")

    for path in file_paths:
        if not os.path.exists(path):
            print(f"File not found: {path}")
            continue
            
        with open(path, "rb") as f:
            df = pickle.load(f)
            
        file_name = os.path.basename(path)
        print(f"Processing: {file_name} ({len(df)} chunks)")

        # Extract metadata fields into a list of dicts
        for _, row in df.iterrows():
            md = row.get('metadata', {})
            all_rows.append({
                "file": file_name,
                "section": md.get('section'),
                "subsection": md.get('subsection'),
                "item": md.get('item')
            })

    # Create the consolidated DataFrame
    metadata_df = pd.DataFrame(all_rows)
    
    print(f"\n--- EXTRACTION COMPLETE ---")
    print(f"Total metadata rows extracted: {len(metadata_df)}")
    print("\nDataFrame Head:")
    print(metadata_df.head(10))
    
    # Basic stats on non-null values
    print("\nMetadata Coverage:")
    print(metadata_df.notna().sum())

    return metadata_df

if __name__ == "__main__":
    BASE_DIR = "/Users/nadavsmacbookair/Desktop/Thesis/data/financial_corpora/chunks/indexed-at-13-06-26/headers_split"
    target_files = [
        os.path.join(BASE_DIR, "AMZN_10K_2020.pkl"),
        os.path.join(BASE_DIR, "KO_10K_2024.pkl"),
        os.path.join(BASE_DIR, "JPM_10K_2025.pkl"),
        os.path.join(BASE_DIR, "WMT_10K_2022.pkl")
    ]
    
    df = analyze_metadata_overlaps(target_files)
    df.to_excel("metadata_analysis.xlsx")