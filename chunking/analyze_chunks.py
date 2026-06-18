import os
import pickle
import pandas as pd

def chunk_to_excel(chunks: []):
    for chunk in chunks:
        
        chunk.to_excel

def run_statistics(directory, save_excel=[]):
    print(f"{'Filename':<30} | {'Total':<6} | {'Sec':<4} | {'Sub':<4} | {'Item':<4} | {'S-Itm':<5}")
    print("-" * 65)
    
    files = [f for f in os.listdir(directory) if f.endswith(".pkl")]
    if not files:
        print(f"No files found in {directory}")
        return

    for filename,save_ in zip(sorted(files),save_excel):
        path = os.path.join(directory, filename)
        try:
            with open(path, "rb") as f:
                df = pickle.load(f)
            if save_:
                
                df.to_excel(f"/Users/nadavsmacbookair/Desktop/Thesis/code/chunking/data{filename[:-4]}.xlsx")

            if not isinstance(df, pd.DataFrame) or 'metadata' not in df.columns:
                continue

            total = len(df)
            # Count occurrences of metadata keys
            sec = df['metadata'].apply(lambda x: 'section' in x if isinstance(x, dict) else False).sum()
            sub = df['metadata'].apply(lambda x: 'subsection' in x if isinstance(x, dict) else False).sum()
            itm = df['metadata'].apply(lambda x: 'item' in x if isinstance(x, dict) else False).sum()
            s_itm = df['metadata'].apply(lambda x: 'subitem' in x if isinstance(x, dict) else False).sum()

            # doc-level metadata
            first_meta = df['metadata'].iloc[0] if not df.empty and isinstance(df['metadata'].iloc[0], dict) else {}
            tick = first_meta.get('ticker')
            comp_name = first_meta.get('company_name')
            fis_year = first_meta.get('fiscal_year_end')

            # Flag outliers (e.g., 1 chunk usually means a split failed)
            alert = "!!" if total <= 1 else ""
            
            print(f"{filename[:30]:<30} | Ticker:{tick} | Company Name: {comp_name} | Fiscal Year End: {fis_year} |  {total:<6} | {sec:<4} | {sub:<4} | {itm:<4} | {s_itm:<5} {alert}")

        except Exception as e:
            print(f"Error reading {filename}: {e}")

if __name__ == "__main__":
    #TARGET_DIR = "/Users/nadavsmacbookair/Desktop/Thesis/data/financial_corpora/chunks/indexed-at-16-06-26/headers_split"
    TARGET_DIR = "/Users/nadavsmacbookair/Desktop/Thesis/data/financial_corpora/chunks/indexed-at-17-06-26/headers_chars_split"
    if os.path.exists(TARGET_DIR):
        save_excel = [False for _ in os.listdir(TARGET_DIR)]
        save_excel[6] = True
        run_statistics(TARGET_DIR, save_excel=save_excel)
    else:
        print(f"Path not found: {TARGET_DIR}")


#%%
