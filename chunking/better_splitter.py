#%%
from langchain_text_splitters import ExperimentalMarkdownSyntaxTextSplitter,MarkdownHeaderTextSplitter
import pandas as pd
with open("/Users/nadavsmacbookair/Desktop/Thesis/data/financial_corpora/md/indexed-at-13-06-26/KO_10K_2024.md",'r') as f:
    md_text = f.read()
#%%

splitter = MarkdownHeaderTextSplitter(
    headers_to_split_on=[
            ("#", "section"),
            ("##", "subsection"),
            ("###", "item"),
            ("####", "subitem")],
            return_each_line=False, 
            strip_headers=False) # if I do no, what will happen?

####### 
# number of chunks increased
# it prepends the lower hierarchy meta item (by number of '#') to the text
# it increases the number of chunks due to empty sections. 
# ####### 

chunks = splitter.split_text(md_text)
chunks_pd = pd.DataFrame(chunks)
chunks_pd.to_excel("new_splits.xlsx")
#%%
splitter_old = MarkdownHeaderTextSplitter(
    headers_to_split_on=[
            ("#", "section"),
            ("##", "subsection"),
            ("###", "item"),
            ("####", "subitem")]
            #return_each_line=False,# these are the defaults. 
            #trip_headers=True
            )

chunks_old = splitter_old.split_text(md_text)

chunks_old_pd = pd.DataFrame(chunks_old)
chunks_old_pd.to_excel("old_splits.xlsx")


#%%
# 
import os
import pickle
import pandas as pd

def analyze_chunk_statistics(base_dir):
    """
    Analyzes chunk statistics for all pickle files in a given directory.
    For each file, it reports the total number of chunks and the counts of
    chunks containing 'section', 'subsection', 'item', and 'subitem' metadata.
    """
    print(f"--- Starting chunk statistics analysis for directory: {base_dir} ---")
    
    all_files = [f for f in os.listdir(base_dir) if f.endswith(".pkl")]
    if not all_files:
        print(f"No .pkl files found in {base_dir}")
        return

    for filename in all_files:
        file_path = os.path.join(base_dir, filename)
        try:
            with open(file_path, "rb") as f:
                df = pickle.load(f)
            
            if not isinstance(df, pd.DataFrame) or 'metadata' not in df.columns:
                print(f"Skipping {filename}: Not a valid DataFrame or missing 'metadata' column.")
                continue

            total_chunks = len(df)
            section_count = df['metadata'].apply(lambda x: 'section' in x if isinstance(x, dict) else False).sum()
            subsection_count = df['metadata'].apply(lambda x: 'subsection' in x if isinstance(x, dict) else False).sum()
            item_count = df['metadata'].apply(lambda x: 'item' in x if isinstance(x, dict) else False).sum()
            subitem_count = df['metadata'].apply(lambda x: 'subitem' in x if isinstance(x, dict) else False).sum()

            print(f"\nFile: {filename}")
            print(f"  Total Chunks: {total_chunks}")
            print(f"  Chunks with 'section' metadata: {section_count}")
            print(f"  Chunks with 'subsection' metadata: {subsection_count}")
            print(f"  Chunks with 'item' metadata: {item_count}")
            print(f"  Chunks with 'subitem' metadata: {subitem_count}")

        except Exception as e:
            print(f"Error processing {filename}: {e}")
    print("\n--- Analysis Complete ---")

BASE_DIR = "/Users/nadavsmacbookair/Desktop/Thesis/data/financial_corpora/chunks/indexed-at-13-06-26/headers_split"
analyze_chunk_statistics(BASE_DIR)

########FUCKED UP CHUNKS:
# too little chunks: TSLA_10K_2023.pkl, INTC_10K_2024.pkl,INTC_10K_2025.pkl,INTC_10K_2021.pkl,INTC_10K_2020.pkl,  INTC_10K_2022.pkl, INTC_10K_2023.pkl,
# BBY_10K_2020.pkl, BBY_10K_2021.pkl,BBY_10K_2023.pkl,BBY_10K_2022.pkl,BBY_10K_2024.pkl
# example of good:
# 
# proposed solution 
# split like this: 
#             ("#", "section"),
#             ("##", "subsection"),
#             ("***", "item"),
#             ("###", "item"),
#             ("####", "subitem")\
# or keep the header split the same, but add 
#               custom_header_patterns({"***":3})
#  ###

# ############ 















# the problem of missed context due to stripping of the 
# headers SHOULD:) be dealt with in the embedding process


#%%
# print chunks statistics for all chunks in "/Users/nadavsmacbookair/Desktop/Thesis/data/financial_corpora/chunks/indexed-at-13-06-26/headers_split",
# for each print number of chunks, and numbers of 'sections','subsection','item'