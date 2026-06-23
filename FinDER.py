#%%
import pandas as pd
#%%
import re
def filter_FinDER(df, strings_to_filter):
    # Using word boundaries (\b) ensures we match the ticker as a standalone word.
    fltr_trms = "|".join([fr"\b{re.escape(t.strip())}\b" for t in strings_to_filter])
    filt_cond = df["text"].str.contains(fltr_trms, case=False, na=False)
    return df[filt_cond]




def run_finder(companies):
    # companies = ["pypl","paypal"]
    df = pd.read_parquet("/Users/nadavsmacbookair/Desktop/Thesis/data/FinDER/train.parquet")
    return (filter_FinDER(df, companies))
