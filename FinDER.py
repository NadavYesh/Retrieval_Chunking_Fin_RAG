#%%
import pandas as pd
#%%
import re
def filter_FinDER(df, strings_to_filter):
    # Using word boundaries (\b) ensures we match the ticker as a standalone word.
    fltr_trms = "|".join([fr"\b{re.escape(t.strip())}\b" for t in strings_to_filter])
    filt_cond = df["text"].str.contains(fltr_trms, case=False, na=False)
    return df[filt_cond]




def run_finder():
    companies = ["pypl","paypal","bestbuy","bby","google"
                                ,"googl","amazon","amzn","JPMorgan","jpm","tesla","tsla",
                                "meta","facebook","fb","newmont","nem","agnico","aem","boeing","ba",
                                "intc","intel","nvda","nvidia","aapl","apple","tsmc",
                                "taiwan semiconductor","wmt","walmart","xom","exxon",
                                "ko","cocacola","coca-cola","cola","chevron","cvx","welltower",
                                "iff","international flavors and fragrances"]
    df = pd.read_parquet("/Users/nadavsmacbookair/Desktop/Thesis/data/FinDER/train.parquet")
    return (filter_FinDER(df, companies))
