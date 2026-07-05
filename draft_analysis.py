#%%
import pickle
from FinDER import run_finder

#%% chunking analysis
with open("/Users/nadavsmacbookair/Desktop/Thesis/data/financial_corpora/chunks/hierarchical/indexed-at-26-06-26/header/WMT_10K_2026.pkl","rb") as f:
    doc = pickle.load(f)
#%%
txt_search = "primarily consist of inventory"
search = doc["text"].str.lower().str.contains(txt_search)
doc[search]

#%% finder analysis
run_finder(['tsla','wmt'])
# %%
