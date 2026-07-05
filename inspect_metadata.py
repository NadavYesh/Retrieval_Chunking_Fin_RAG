import pickle, os

path = '/Users/nadavsmacbookair/Desktop/Thesis/data/financial_corpora/chunks/hierarchical/indexed-at-22-06-26/header'
files = sorted([f for f in os.listdir(path) if f.endswith('.pkl')])

for f in files:
    with open(os.path.join(path, f), 'rb') as fh:
        df = pickle.load(fh)
    meta = df.iloc[0]['metadata']
    print(f'=== {f} ===')
    for k, v in meta.items():
        print(f'  {k}: {v}')
    print()
