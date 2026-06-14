import os
import pickle
import uuid
import pandas as pd

CHUNKS_DIRS = ["/Users/nadavsmacbookair/Desktop/Thesis/data/financial_corpora/chunks/indexed-at-13-06-26/headers_split", "/Users/nadavsmacbookair/Desktop/Thesis/data/financial_corpora/chunks/indexed-at-13-06-26/headers_chars_split"]

def migrate_files():
    for DIR in CHUNKS_DIRS:
        print(f"Starting UUID migration in {DIR}...")
        for root, _, files in os.walk(DIR):
            for file in files:
                if file.endswith(".pkl"):
                    path = os.path.join(root, file)
                    try:
                        with open(path, "rb") as f:
                            df = pickle.load(f)
                        
                        if isinstance(df, pd.DataFrame) and 'id' not in df.columns:
                            df['id'] = [str(uuid.uuid4()) for _ in range(len(df))]
                            df.to_pickle(path)
                            print(f"Successfully migrated: {file}")
                        else:
                            print(f"Skipping {file} (already migrated or invalid format)")
                            
                    except Exception as e:
                        print(f"Error migrating {file}: {e}")
        print("migrated ",DIR)
    print("Migration complete.")

if __name__ == "__main__":
    migrate_files()