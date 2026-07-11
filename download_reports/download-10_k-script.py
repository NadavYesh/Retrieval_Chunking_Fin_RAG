#%% Download 10-k
# Download 10-k files, in html, for the companies listed in the S&P500 reference CSV.
# save at /Users/nadavsmacbookair/Desktop/Thesis/data/html
import os
import pandas as pd
from edgar import find
from datetime import date
import re


# Define the target directory
SAVE_PATH = "/Users/nadavsmacbookair/Desktop/Thesis/data/html"
if not os.path.exists(SAVE_PATH):
    os.makedirs(SAVE_PATH)

os.environ['EDGAR_IDENTITY'] = "ThesisResearchProject nadav@uva.nl"

SP500_REFERENCE_CSV = "/Users/nadavsmacbookair/Desktop/Thesis/data/reference/sp500_companies.csv"
FILINGS_PER_TICKER = 6  # ~fiscal years 2018-2024

# Extract tickers: generally 1-5 chars, excluding common names from your list
exclude_names = {"cola", "intel", "apple", "tesla", "exxon", "google"}
companies = pd.read_csv(SP500_REFERENCE_CSV)["Symbol"].tolist()
tickers = [t.upper() for t in companies if len(t) <= 5 and t.lower() not in exclude_names]

print(f"Starting download for tickers: {tickers}")

for ticker in tickers:
    try:
        print(f"Fetching 10-Ks for {ticker}...")
        company = find(ticker)
        filings = company.get_filings(form="10-K").head(FILINGS_PER_TICKER)

        count = 0
        for  filing in filings:
            html_content = filing.html()
            if html_content:
                clean_content = re.sub(r'</html>.*', '</html>', html_content, flags=re.DOTALL | re.IGNORECASE)
                
                filename = f"{ticker}_10K_{2025-count}.html"
                with open(os.path.join(SAVE_PATH, filename), "w", encoding="utf-8") as w:
                    w.write(clean_content)
                count += 1
        print(f"Saved {count} cleaned filings for {ticker}")
    except Exception as e:
        print(f"Could not download filings for {ticker}: {e}")

print("Download process complete.")
