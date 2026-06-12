import json
import os

DEFAULT_DATASET_PATH = '/Users/nadavsmacbookair/Desktop/Thesis/Sources/data/financebench_open_source.jsonl'

def get_fb_points(company_query, year, dataset_path=DEFAULT_DATASET_PATH):
    """
    Extracts question-answer pairs from FinanceBench for a specific company and year.
    
    Args:
        company_query (str): Company name or ticker to search for (e.g., '3m', 'apple').
        year (int|str): Fiscal year to search for (e.g., 2022).
        dataset_path (str): Path to the financebench_open_source.jsonl file.
        
    Returns:
        list[dict]: List of filtered records.
    """
    fb_data = []
    company_query = company_query.lower()
    year_str = str(year)
    
    try:
        if not os.path.exists(dataset_path):
            print(f"Warning: Dataset not found at {dataset_path}")
            return []
            
        with open(dataset_path, 'r') as f:
            for line in f:
                obj = json.loads(line)
                company = obj.get('company', '').lower()
                doc_name = obj.get('doc_name', '').lower()
                
                # Check for company match (contains query) and year match
                if company_query in company and year_str in doc_name:
                    fb_data.append(obj)
                    
        print(f"Found {len(fb_data)} records for '{company_query}' in {year_str}")
        return fb_data
        
    except Exception as e:
        print(f"Error reading dataset: {e}")
        return []

if __name__ == "__main__":
    # Example usage:
    # results = get_fb_points('3m', 2022)
    # for r in results[:2]:
    #     print(f"Q: {r['question']}")
    pass
