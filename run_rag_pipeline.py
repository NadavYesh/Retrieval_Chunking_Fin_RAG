import json
import time
from mlx_lm import load
from sentence_transformers import SentenceTransformer
from search_engine import search_agent, generate_rag_response
from extract_financebench_data import get_fb_points

# configure

# Default collection name
DEFAULT_COLL_NAME = "--split headers --embeddings text,meta"
# Load models
model, tokenizer, *extra = load("mlx-community/Llama-3.2-3B-Instruct-4bit")
embed_model = SentenceTransformer("google/embeddinggemma-300M", device="mps")


def run_evaluation(company_query, year):
    # from trulens_eval import evaluate_trulens_response
    
    records = get_fb_points(company_query, year)
    filtered_pairs = [
        {"question": r["question"], "reference_answer": r["answer"]}
        for r in records
    ]
    
    print(f"Found {len(filtered_pairs)} relevant question-answer pairs for {company_query} in {year}.")
    
    for i, pair in enumerate(filtered_pairs, 1):
        user_query = pair["question"]
        ref_answer = pair["reference_answer"]
        print(f"\n{'='*50}")
        print(f"Query {i}/{len(filtered_pairs)}: {user_query}")
        
        try:
            opt_ret_query, results = search_agent(user_query, model, tokenizer, embed_model, coll_name = DEFAULT_COLL_NAME)
            
            if results and results.points:
                answer, chunk_sources = generate_rag_response(user_query, results, model, tokenizer)
                print("\n--- LLM Answer ---")
                print(answer)
                
                print("\n--- Running TRULENS Evaluation ---")
                time.sleep(20)
                # for now we stop with trulens
                # scores = evaluate_trulens_response(user_query, answer, results)
                # print("\n--- trulens Scores [0-1] ---")
                # for metric, score in scores.items():
                #     if metric not in ["question", "answer", "contexts"]: 
                #         print(f"{metric.capitalize()}: {score}")
            else:
                print("Search failed or returned no results.")
        except Exception as e:
            print(f"Pipeline failed for query {i}: {e}")

if __name__ == "__main__":
    # run_evaluation()
