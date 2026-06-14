import json
import time
from mlx_lm import load
from sentence_transformers import SentenceTransformer
from search_engine import search_agent, generate_llm_answer
from extract_financebench_data import get_fb_points
#%%
# configure

# Default collection name
# DEFAULT_COLL_NAME = "--split headers --embeddings text,meta"
DEFAULT_COLL_NAME = "--split headers,chars --embeddings text,meta"
# Load models
model, tokenizer, *extra = load("mlx-community/Llama-3.2-3B-Instruct-4bit")
embed_model = SentenceTransformer("google/embeddinggemma-300M", device="mps")


def run_evaluation(company_name, year):
    # from trulens_eval import evaluate_trulens_response
    
    records = get_fb_points(company_name, year)
    filtered_pairs = [
        {"question": r["question"], "reference_answer": r["answer"], "evidence":r["evidence"]}
        for r in records
    ]
    print(f"Found {len(filtered_pairs)} relevant question-answer pairs for {company_name} in {year}.")
    
    #====================================== main loop
    for i, pair in enumerate(filtered_pairs, 1):
        user_query = pair["question"]
        ref_answer = pair["reference_answer"]
        print(f"\n{'='*50}")
        print(f"Original Query {i}/{len(filtered_pairs)}: {user_query}")
        
        try:
            opt_retr_query, results_enhanced, results_raw = search_agent(user_query, model, tokenizer, 
                                                  embed_model, coll_name = DEFAULT_COLL_NAME, 
                                                  ENAHNCE_QUERY=True, BOTH = True) #change these to isolate
        except Exception as e:
            print(f"Pipeline failed for query {i}: {e}")
        if opt_retr_query:
                print("\n--- Enhanced Query ---")
                print(opt_retr_query)

        print("\n--- REF Answer   ---")
        print(ref_answer)
        
        print("\n--- Evidence (financebench)   ---")
        print(pair["evidence"])

        if results_enhanced and results_enhanced.points:
            answer, chunk_sources = generate_llm_answer(user_query, results_enhanced, model, tokenizer)
            print("\n--- LLM Answer : Enhanced Query---")
            print(answer)
            

            # print("\n--- Running TRULENS Evaluation ---")
            # time.sleep(20)
            # for now we stop with trulens
            # scores = evaluate_trulens_response(user_query, answer, results)
            # print("\n--- trulens Scores [0-1] ---")
            # for metric, score in scores.items():
            #     if metric not in ["question", "answer", "contexts"]: 
            #         print(f"{metric.capitalize()}: {score}")

        if results_raw and results_raw.points:
            answer, chunk_sources = generate_llm_answer(user_query, results_raw, model, tokenizer)
            print("\n--- LLM Answer : Raw Query ---")
            print(answer)
            


            # print("\n--- Running TRULENS Evaluation ---")
            # time.sleep(20)
            # for now we stop with trulens
            # scores = evaluate_trulens_response(user_query, answer, results)
            # print("\n--- trulens Scores [0-1] ---")
            # for metric, score in scores.items():
            #     if metric not in ["question", "answer", "contexts"]: 
            #         print(f"{metric.capitalize()}: {score}")



if __name__ == "__main__":
    run_evaluation("3m",2022)
