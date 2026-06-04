#%%
from openai import OpenAI
import re

# Use the same client setup as LLM_search for manual evaluation
client = OpenAI(base_url="http://localhost:8000/v1", api_key="dummy")
MODEL = "mlx-community/Llama-3.2-3B-Instruct-4bit"
#%%
def _extract_score(text):
    """Fallback score extractor from text."""
    try:
        # First, look for a explicit "Score: X" pattern
        score_match = re.search(r"[Ss]core:\s*(\d+(?:\.\d+)?)", text)
        if score_match:
            val = float(score_match.group(1))
        else:
            # Look for ratings like 8/10
            ratio_match = re.search(r"(\d+(?:\.\d+)?)\s*/\s*10", text)
            if ratio_match:
                val = float(ratio_match.group(1))
            else:
                # Last resort: find all numbers and pick the most likely one (usually the first or last)
                matches = re.findall(r"(\d+(?:\.\d+)?)", text)
                if not matches: return 0.0
                # If there are multiple numbers, the one <= 10 is likely the score
                candidates = [float(m) for m in matches if float(m) <= 10]
                val = candidates[-1] if candidates else float(matches[-1])
        
        if val > 10: val = val / 10.0 # normalize if 100-based
        return min(max(val / 10.0, 0.0), 1.0) # normalize to 0.0-1.0
    except:
        pass
    return 0.0

def evaluate_trulens_response(query, response_text, retrieved_results):
    """
    Manually evaluate metrics using the local LLM to avoid 422/404 errors 
    caused by TruLens structured output requirements on mlx-openai-server.
    """
    # Payloads in vector_db.py use 'page_content' usually
    contexts = [res.payload.get("page_content", "") for res in retrieved_results.points]
    contexts = [c for c in contexts if c.strip()]
    
    scores = {}
    print(f"DEBUG: Manually evaluating metrics for Query: {query[:50]}...")

    # 1. Answer Relevance
    try:
        prompt = f"Question: {query}\nAnswer: {response_text}\n\nRate the relevance of the answer to the question on a scale from 0 to 10, where 10 is perfectly relevant. Provide your reasoning and end with 'Score: X'."
        res = client.chat.completions.create(model=MODEL, messages=[{"role": "user", "content": prompt}])
        raw = res.choices[0].message.content
        scores["answer_relevance"] = _extract_score(raw)
        print(f"  - Answer Relevance: {scores['answer_relevance']}")
    except Exception as e:
        print(f"  - Answer Relevance Error: {e}")
        scores["answer_relevance"] = 0.0

    # 2. Context Relevance
    try:
        ctx_scores = []
        for ctx in contexts[:3]: # Evaluate top 3 contexts
            prompt = f"Question: {query}\nContext: {ctx[:1000]}\n\nHow relevant is this context to answering the question? Rate from 0 to 10. End with 'Score: X'."
            res = client.chat.completions.create(model=MODEL, messages=[{"role": "user", "content": prompt}])
            raw = res.choices[0].message.content
            score = _extract_score(raw)
            print(f"    - Raw context eval: {raw[:100]}... -> Score: {score}")
            ctx_scores.append(score)
        scores["context_relevance"] = sum(ctx_scores) / len(ctx_scores) if ctx_scores else 0.0
        print(f"  - Context Relevance: {scores['context_relevance']}")
    except Exception as e:
        print(f"  - Context Relevance Error: {e}")
        scores["context_relevance"] = 0.0

    # 3. Groundedness
    try:
        all_ctx = "\n---\n".join(contexts[:3])
        prompt = f"Context: {all_ctx[:2000]}\nAnswer: {response_text}\n\nIs the answer supported by the context? Rate the groundedness from 0 to 10. End with 'Score: X'."
        res = client.chat.completions.create(model=MODEL, messages=[{"role": "user", "content": prompt}])
        raw = res.choices[0].message.content
        scores["groundedness"] = _extract_score(raw)
        print(f"  - Groundedness: {scores['groundedness']}")
    except Exception as e:
        print(f"  - Groundedness Error: {e}")
        scores["groundedness"] = 0.0

    return scores
