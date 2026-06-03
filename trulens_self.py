#%%
import numpy as np
from trulens.core import Metric, TruSession, Selector
from trulens.providers.litellm import LiteLLM
import litellm
from trulens.dashboard import run_dashboard
session = TruSession()
session.reset_database()


litellm.api_base = "http://localhost:8080/v1"
litellm.api_key = "dummy"

provider = LiteLLM(
    model_engine="openai/mlx-community/Llama-3.2-3B-Instruct-4bit"
)

#%%


# Define a groundedness metric
f_groundedness = Metric(
    implementation=provider.groundedness_measure_with_cot_reasons_consider_answerability,
    name="Groundedness",
    selectors={
        "source": Selector.select_context(collect_list=True),
        "statement": Selector.select_record_output(),
        "question": Selector.select_record_input(),
    },
)

# Question/answer relevance between overall question and answer.
f_answer_relevance = Metric(
    implementation=provider.relevance_with_cot_reasons,
    name="Answer Relevance",
    selectors={
        "prompt": Selector.select_record_input(),
        "response": Selector.select_record_output(),
    },
)

# Context relevance between question and each context chunk.
f_context_relevance = Metric(
    implementation=provider.context_relevance_with_cot_reasons,
    name="Context Relevance",
    selectors={
        "question": Selector.select_record_input(),
        "context": Selector.select_context(collect_list=False),
    },
    agg=np.mean,  # choose a different aggregation method if you wish
)
# %%

def evaluate_trulens_response(
    query,
    response_text,
    search_results,
    model_id="mlx-community/Llama-3.2-3B-Instruct-4bit",
):
    """
    Evaluates a single RAG turn using TruLens RAG triad feedback functions.
    Returns a dictionary of scores.
    """
    contexts = [p.payload.get("text", "") for p in search_results.points]
    contexts = [context for context in contexts if context.strip()]
    if not contexts:
        return {"error": "No context found in search results."}

    # Use the configured LiteLLM provider and the metric implementations above.
    # Helper to extract numeric score from various possible return shapes.
    def _extract_score(result):
        if result is None:
            raise ValueError("No result returned")
        if isinstance(result, (int, float)):
            return float(result)
        if isinstance(result, tuple) and len(result) > 0 and isinstance(result[0], (int, float)):
            return float(result[0])
        if isinstance(result, dict):
            # common keys
            for k in ("score", "value", "rating"):
                if k in result:
                    return float(result[k])
            # try to find a numeric in nested structure
            for v in result.values():
                if isinstance(v, (int, float)):
                    return float(v)
        raise ValueError(f"Unable to extract numeric score from: {result}")

    scores = {}
    errors = {}
    #WORKS!
    # Answer relevance: between query and response
    try:
        raw = provider.relevance_with_cot_reasons(prompt=query, response=response_text)
        print(f"Raw answer relevance: {raw}")
        scores["answer_relevance"] = _extract_score(raw)
    except Exception as exc:
        errors["answer_relevance"] = str(exc)
    
    #WORKS!
    # Context relevance: per-context between query and each context chunk
    try:
        context_scores = []
        for context in contexts:
            raw = provider.context_relevance_with_cot_reasons(question=query, context=context)
            print(f"Raw context relevance: {raw}")
            context_scores.append(_extract_score(raw))
        scores["context_relevance"] = sum(context_scores) / len(context_scores)
    except Exception as exc:
        errors["context_relevance"] = str(exc)
    # Groundedness: evaluate response against all contexts
    try:
        raw = provider.groundedness_measure_with_cot_reasons_consider_answerability(
            source=contexts, statement=response_text, question=query
        )
        print(f"Raw groundedness: {raw}")
        scores["groundedness"] = _extract_score(raw)
    except Exception as exc:
        errors["groundedness"] = str(exc)

    if errors:
        scores["error"] = errors

    return scores