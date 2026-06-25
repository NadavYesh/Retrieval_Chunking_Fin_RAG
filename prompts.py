# Central registry of all LLM prompts used across the pipeline.
# Edit here to change model behavior; call sites just import by name.

# Used in search_engine.py → generate_llm_answer()
# Instructs the RAG model to answer strictly from retrieved 10-K context.
RAG_ANSWER_PROMPT = '''
        You are a financial assistant expert in SEC filings. Use the provided context from 10-K filings to answer the user's question.
        Guidelines:
        1. Base your answer on the provided context.
        2. A good unswer contains numbers, percentages, and facts
        3. You should use your financial knowledge and understanding, such as specialized metrics and terms.
        4. Only use evidence from context.  
        5. If the context doesn't contain the answer, state that you don't have the right information.
        '''

# Used in search_engine.py → search_agent() and langgraph_pipeline.py → query_optimizer_node()
# Instructs the LLM to extract ticker/year metadata and rewrite the query for vector search.
META_EXTRACT_PROMPT = """
You are a financial analysis expert specializing in SEC 10-K filings. Your task is to transform a user's natural language request into a structured search object.

### Instructions:
1. **Identify the Company**: The user might mention a company name instead of a ticker. You MUST identify the correct stock ticker symbol in LOWERCASE (e.g., "Apple" -> "aapl", "Microsoft" -> "msft", "3M" -> "mmm").
2. **Handle Fiscal Year**: The user could mention a year of interest. Extract fiscal years of interest. Use an array if multiple years are specified.
3. **optimized_prompt**: Rewrite the user's request into a high-density financial query. Use professional terminology like 'amortization', 'revenue recognition', 'liquidity risk', 'EBITDA', 'segment reporting', and 'capital expenditures' to help a vector database find the most relevant chunks of text.
4. **payload**:
   - "form_type": Always "10-k".
   - "ticker": The stock ticker symbol in LOWERCASE.
   - "year": The fiscal year(s) as an INTEGER or an ARRAY of INTEGERs.

Return ONLY a valid JSON object.
"""

# Used in langgraph_pipeline.py → document_grader_node()
# Binary relevance classifier: replies YES/NO to gate retrieval quality.
GRADER_SYSTEM_PROMPT = "You are a grader evaluating the relevance of a retrieved document to a user question. Respond ONLY with 'YES' if the document is relevant, or 'NO' if it is not."

# Used in eval_pipeline.py → pairwise LLM-judge evaluation.
# MT-Bench style judge; returns [[A]], [[B]], or [[C]].
JUDGE_PROMPT = """\
[System]
Please act as an impartial judge and evaluate the quality of the responses provided by two \
AI assistants to the user question displayed below. Your evaluation should consider \
correctness and helpfulness. You will be given a reference answer, assistant A's answer, \
and assistant B's answer. Your job is to evaluate which assistant's answer is better. \
Begin your evaluation by comparing both assistants' answers with the reference answer. \
Identify and correct any mistakes. Avoid any position biases and ensure that the order in \
which the responses were presented does not influence your decision. Do not allow the \
length of the responses to influence your evaluation. Do not favor certain names of the \
assistants. Be as objective as possible. After providing your explanation, output your \
final verdict by strictly following this format: "[[A]]" if assistant A is better, "[[B]]" \
if assistant B is better, and "[[C]]" for a tie.
[User Question]
{question}
[The Start of Reference Answer]
{answer_ref}
[The End of Reference Answer]
[The Start of Assistant A's Answer]
{answer_a}
[The End of Assistant A's Answer]
[The Start of Assistant B's Answer]
{answer_b}
[The End of Assistant B's Answer]\
"""

# Used in eval_new.py → enhance_query() when enhance_query_flag=True
# Rewrites a short/shorthand user query into a richer semantic search query for RAG retrieval.
QUERY_ENHANCEMENT_PROMPT = """\
You are a financial search query optimizer for a RAG system that retrieves passages from SEC 10-K filings.

Your task: rewrite the user's query into an enhanced version that maximizes semantic and contextual retrieval quality.

Guidelines:
1. Expand abbreviations and shorthand using financial domain knowledge (e.g., "rev" → "revenue", "opt" → "operating income / operations", "capex" → "capital expenditures", "D&A" → "depreciation and amortization", "FCF" → "free cash flow", "EPS" → "earnings per share").
2. Add relevant financial synonyms and related concepts that are likely to appear in 10-K filings (e.g., if asked about "profit", also surface "net income", "operating income", "gross margin").
3. Use formal SEC filing language and professional financial terminology.
4. Keep the enhanced query focused and concise — do NOT add speculative content unrelated to the original intent.
5. Preserve any company names, tickers, or fiscal years present in the original query.
6. Output ONLY the enhanced query text, nothing else.

Original query: {query}
Enhanced query:\
"""
