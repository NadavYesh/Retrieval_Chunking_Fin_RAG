#%%
import pickle
import pandas as pd
#%%
import re
def filter_FinDER(df, strings_to_filter):

    fltr_trms_txt = "|".join([fr"\b{re.escape(t.strip())}\b" for t in strings_to_filter])
    filt_cond_txt = df["query"].str.contains(fltr_trms_txt, case=False, na=False)
    # remove entries were references are none (we test 10-k knowledge)

    return df[filt_cond_txt]




def run_finder(tickers):
    df = pd.read_parquet("/Users/nadavsmacbookair/Desktop/Thesis/data/FinDER/train.parquet")
    df = df.rename(columns={"text": "query", "answer": "truth_answer", "references": "truth_ref"})
    def _has_real_refs(x):
        if x is None:
            return False
        refs = [x] if isinstance(x, str) else list(x)
        if not refs:
            return False
        return any(str(r).strip().lower() not in ("none", "none.", "") for r in refs)

    df = df[df["truth_ref"].apply(_has_real_refs)]
    return filter_FinDER(df, tickers)

# %%
# nvda = run_finder(['nvda'])

# # %%
# from prompts import RAG_SYSTEM_PROMPT
# from mlx_lm import load, generate
# #model, tokenizer, *_ = load("mlx-community/Llama-3.2-3B-Instruct-4bit")
# model, tokenizer, *_ = load("mlx-community/phi-4-4bit")

# user_query="How are energy efficiency and Green500 recognition impacting profitability and cost mgmt in Nvidia GPU businesses (NVDA)?"
# context_text = "['NVIDIA invents computing technologies that improve lives and address global challenges. Our goal is to integrate sound environmental, social, and corporate governance principles and practices into every aspect of the Company. The Nominating and Corporate Governance Committee of our Board of Directors is responsible for reviewing and discussing with management our practices related to sustainability and corporate governance. We assess our programs annually in consideration of stakeholder expectations, market trends, and business risks and opportunities. These issues are important for our continued business success and reflect the topics of highest concern to NVIDIA and our stakeholders.\nThe following section and the Human Capital Management Section below provide an overview of our principles and practices. More information can be found on our website and in our annual Sustainability Report. Information contained on our website or in our annual Sustainability Report is not incorporated by reference into this or any other report we file with the Securities and Exchange Commission, or the SEC. Refer to “Item 1A. Risk Factors” for a discussion of risks and uncertainties we face related to sustainability.\nClimate Change\nIn the area of environmental sustainability, we address our climate impacts across our product lifecycle and assess risks, including current and emerging regulations and market impacts.\nIn May 2023, we published metrics related to our environmental impact for fiscal year 2023. Fiscal year 2024 metrics are expected to be published in the first half of fiscal year 2025. There has been no material impact to our capital expenditures, results of operations or competitive position associated with global environmental sustainability regulations, compliance, or costs from sourcing renewable energy. By the end of fiscal year 2025, our goal is to purchase or generate enough renewable energy to match 100% of our global electricity usage for our offices and data centers. In fiscal year 2023, we increased the percentage of our total electricity use matched by renewable energy purchases to 44%. By fiscal year 2026, we aim to engage manufacturing suppliers comprising at least 67% of NVIDIA’s scope 3 category 1 GHG emissions with goal of effecting supplier adoption of science-based targets.\nWhether it is creation of technology to power next-generation laptops or designs to support high-performance supercomputers, improving energy efficiency is important in our research, development, and design processes. GPU-accelerated computing is inherently more energy efficient than traditional computing for many workloads because it is optimized for throughput, performance per watt, and certain AI workloads. The energy efficiency of our products is evidenced by our continued strong presence on the Green500 list of the most energy-efficient systems. We powered 24 of the top 30 most energy efficient systems, including the top supercomputer, on the Green500 list.\nWe plan to build Earth-2, a digital twin of the Earth on NVIDIA AI and NVIDIA Omniverse platforms. Earth-2 will enable scientists, companies, and policy makers to do ultra-high-resolution predictions of the impact of climate change and explore mitigation and adaptation strategies.']"

# messages = [
#     {"role": "system", "content": RAG_SYSTEM_PROMPT},
#     {"role": "user", "content": f"Context:\n{context_text}\n\n Question: {user_query}"}
# ]

# prompt = tokenizer.apply_chat_template(
#     messages, tokenize=False, add_generation_prompt=True
# )

# # Generate response
# generated_text = generate(model, tokenizer, prompt=prompt, verbose=False)
# # %% change finder columns
# df = pd.read_parquet("/Users/nadavsmacbookair/Desktop/Thesis/data/FinDER/train.parquet")
# df.rename(columns={'text':'query','answer':'truth_answer','references':'truth_ref'})

# # %%
