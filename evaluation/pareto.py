#%%
import numpy as np
import pandas as pd

df = pd.read_excel(
    "/Users/nadavsmacbookair/Desktop/Thesis/data/eval_results/analysis/analysis_multi_PYPL-TSLA_20260701_1243.xlsx",
    sheet_name="baseline_comparison",
)

metric_cols = ["Δ_word_recall", "Δ_num_recall",
                "Δ_soft_Recall@5",
                ]
values = df[metric_cols].to_numpy()

#%%


def pareto_efficient_mask(points: np.ndarray) -> np.ndarray:
    """Return a boolean mask of Pareto-efficient points, maximizing every column."""
    n = points.shape[0]
    is_efficient = np.ones(n, dtype=bool)
    for i in range(n):
        if not is_efficient[i]:
            continue
        dominates = np.all(points >= points[i], axis=1) & np.any(points > points[i], axis=1)
        if np.any(dominates):
            is_efficient[i] = False
    return is_efficient


mask = pareto_efficient_mask(values)

#%%

pareto_df = df.loc[mask, ["config_key", *metric_cols]].copy()
#print(f"{mask.sum()} / {len(df)} configs are Pareto-efficient")

#%% min max

norm_cols = [f"{col}_norm" for col in metric_cols]
col_min = pareto_df[metric_cols].min()
col_max = pareto_df[metric_cols].max()
pareto_df[norm_cols] = (pareto_df[metric_cols] - col_min) / (col_max - col_min)
pareto_df["norm_sum"] = pareto_df[norm_cols].sum(axis=1)

pareto_df = pareto_df.sort_values(by="norm_sum", ascending=False)
# %% Print full df with Pareto rows
df = df.iloc[pareto_df.index]
# %% Heatmap: config_key (rows) x metric_cols (columns)

import matplotlib.pyplot as plt
import seaborn as sns

heatmap_data = df.set_index("config_key")[metric_cols]

plt.figure(figsize=(1.5 * len(metric_cols) + 2, 0.4 * len(heatmap_data) + 2))
sns.heatmap(heatmap_data, annot=True, fmt=".3f", cmap="RdYlGn", center=0)
plt.xlabel("Metric")
plt.ylabel("Config")
plt.title("Δ Metrics by Config")
plt.tight_layout()
plt.show()
# %%
