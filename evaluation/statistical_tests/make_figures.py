"""
Figures for the Results chapter.

Reads judge_ordinal_scores.xlsx (the 34 x 36 grid carrying soft_Recall@3
and the judged ordinal score) and writes vector PDFs to
Master_Thesis_Tex/media/figs/.

The bootstrap here is the one in judge_tie_breakdown.py: queries are the unit of
independence, so queries -- not runs -- are resampled with replacement
(SEED=0, N_BOOT=5000). Point estimates therefore reproduce the chapter's tables.
"""
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.lines import Line2D

SRC = Path("/Users/nadavsmacbookair/Desktop/Thesis/data/eval_results/"
           "ready_for_analysis/analysis/judge_ordinal_scores.xlsx")
OUT = Path("/Users/nadavsmacbookair/Desktop/Thesis/Master_Thesis_Tex/media/figs")

N_BOOT, SEED = 5000, 0
BASELINE = "L1_BM25_PLAIN_A0"

# Ink and chrome. Grid and axes are recessive hairlines; text never wears a
# series colour.
INK, SECONDARY, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, AXIS = "#e1e0d9", "#c3c2b7"
BLUE, RED, AQUA = "#2a78d6", "#e34948", "#1baf7a"
GOOD, CRIT = "#0ca30c", "#d03b3b"

# Level is an ordered factor, so it takes an ordinal ramp (one hue, light->dark)
# rather than three categorical hues.
LEVEL_RAMP = {1: "#86b6ef", 2: "#2a78d6", 3: "#104281"}
MODE_MARK = {"sparse": "s", "dense": "o", "hybrid": "D"}
MODE_LABEL = {"sparse": "BM25 (sparse)", "dense": "Dense", "hybrid": "Hybrid"}

SEQ_BLUE = LinearSegmentedColormap.from_list(
    "seq_blue", ["#eaf2fd", "#cde2fb", "#9ec5f4", "#5598e7", "#2a78d6", "#1c5cab", "#0d366b"])
# Judge scores are win-rates against a fixed baseline: 0.5 is a true neutral
# midpoint ("tie"), which is what licenses a diverging ramp.
DIV_BR = LinearSegmentedColormap.from_list(
    "div_br", ["#8f2321", "#d03b3b", "#e88a89", "#f0efec", "#86b6ef", "#2a78d6", "#0d366b"])

mpl.rcParams.update({
    "font.family": "serif",                     # matches the Libertine body text
    "font.serif": ["Linux Libertine O", "Libertinus Serif", "DejaVu Serif"],
    "font.size": 9,
    "axes.edgecolor": AXIS, "axes.linewidth": 0.6,
    "axes.labelcolor": SECONDARY, "axes.titlesize": 9.5,
    "xtick.color": MUTED, "ytick.color": MUTED,
    "xtick.labelcolor": SECONDARY, "ytick.labelcolor": SECONDARY,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "grid.color": GRID, "grid.linewidth": 0.6, "grid.linestyle": "-",
    "legend.frameon": False, "legend.fontsize": 8,
    "figure.dpi": 150, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
    "pdf.fonttype": 42,
})

# soft_NDCG@5 is gone (graded gains over a binary relevance label add nothing).
# soft_MRR is reported again: it reads the ORDER of the retrieved chunks, which was
# arbitrary while parent-fetch collapsed level-2/3 child hits onto their level-1
# parents through a set — but _collapse_to_parents now ranks parents by an RRF-sum
# over their child ranks and re-sorts explicitly, so rank is well-defined at every
# level. soft_Recall@3 is the coarser check: a hit anywhere in the top 3 of the 5
# retrieved chunks, so it reads rank too, but only through that cutoff.
METRICS = {"ordinal": "Judged ordinal score",
           "soft_Recall@3": "soft_Recall@3",
           "soft_MRR": "soft_MRR"}


def pretty(cfg: str) -> str:
    """L2_HYBRID_ENH_A0 -> L2 hybrid enh."""
    lvl, mode, enh, alpha = cfg.split("_")
    mode = {"BM25": "bm25", "DENSE": "dense", "HYBRID": "hybrid"}[mode]
    tail = " enh" if enh == "ENH" else ""
    tail += r" $\alpha$=1" if alpha == "A1" else ""
    return f"{lvl} {mode}{tail}"


def wide(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    """queries x configs; the matrix the bootstrap resamples rows of."""
    return df.pivot_table(index="query", columns="config_key", values=metric)


def boot_means(w: pd.DataFrame) -> np.ndarray:
    """(n_boot x n_configs) matrix of config means over resampled query sets."""
    rng = np.random.default_rng(SEED)
    q = w.index.to_numpy()
    return np.array([w.loc[rng.choice(q, q.size, replace=True)].mean().to_numpy()
                     for _ in range(N_BOOT)])


def summarise(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Per-config mean, 95% cluster-bootstrap CI, and P(best)."""
    w = wide(df, metric)
    draws = boot_means(w)
    lo, hi = np.percentile(draws, [2.5, 97.5], axis=0)
    p_best = np.bincount(draws.argmax(axis=1), minlength=w.shape[1]) / N_BOOT
    return pd.DataFrame({"mean": w.mean().to_numpy(), "lo": lo, "hi": hi,
                         "p_best": p_best}, index=w.columns)


# ── Figure 1: the divergence between retrieval and judged quality ────────────
def fig_divergence(df: pd.DataFrame) -> None:
    d = df[df.alpha == 0]
    g = d.groupby(["config_key", "level", "mode"]).agg(
        rec=("soft_Recall@3", "mean"), ordv=("ordinal", "mean")).reset_index()
    base = g[g.config_key == BASELINE].iloc[0]

    fig, ax = plt.subplots(figsize=(5.6, 4.3))
    ax.grid(True, lw=0.6, zorder=0)
    ax.set_axisbelow(True)

    # The baseline sits at 0.5 on the judge axis by construction; its soft_Recall@3
    # is an empirical value. Both are reference lines, not data.
    ax.axhline(0.5, color=AXIS, lw=0.8, zorder=1)
    ax.axvline(base.rec, color=AXIS, lw=0.8, zorder=1)
    ax.text(base.rec - 0.004, 0.30, "BM25 baseline soft_Recall@3", color=MUTED,
            fontsize=7.5, ha="right", va="bottom", rotation=90)
    # Axes-fraction x so this label stays put whatever range Recall@3 autoscales to.
    ax.text(0.98, 0.505, "judged tie with baseline", color=MUTED, fontsize=7.5,
            ha="right", va="bottom", transform=ax.get_yaxis_transform())

    for _, r in g.iterrows():
        ax.scatter(r.rec, r.ordv, s=54, marker=MODE_MARK[r["mode"]],
                   facecolor=LEVEL_RAMP[r.level], edgecolor="#fcfcfb", lw=1.2,
                   zorder=3)

    # Direct-label only the configurations the chapter argues about.
    for cfg, dx, dy, ha in [("L3_HYBRID_PLAIN_A0", 0, .022, "center"),
                            ("L1_HYBRID_PLAIN_A0", .004, -.030, "center"),
                            ("L2_HYBRID_ENH_A0", -.008, .006, "right"),
                            ("L1_BM25_PLAIN_A0", 0, -.032, "center")]:
        r = g[g.config_key == cfg].iloc[0]
        ax.annotate(pretty(cfg), (r.rec + dx, r.ordv + dy), ha=ha, va="center",
                    fontsize=7.5, color=INK, zorder=4)

    ax.set_xlabel("soft_Recall@3  (retrieval)")
    ax.set_ylabel("Judged ordinal score  (answer quality)")
    # x autoscales: Recall@3 does not live on the same range the old MRR axis was
    # hand-tuned for. The annotation offsets above may need nudging once seen.
    ax.set_ylim(0.28, 0.80)

    lv = [Line2D([], [], marker="o", ls="", ms=6.5, mfc=LEVEL_RAMP[l],
                 mec="#fcfcfb", label=f"$L_{l}$") for l in (1, 2, 3)]
    md = [Line2D([], [], marker=MODE_MARK[m], ls="", ms=6, mfc=MUTED,
                 mec="#fcfcfb", label=MODE_LABEL[m]) for m in MODE_MARK]
    # Both legends sit in the empty lower-left quadrant, clear of the point cloud
    # and of the L1-hybrid label in the lower right.
    leg = ax.legend(handles=lv, loc="lower left", title="Chunking level",
                    bbox_to_anchor=(0.005, 0.005), labelspacing=.35,
                    handletextpad=.2)
    leg.get_title().set_fontsize(8)
    leg.get_title().set_color(SECONDARY)
    ax.add_artist(leg)
    leg2 = ax.legend(handles=md, loc="lower left", title="Retrieval mode",
                     bbox_to_anchor=(0.20, 0.005), labelspacing=.35,
                     handletextpad=.2)
    leg2.get_title().set_fontsize(8)
    leg2.get_title().set_color(SECONDARY)

    fig.savefig(OUT / "divergence.pdf")
    plt.close(fig)


# ── Figure 2: ranked configs with bootstrap CIs (the separability picture) ───
def fig_caterpillar(df: pd.DataFrame) -> None:
    stats = {m: summarise(df, m) for m in METRICS}
    order = stats["ordinal"].sort_values("mean", ascending=False).index

    fig, axes = plt.subplots(1, len(METRICS), figsize=(5.4, 6.6), sharey=True)
    y = np.arange(len(order))[::-1]

    for ax, (metric, label) in zip(axes, METRICS.items()):
        s = stats[metric].loc[order]
        alpha1 = np.array([c.endswith("A1") for c in order])
        col = np.where(alpha1, RED, BLUE)

        for yi, (_, r), c in zip(y, s.iterrows(), col):
            ax.plot([r.lo, r.hi], [yi, yi], color=c, lw=1.6, alpha=.30,
                    solid_capstyle="round", zorder=2)
            ax.plot(r["mean"], yi, "o", ms=4.6, mfc=c, mec="#fcfcfb", mew=.9,
                    zorder=3)

        ref = 0.5 if metric == "ordinal" else df[df.config_key == BASELINE][metric].mean()
        ax.axvline(ref, color=AXIS, lw=0.8, zorder=1)
        ax.grid(True, axis="x", lw=0.6, zorder=0)
        ax.set_axisbelow(True)
        ax.set_title(label, color=INK, pad=6)
        ax.set_xlim(0, 1)
        ax.set_xticks([0, .25, .5, .75, 1])
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.tick_params(axis="y", length=0)

    axes[0].set_yticks(y)
    axes[0].set_yticklabels([pretty(c) for c in order], fontsize=7.6)
    axes[0].set_ylim(-1, len(order))

    handles = [Line2D([], [], marker="o", ls="-", ms=5, color=BLUE,
                      mec="#fcfcfb", label=r"$\alpha$ = 0  (no routing)"),
               Line2D([], [], marker="o", ls="-", ms=5, color=RED,
                      mec="#fcfcfb", label=r"$\alpha$ = 1  (strict routing)")]
    # Reserve a band under the axes for the legend + caption so neither lands on
    # the x tick labels.
    fig.tight_layout(rect=(0, 0.055, 1, 1))
    fig.legend(handles=handles, loc="lower center", ncol=2,
               bbox_to_anchor=(0.55, 0.022))
    fig.text(0.55, 0.002, "Point = mean over 34 queries; bar = 95% cluster-bootstrap CI. "
             "Vertical rule = BM25 baseline.", ha="center", fontsize=7.5, color=MUTED)
    fig.savefig(OUT / "caterpillar.pdf")
    plt.close(fig)


# ── Figure 3: heatmaps of the configuration grid ─────────────────────────────
def fig_heatmap(df: pd.DataFrame) -> None:
    d = df[df.alpha == 0]
    rows = [(l, e) for l in (1, 2, 3) for e in ("NO", "YES")]
    cols = ["sparse", "dense", "hybrid"]
    rlab = [f"$L_{l}$" + (" enh" if e == "YES" else "") for l, e in rows]

    panels = [("ordinal", "Judged ordinal score", DIV_BR,
               TwoSlopeNorm(vmin=0.25, vcenter=0.5, vmax=0.75)),
              ("soft_Recall@3", "soft_Recall@3", SEQ_BLUE,
               mpl.colors.Normalize(0.0, 1.0))]

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.1))
    for ax, (metric, label, cmap, norm) in zip(axes, panels):
        M = np.array([[d[(d.level == l) & (d.enhanced == e) & (d["mode"] == m)][metric].mean()
                       for m in cols] for l, e in rows])
        im = ax.imshow(M, cmap=cmap, norm=norm, aspect="auto")
        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                # Ink flips to white only where the cell is dark enough to need it.
                dark = norm(M[i, j]) > 0.78 or norm(M[i, j]) < 0.16
                ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center",
                        fontsize=8.5, color="#ffffff" if dark else INK)
        ax.set_xticks(range(3), [MODE_LABEL[c] for c in cols], fontsize=8)
        ax.set_yticks(range(len(rows)), rlab, fontsize=8)
        ax.set_title(label, color=INK, pad=6)
        ax.tick_params(length=0)
        for s in ax.spines.values():
            s.set_visible(False)
        cb = fig.colorbar(im, ax=ax, fraction=.046, pad=.03)
        cb.outline.set_visible(False)
        cb.ax.tick_params(length=0, labelsize=7.5)
        if metric == "ordinal":
            cb.set_ticks([0.25, 0.5, 0.75])
            cb.ax.set_yticklabels(["0.25\n(worse)", "0.50\n(tie)", "0.75\n(better)"],
                                  fontsize=7)

    fig.text(0.5, -0.06, r"$\alpha = 0$ only. Judge cells are win-rates against the "
             "BM25 baseline; 0.50 is a tie.", ha="center", fontsize=7.5, color=MUTED)
    fig.tight_layout()
    fig.savefig(OUT / "heatmap.pdf")
    plt.close(fig)


# ── Figure 4: section routing, alpha=0 -> alpha=1 by category ────────────────
def fig_routing(df: pd.DataFrame) -> None:
    metrics = [("soft_Recall@3", "soft_Recall@3"), ("ordinal", "Judged ordinal")]
    g = df.groupby(["category", "alpha"])[[m for m, _ in metrics]].mean()
    nq = df.groupby("category")["query"].nunique()

    # One shared category order across panels, by the headline Recall@3 effect.
    delta = (g.xs(1, level="alpha")["soft_Recall@3"] - g.xs(0, level="alpha")["soft_Recall@3"])
    order = delta.sort_values(ascending=False).index.tolist()
    y = np.arange(len(order))[::-1]

    fig, axes = plt.subplots(1, len(metrics), figsize=(5.4, 3.5), sharey=True)
    for ax, (metric, label) in zip(axes, metrics):
        a0 = g.xs(0, level="alpha")[metric].loc[order].to_numpy()
        a1 = g.xs(1, level="alpha")[metric].loc[order].to_numpy()
        for yi, x0, x1 in zip(y, a0, a1):
            c = GOOD if x1 >= x0 else CRIT
            ax.annotate("", xy=(x1, yi), xytext=(x0, yi),
                        arrowprops=dict(arrowstyle="-|>,head_width=.16,head_length=.38",
                                        color=c, lw=1.5, shrinkA=2.5, shrinkB=0,
                                        alpha=.9))
            ax.plot(x0, yi, "o", ms=4.4, mfc="#fcfcfb", mec=MUTED, mew=1.1, zorder=3)
        ax.grid(True, axis="x", lw=0.6, zorder=0)
        ax.set_axisbelow(True)
        ax.set_title(label, color=INK, pad=6)
        ax.set_xlim(-0.04, 1.04)
        ax.set_xticks([0, .5, 1])
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.tick_params(axis="y", length=0)

    axes[0].set_yticks(y)
    axes[0].set_yticklabels([f"{c}  ({nq[c]}q)" for c in order], fontsize=8)
    axes[0].set_ylim(-0.8, len(order) - 0.2)

    handles = [Line2D([], [], marker="o", ls="", ms=5, mfc="#fcfcfb", mec=MUTED,
                      label=r"$\alpha$ = 0 (no routing)"),
               Line2D([], [], color=GOOD, lw=1.6, label="routing improves"),
               Line2D([], [], color=CRIT, lw=1.6, label="routing degrades")]
    fig.legend(handles=handles, loc="lower center", ncol=3, bbox_to_anchor=(0.55, -0.03))
    fig.text(0.55, -0.085, "Arrow runs from $\\alpha$ = 0 to $\\alpha$ = 1. Categories ordered "
             "by soft_Recall@3 effect.", ha="center", fontsize=7.5, color=MUTED)
    fig.tight_layout()
    fig.savefig(OUT / "routing.pdf")
    plt.close(fig)


# ── Figure 5: P(best) ────────────────────────────────────────────────────────
def fig_pbest(df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, len(METRICS), figsize=(5.4, 2.5))
    for ax, (metric, label) in zip(axes, METRICS.items()):
        s = summarise(df, metric)["p_best"].sort_values(ascending=False).head(6)[::-1]
        y = np.arange(len(s))
        ax.barh(y, s.to_numpy(), height=.62, color=BLUE, zorder=2)
        for yi, v in zip(y, s.to_numpy()):
            ax.text(v + .012, yi, f"{v:.2f}", va="center", fontsize=7.5, color=SECONDARY)
        ax.set_yticks(y, [pretty(c) for c in s.index], fontsize=7.6)
        ax.set_xlim(0, max(.72, s.max() * 1.28))
        ax.set_xticks([0, .25, .5])
        ax.grid(True, axis="x", lw=.6, zorder=0)
        ax.set_axisbelow(True)
        ax.set_title(label, color=INK, pad=6)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.tick_params(axis="y", length=0)
    fig.text(0.5, -0.07, "$P$(best) over 5,000 query bootstraps; six leading configurations "
             "per metric.", ha="center", fontsize=7.5, color=MUTED)
    fig.tight_layout()
    fig.savefig(OUT / "pbest.pdf")
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    df = pd.read_excel(SRC)
    assert df["query"].nunique() == 34 and df.config_key.nunique() == 36, "grid changed"

    fig_divergence(df)
    fig_caterpillar(df)
    fig_heatmap(df)
    fig_routing(df)
    fig_pbest(df)

    # Echo the numbers the chapter quotes, so a figure can never silently drift
    # from the tables.
    for m in METRICS:
        s = summarise(df, m).sort_values("mean", ascending=False)
        top = s.head(3)
        print(f"\n{m}: " + " | ".join(
            f"{c} {r['mean']:.3f} [{r.lo:.2f},{r.hi:.2f}] P(best)={r.p_best:.3f}"
            for c, r in top.iterrows()))
    print(f"\nwrote {len(list(OUT.glob('*.pdf')))} figures to {OUT}")


if __name__ == "__main__":
    main()
