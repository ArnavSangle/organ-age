"""
v4_5_plot_ig_liver_rna.py
==========================
Plot the top-K latent RNA features driving liver biological age predictions,
with a secondary cumulative attribution overlay.

Given an IG CSV produced by ``v4_5_explain_ig.py`` (columns: ``feature``,
``mean_abs_ig``), this script:

  1. Selects the top ``--top_k`` latent dimensions by mean absolute IG.
  2. Renders a bar chart of ``mean_abs_ig`` per latent feature.
  3. Overlays the cumulative attribution fraction (running sum / total sum)
     on a secondary y-axis as a line plot with circle markers.

This dual-axis figure makes it easy to see both which specific latent
dimensions are most important and how quickly the attribution mass is
concentrated in the top features.

Output is saved to ``figures/v4_5/ig_liver_rna_top<K>.png``.
"""
import argparse
from pathlib import Path

import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

# ============================
# Colorblind-friendly palette (Wong 2011)
# ============================
OA_COLORS = {
    "slate":  "#1D3557",
    "slate2": "#457B9D",
    "blue":   "#0072B2",
    "teal":   "#009E73",
    "amber":  "#E69F00",
    "grid":   "#CFD8DC",
    "bg":     "#FFFFFF",
}

PAPER_STYLE = {
    "font.family":      "serif",
    "font.serif":       ["DejaVu Serif", "Georgia", "Times New Roman", "serif"],
    "mathtext.fontset": "dejavuserif",
    "font.size":         16,
    "axes.titlesize":    18,
    "axes.labelsize":    16,
    "xtick.labelsize":   14,
    "ytick.labelsize":   14,
    "figure.titlesize":  19,
    "axes.titlepad":     10,
    "axes.labelpad":      8,
    "axes.linewidth":     1.2,
    "xtick.major.width":  1.1,
    "ytick.major.width":  1.1,
    "xtick.major.size":   5.0,
    "ytick.major.size":   5.0,
    "axes.edgecolor":  OA_COLORS["slate"],
    "text.color":      OA_COLORS["slate"],
    "axes.labelcolor": OA_COLORS["slate"],
    "xtick.color":     OA_COLORS["slate"],
    "ytick.color":     OA_COLORS["slate"],
    "legend.frameon":  False,
    "savefig.dpi":     320,
    "figure.dpi":      120,
}
matplotlib.rcParams.update(PAPER_STYLE)


def main():
    """
    Load a liver IG CSV, select the top-K features, and render the
    bar+cumulative-overlay figure.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--ig_path", type=str,
                        default="data/analysis/v4_5_ig_liver_rna.csv")
    parser.add_argument("--top_k",   type=int, default=20)
    parser.add_argument("--out_png", type=str,
                        default="figures/v4_5/ig_liver_rna_top20.png")
    args = parser.parse_args()

    ig_path = Path(args.ig_path)
    df = pd.read_csv(ig_path)
    df = df.sort_values("mean_abs_ig", ascending=False).head(args.top_k)

    out_png = Path(args.out_png)
    out_png.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10.4, 5.8), constrained_layout=True)
    bar_color = OA_COLORS["blue"]
    line_color = OA_COLORS["amber"]

    values = df["mean_abs_ig"].to_numpy()
    ax.bar(range(len(df)), values,
           color=bar_color, alpha=0.88, edgecolor="none")
    ax.set_xticks(range(len(df)))
    ax.set_xticklabels(df["feature"], rotation=45, ha="right")
    ax.set_ylabel("Mean |Integrated Gradient|")
    ax.set_title("Top Latent RNA Features Driving Liver Biological Age (with Cumulative Overlay)")

    # Grid on y-axis
    ax.grid(True, axis="y", linewidth=0.6, alpha=0.30, color=OA_COLORS["grid"])
    ax.set_axisbelow(True)

    # Overlay cumulative attribution on secondary axis.
    cum = np.cumsum(values) / np.sum(values)
    ax2 = ax.twinx()
    ax2.plot(range(len(df)), cum, marker="o", color=line_color, linewidth=2.1, markersize=3.8)
    ax2.set_ylabel("Cumulative Attribution Fraction", color=line_color)
    ax2.tick_params(axis="y", colors=line_color)
    ax2.set_ylim(0.0, 1.05)

    fig.savefig(out_png, dpi=320, bbox_inches="tight", facecolor=OA_COLORS["bg"])
    plt.close(fig)
    print(f"[PLOT IG] Saved -> {out_png}")


if __name__ == "__main__":
    main()
