"""
v4_5_plot_ig_panels.py
=======================
Generate publication-quality Integrated Gradient (IG) visualisations for
the v4.5 model's latent-space feature importance, covering three organs:
Liver, Brain Cortex, and Kidney.

Two figures are produced:

1. **Three-panel bar chart** (``ig_three_panel_rna.png``) – one horizontal
   bar chart per organ showing the top ``--top_n`` latent dimensions ranked
   by mean absolute IG.

2. **Rank overlay line plot** (``ig_three_overlay_rna.png``) – IG magnitude
   vs. rank plotted as overlapping lines for all three organs, enabling
   visual comparison of attribution concentration.

All figures use a colorblind-friendly palette (Wong 2011) and a serif
paper style suitable for academic publications.

Requires three pre-computed IG CSV files (produced by
``v4_5_explain_ig.py``) with columns ``feature`` and ``mean_abs_ig``.
"""
import argparse
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd

# ============================
# Colorblind-friendly palette (Wong 2011)
# ============================
OA_COLORS = {
    "slate":  "#1D3557",
    "slate2": "#457B9D",
    "blue":   "#0072B2",
    "teal":   "#009E73",
    "amber":  "#E69F00",
    "rose":   "#D55E00",
    "grid":   "#CFD8DC",
    "bg":     "#FFFFFF",
}

PANEL_COLORS = [OA_COLORS["blue"], OA_COLORS["amber"], OA_COLORS["teal"]]

PAPER_STYLE = {
    "font.family":      "serif",
    "font.serif":       ["DejaVu Serif", "Georgia", "Times New Roman", "serif"],
    "mathtext.fontset": "dejavuserif",
    "font.size":         16,
    "axes.titlesize":    17,
    "axes.labelsize":    15,
    "xtick.labelsize":   13,
    "ytick.labelsize":   13,
    "figure.titlesize":  19,
    "axes.titlepad":      9,
    "axes.labelpad":      7,
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


def _grid(ax):
    """
    Apply a subtle horizontal grid to ``ax`` for readability.

    Enables x-axis grid lines with reduced opacity and thickness, and
    ensures bars are drawn on top of the grid.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Target axes object.
    """
    ax.grid(True, axis="x", linewidth=0.6, alpha=0.30, color=OA_COLORS["grid"])
    ax.set_axisbelow(True)


def load_top_ig(csv_path: Path, top_n: int = 15):
    """
    Load an IG CSV and return the top ``top_n`` rows by ``mean_abs_ig``.

    Parses the latent dimension index from the ``feature`` column (e.g.
    ``'z_rna_224'`` -> ``dim=224``).

    Parameters
    ----------
    csv_path : Path
        Path to the IG CSV with columns ``['feature', 'mean_abs_ig']``.
    top_n : int
        Number of top-ranked dimensions to return.

    Returns
    -------
    pd.DataFrame
        Filtered DataFrame with columns ``['feature', 'mean_abs_ig', 'dim']``,
        sorted by ``mean_abs_ig`` descending.

    Raises
    ------
    RuntimeError
        If the required columns are not present in the CSV.
    """
    df = pd.read_csv(csv_path)
    if "mean_abs_ig" not in df.columns or "feature" not in df.columns:
        raise RuntimeError(
            f"{csv_path} missing required columns ['feature', 'mean_abs_ig']."
        )
    df = df.sort_values("mean_abs_ig", ascending=False).head(top_n).copy()
    # Parse latent dimension index from e.g. "z_rna_224"
    df["dim"] = (
        df["feature"]
        .str.replace("z_rna_", "", regex=False)
        .astype(int)
    )
    return df


def plot_rank_overlay(liver_df: pd.DataFrame, brain_df: pd.DataFrame, kidney_df: pd.DataFrame, out_path: Path):
    """
    Plot IG magnitude vs. rank for three organs on a single overlaid line chart.

    Each organ is drawn as a separate line with a distinct colorblind-safe
    colour.  The x-axis shows rank (1 = highest attribution) and the y-axis
    shows mean absolute IG.

    Parameters
    ----------
    liver_df : pd.DataFrame
        Top-IG rows for liver (output of ``load_top_ig``).
    brain_df : pd.DataFrame
        Top-IG rows for brain cortex.
    kidney_df : pd.DataFrame
        Top-IG rows for kidney.
    out_path : Path
        Destination path for the saved PNG file.
    """
    fig, ax = plt.subplots(figsize=(9.4, 5.8), constrained_layout=True)
    series = [
        ("Liver", liver_df, PANEL_COLORS[0]),
        ("Brain Cortex", brain_df, PANEL_COLORS[1]),
        ("Kidney", kidney_df, PANEL_COLORS[2]),
    ]

    for label, df, color in series:
        d = df.sort_values("mean_abs_ig", ascending=False).reset_index(drop=True)
        rank = range(1, len(d) + 1)
        ax.plot(rank, d["mean_abs_ig"], marker="o", markersize=4, linewidth=2.2,
                color=color, alpha=0.95, label=label)

    ax.set_xlabel("Rank (1 = highest attribution)")
    ax.set_ylabel("Mean |Integrated Gradient|")
    ax.set_title("Latent Attribution Rank Overlay Across Organs")
    ax.set_xticks(range(1, len(liver_df) + 1, 2))
    _grid(ax)
    ax.legend(loc="upper right", ncol=2, fontsize=12)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=320, bbox_inches="tight", facecolor=OA_COLORS["bg"])
    plt.close(fig)
    print(f"[IG PLOT] Saved overlay -> {out_path}")


def plot_three_panel(
    liver_csv: Path,
    brain_csv: Path,
    kidney_csv: Path,
    out_path: Path,
    top_n: int = 15,
):
    """
    Build and save the three-panel horizontal bar chart and rank overlay plot.

    Loads the top ``top_n`` IG rows for each organ, renders a 1x3 subplot
    grid of horizontal bar charts (one panel per organ), saves it, and then
    calls ``plot_rank_overlay`` for the companion overlay figure.

    Parameters
    ----------
    liver_csv : Path
        Path to the liver IG CSV.
    brain_csv : Path
        Path to the brain cortex IG CSV.
    kidney_csv : Path
        Path to the kidney IG CSV.
    out_path : Path
        Destination path for the three-panel PNG.
    top_n : int
        Number of top latent dimensions to display per organ.
    """
    liver_df  = load_top_ig(liver_csv,  top_n=top_n)
    brain_df  = load_top_ig(brain_csv,  top_n=top_n)
    kidney_df = load_top_ig(kidney_csv, top_n=top_n)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5), constrained_layout=True)

    panels = [
        ("Liver (RNA)",          liver_df,  axes[0], PANEL_COLORS[0]),
        ("Brain Cortex (RNA)",   brain_df,  axes[1], PANEL_COLORS[1]),
        ("Kidney (RNA)",         kidney_df, axes[2], PANEL_COLORS[2]),
    ]

    for title, df, ax, color in panels:
        df = df.sort_values("mean_abs_ig", ascending=True)
        ax.barh(
            df["dim"].astype(str),
            df["mean_abs_ig"],
            color=color, alpha=0.88, edgecolor="none",
        )
        ax.set_title(title)
        ax.set_xlabel("Mean |Integrated Gradient|")
        ax.set_ylabel("Latent Dim  (z_rna_k)")
        _grid(ax)

    fig.suptitle(
        "Latent-Space Feature Importance for Organ-Specific Biological Age",
        fontsize=19,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=320, bbox_inches="tight", facecolor=OA_COLORS["bg"])
    plt.close(fig)
    print(f"[IG PLOT] Saved 3-panel -> {out_path}")

    overlay_path = out_path.parent / "ig_three_overlay_rna.png"
    plot_rank_overlay(liver_df, brain_df, kidney_df, overlay_path)


def main():
    """
    Command-line entry point for the IG panel plot generator.

    Parses paths to the three organ IG CSVs, the top-N count, and the
    output path, then calls ``plot_three_panel``.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--liver_csv",  type=str,
                        default="data/analysis/v4_5_ig_liver_rna.csv")
    parser.add_argument("--brain_csv",  type=str,
                        default="data/analysis/v4_5_ig_brain_cortex_rna.csv")
    parser.add_argument("--kidney_csv", type=str,
                        default="data/analysis/v4_5_ig_kidney_rna.csv")
    parser.add_argument("--top_n",      type=int, default=15)
    parser.add_argument("--out_path",   type=str,
                        default="figures/v4_5/ig_three_panel_rna.png")
    args = parser.parse_args()

    plot_three_panel(
        Path(args.liver_csv),
        Path(args.brain_csv),
        Path(args.kidney_csv),
        Path(args.out_path),
        top_n=args.top_n,
    )


if __name__ == "__main__":
    main()
