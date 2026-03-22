"""
v4_5_plot_top_genes.py
======================
Visualise the top-K gene-level importance scores for each of the four
primary organs: liver, heart, kidney, and brain cortex.

Input data comes from the per-organ ``gene_importance_<organ>.csv`` files
produced by ``v4_5_ig_to_genes.py`` (stored under
``analysis/v4_5_gene_importance/``).

Three publication figures are generated:

1. **Individual organ plots** – one horizontal bar chart per organ saved as
   ``figures/v4_5/top_genes/top_<K>_genes_<organ>.png``.

2. **2x2 panel** – all four organs in a single compound figure for
   side-by-side comparison (``top_<K>_genes_panel_2x2.png``).

3. **Rank overlay** – importance score vs. rank for each organ overlaid on
   one line chart (``top_<K>_genes_rank_overlay.png``).

A combined CSV of all top-K genes across all organs is also written to
``analysis/v4_5_top_genes_table/v4_5_top20_genes_all_organs.csv``.
"""
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from pathlib import Path

# ============================
# Colorblind-friendly palette (Wong 2011)
# ============================
OA_COLORS = {
    "slate":  "#1D3557",
    "slate2": "#457B9D",
    "blue":   "#0072B2",
    "teal":   "#009E73",
    "sage":   "#56B4E9",
    "amber":  "#E69F00",
    "rose":   "#D55E00",
    "purple": "#CC79A7",
    "grid":   "#CFD8DC",
    "bg":     "#FFFFFF",
}

ORGAN_COLORS = [
    OA_COLORS["blue"],
    OA_COLORS["amber"],
    OA_COLORS["teal"],
    OA_COLORS["rose"],
]

PAPER_STYLE = {
    "font.family":      "serif",
    "font.serif":       ["DejaVu Serif", "Georgia", "Times New Roman", "serif"],
    "mathtext.fontset": "dejavuserif",
    "font.size":         16,
    "axes.titlesize":    18,
    "axes.labelsize":    16,
    "xtick.labelsize":   14,
    "ytick.labelsize":   13,
    "legend.fontsize":   13,
    "figure.titlesize":  20,
    "axes.titlepad":     10,
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


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

IMPORT_DIR = Path("analysis/v4_5_gene_importance")
OUT_DIR_FIG = Path("figures/v4_5/top_genes")
OUT_DIR_TAB = Path("analysis/v4_5_top_genes_table")

ORGANS = [
    "liver",
    "heart",
    "kidney",
    "brain_cortex",
]

TOP_K = 20


# ---------------------------------------------------------------------
# Column detection helpers
# ---------------------------------------------------------------------

def detect_score_column(df: pd.DataFrame) -> str:
    """
    Heuristically detect the importance-score column in a gene CSV.

    Checks for common column names (case-insensitive) in order of preference.
    Falls back to the first numeric column if none of the preferred names are
    found.

    Parameters
    ----------
    df : pd.DataFrame
        Gene importance DataFrame loaded from CSV.

    Returns
    -------
    str
        Name of the column to use as the importance score.

    Raises
    ------
    RuntimeError
        If no numeric columns are found in ``df``.
    """
    candidates = ["score_abs", "score", "importance", "ig", "value"]
    lower_cols = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand in lower_cols:
            return lower_cols[cand]
    num_cols = df.select_dtypes(include="number").columns
    if len(num_cols) == 0:
        raise RuntimeError(f"No numeric columns found. Columns = {list(df.columns)}")
    if len(num_cols) > 1:
        print(f"[WARN] Multiple numeric columns {list(num_cols)}; using '{num_cols[0]}'.")
    return num_cols[0]


def detect_gene_column(df: pd.DataFrame) -> str:
    """
    Heuristically detect the gene-name column in a gene CSV.

    Checks for preferred column names (case-insensitive) first, then falls
    back to the first non-numeric column.

    Parameters
    ----------
    df : pd.DataFrame
        Gene importance DataFrame loaded from CSV.

    Returns
    -------
    str
        Name of the column containing gene symbols / names.

    Raises
    ------
    RuntimeError
        If no non-numeric columns are found in ``df``.
    """
    candidates = ["gene", "gene_name"]
    lower_cols = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand in lower_cols:
            return lower_cols[cand]
    non_num_cols = [c for c in df.columns if c not in df.select_dtypes(include="number").columns]
    if not non_num_cols:
        raise RuntimeError(f"No non-numeric columns found. Columns = {list(df.columns)}")
    if len(non_num_cols) > 1:
        print(f"[WARN] Multiple non-numeric columns {non_num_cols}; using '{non_num_cols[0]}'.")
    return non_num_cols[0]


# ---------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------

def load_top_genes(organ: str, k: int = TOP_K):
    """
    Load and return the top-K genes by importance score for a given organ.

    Reads the gene importance CSV, detects the score and gene-name columns
    automatically, sorts by score descending, and returns the top-K rows
    with standardised column names (``gene``, ``score``) plus an ``organ``
    label column.

    Parameters
    ----------
    organ : str
        Organ name (e.g. ``'liver'``); used to locate the CSV file.
    k : int
        Number of top genes to return.

    Returns
    -------
    pd.DataFrame
        Columns: ``['gene', 'score', 'organ']``, top-K rows sorted by score
        descending.

    Raises
    ------
    FileNotFoundError
        If the CSV for ``organ`` does not exist under ``IMPORT_DIR``.
    """
    csv_path = IMPORT_DIR / f"gene_importance_{organ}.csv"
    if not csv_path.exists():
        raise FileNotFoundError(f"Could not find {csv_path}")

    df = pd.read_csv(csv_path)
    print(f"[LOAD] {organ}: columns = {list(df.columns)}")

    score_col = detect_score_column(df)
    gene_col  = detect_gene_column(df)

    print(f"[INFO] score='{score_col}', gene='{gene_col}' for organ={organ}")

    df = df[[gene_col, score_col]].copy()
    df = df.sort_values(score_col, ascending=False).head(k).copy()
    df.rename(columns={gene_col: "gene", score_col: "score"}, inplace=True)
    df["organ"] = organ
    return df


# ---------------------------------------------------------------------
# Plot functions
# ---------------------------------------------------------------------

def _grid(ax):
    """
    Apply a subtle horizontal grid to ``ax`` for readability.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Target axes object.
    """
    ax.grid(True, axis="x", linewidth=0.6, alpha=0.30, color=OA_COLORS["grid"])
    ax.set_axisbelow(True)


def plot_top_genes(df: pd.DataFrame, organ: str, outdir: Path = OUT_DIR_FIG,
                   color: str | None = None):
    """
    Render and save a horizontal bar chart of the top genes for one organ.

    Parameters
    ----------
    df : pd.DataFrame
        Top-K gene data with columns ``['gene', 'score']``, sorted by score
        descending.
    organ : str
        Organ name used for the plot title and output filename.
    outdir : Path
        Directory where the PNG will be saved.
    color : str or None
        Hex colour string for the bars; defaults to ``OA_COLORS['blue']``
        if ``None``.
    """
    outdir.mkdir(parents=True, exist_ok=True)
    bar_color = color or OA_COLORS["blue"]

    fig, ax = plt.subplots(figsize=(10, 6.5), constrained_layout=True)
    ax.barh(df["gene"], df["score"], color=bar_color, alpha=0.88, edgecolor="none")
    ax.invert_yaxis()

    ax.set_xlabel("Importance Score (IG-projected)")
    ax.set_ylabel("Gene")
    ax.set_title(f"Top {TOP_K} Genes - {organ.replace('_', ' ').title()} Biological Age")
    _grid(ax)

    outfile = outdir / f"top_{TOP_K}_genes_{organ}.png"
    fig.savefig(outfile, dpi=320, bbox_inches="tight", facecolor=OA_COLORS["bg"])
    plt.close(fig)
    print(f"[PLOT] Saved -> {outfile}")


def plot_panel(top_genes_dict):
    """2x2 publication panel for the 4 organs in ORGANS."""
    OUT_DIR_FIG.mkdir(parents=True, exist_ok=True)
    n_organs = len(ORGANS)

    fig, axes = plt.subplots(2, 2, figsize=(16, 12), constrained_layout=True)
    axes = axes.flatten()

    for i, organ in enumerate(ORGANS):
        ax = axes[i]
        df = top_genes_dict[organ]
        color = ORGAN_COLORS[i % len(ORGAN_COLORS)]

        ax.barh(df["gene"], df["score"], color=color, alpha=0.88, edgecolor="none")
        ax.invert_yaxis()
        ax.set_title(organ.replace("_", " ").title(), fontsize=18)
        ax.set_xlabel("Importance Score", fontsize=15)
        ax.set_ylabel("Gene", fontsize=15)

        for label in ax.get_yticklabels():
            label.set_fontsize(11)
        for label in ax.get_xticklabels():
            label.set_fontsize(12)

        _grid(ax)

    for j in range(n_organs, len(axes)):
        fig.delaxes(axes[j])

    fig.suptitle(f"Top {TOP_K} Genes Per Organ - Biological Age Attribution (v4.5)",
                 fontsize=20)

    panel_path = OUT_DIR_FIG / f"top_{TOP_K}_genes_panel_2x2.png"
    fig.savefig(panel_path, dpi=320, bbox_inches="tight", facecolor=OA_COLORS["bg"])
    plt.close(fig)
    print(f"[PANEL] Saved 2x2 panel -> {panel_path}")


def plot_rank_overlay(top_genes_dict):
    """
    Overlay plot of importance score by rank for each organ.

    Renders one line per organ on a single axes, with rank on the x-axis
    and importance score on the y-axis.  Saved to
    ``OUT_DIR_FIG/top_<TOP_K>_genes_rank_overlay.png``.

    Parameters
    ----------
    top_genes_dict : dict[str, pd.DataFrame]
        Mapping of organ name -> top-K gene DataFrame (columns: gene, score).
    """
    OUT_DIR_FIG.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9.4, 5.8), constrained_layout=True)

    for i, organ in enumerate(ORGANS):
        df = top_genes_dict[organ].sort_values("score", ascending=False).reset_index(drop=True)
        rank = range(1, len(df) + 1)
        color = ORGAN_COLORS[i % len(ORGAN_COLORS)]
        label = organ.replace("_", " ").title()
        ax.plot(
            rank,
            df["score"],
            marker="o",
            markersize=4,
            linewidth=2.2,
            color=color,
            alpha=0.95,
            label=label,
        )

    ax.set_xlabel(f"Rank (1 = highest importance, top {TOP_K})")
    ax.set_ylabel("Importance Score")
    ax.set_title("Gene Importance Rank Overlay Across Organs")
    ax.set_xticks(range(1, TOP_K + 1, 2))
    _grid(ax)
    ax.legend(loc="upper right", ncol=2, fontsize=12)

    overlay_path = OUT_DIR_FIG / f"top_{TOP_K}_genes_rank_overlay.png"
    fig.savefig(overlay_path, dpi=320, bbox_inches="tight", facecolor=OA_COLORS["bg"])
    plt.close(fig)
    print(f"[OVERLAY] Saved rank overlay -> {overlay_path}")


def main():
    """
    Entry point: load top-K genes for all organs, generate per-organ bar
    charts, the 2x2 panel, the rank overlay, and the combined CSV table.
    """
    combined_rows = []
    top_genes_per_organ = {}

    for i, organ in enumerate(ORGANS):
        print(f"[PROC] {organ}")
        top_df = load_top_genes(organ, TOP_K)
        combined_rows.append(top_df)
        top_genes_per_organ[organ] = top_df
        plot_top_genes(top_df, organ, color=ORGAN_COLORS[i % len(ORGAN_COLORS)])

    OUT_DIR_TAB.mkdir(parents=True, exist_ok=True)
    combined = pd.concat(combined_rows, axis=0, ignore_index=True)
    table_path = OUT_DIR_TAB / "v4_5_top20_genes_all_organs.csv"
    combined.to_csv(table_path, index=False)
    print(f"[TABLE] Saved combined CSV -> {table_path}")

    print("[PANEL] Building 2x2 panel figure...")
    plot_panel(top_genes_per_organ)
    plot_rank_overlay(top_genes_per_organ)

    print("[DONE] All figures and tables generated.")


if __name__ == "__main__":
    main()
