"""
figlib.py

Shared figure-generation library for the Organ-Age publication figure pipeline.

This module provides:

* ``Paths``              – dataclass holding all input/output file paths.
* ``OA_COLORS``          – colorblind-friendly palette (Wong 2011 / Paul Tol).
* ``PAPER_STYLE``        – matplotlib rcParams dict for publication-ready styling.
* I/O helpers            – ``_ensure_outdir``, ``_savefig``, ``_read_preds``.
* Axis-level utilities   – ``_grid``, ``_identity_line``,
                           ``_data_limits_for_identity``.
* Metric helpers         – ``compute_metrics``, ``to_latex_table``,
                           ``write_metrics_table``.
* Figure functions       – ``fig_conceptual_overview``, ``fig_pred_vs_true``,
                           ``fig_age_accel``, ``fig_model_comparison``,
                           ``fig_training_curves``, ``fig_umap``.

All figure functions save a PNG to the supplied ``out_path`` and close the
matplotlib figure before returning, so they can be called in sequence without
accumulating memory.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional, Dict

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

from cycler import cycler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

try:
    import umap
except Exception:
    umap = None


# ============================
# Colorblind-friendly palette (Wong 2011 / Paul Tol)
# ============================
OA_COLORS = {
    "slate":   "#1D3557",   # deep navy (text / axes)
    "slate2":  "#457B9D",   # medium steel blue
    "blue":    "#0072B2",   # CBF blue        (primary scatter)
    "teal":    "#009E73",   # CBF bluish-green (error bars, CI)
    "sage":    "#56B4E9",   # CBF sky blue    (histogram fill)
    "amber":   "#E69F00",   # CBF orange      (secondary scatter / bars)
    "rose":    "#D55E00",   # CBF vermilion   (tertiary series)
    "purple":  "#CC79A7",   # CBF pink-purple (fourth series)
    "grid":    "#CFD8DC",   # blue-grey grid
    "bg":      "#FFFFFF",
}

OA_CYCLE = [
    OA_COLORS["blue"],
    OA_COLORS["amber"],
    OA_COLORS["teal"],
    OA_COLORS["rose"],
    OA_COLORS["sage"],
    OA_COLORS["purple"],
]


# ============================
# Paper styling – serif font, publication-legible sizes
# ============================
PAPER_STYLE = {
    # --- Typography (non-Arial serif) ---
    "font.family":      "serif",
    "font.serif":       ["DejaVu Serif", "Georgia", "Times New Roman", "serif"],
    "mathtext.fontset": "dejavuserif",

    # --- Sizes (large enough to read in a two-column layout) ---
    "font.size":         16,
    "axes.titlesize":    19,
    "axes.labelsize":    17,
    "xtick.labelsize":   15,
    "ytick.labelsize":   15,
    "legend.fontsize":   14,
    "figure.titlesize":  20,

    # --- Layout ---
    "axes.titlepad": 10,
    "axes.labelpad":  8,

    # --- Strokes ---
    "axes.linewidth":      1.2,
    "xtick.major.width":   1.1,
    "ytick.major.width":   1.1,
    "xtick.major.size":    5.0,
    "ytick.major.size":    5.0,
    "xtick.minor.size":    3.0,
    "ytick.minor.size":    3.0,

    # --- Colors ---
    "axes.edgecolor":  OA_COLORS["slate"],
    "text.color":      OA_COLORS["slate"],
    "axes.labelcolor": OA_COLORS["slate"],
    "xtick.color":     OA_COLORS["slate"],
    "ytick.color":     OA_COLORS["slate"],

    # --- Cycle ---
    "axes.prop_cycle": cycler(color=OA_CYCLE),

    # --- Save ---
    "savefig.dpi":  320,
    "figure.dpi":   120,

    # --- Legend ---
    "legend.frameon": False,
}

matplotlib.rcParams.update(PAPER_STYLE)


# -----------------------------
# Config / IO
# -----------------------------
@dataclass
class Paths:
    """Centralised container for all input/output file paths used by the pipeline.

    Attributes
    ----------
    preds_unimodal_csv : str
        CSV file with predictions from the unimodal model.
        Must contain columns ``y_true`` and ``y_pred``.
    preds_v3_csv : str
        CSV file with predictions from baseline fusion model v3.
    preds_v35_csv : str
        CSV file with predictions from aligned fusion model v3.5.
    emb_unaligned_npy : str or None
        Path to a ``.npy`` array of embeddings *before* contrastive alignment.
        Shape ``(N, D)``.  Optional; UMAP figures are skipped when ``None``.
    emb_aligned_npy : str or None
        Path to a ``.npy`` array of embeddings *after* contrastive alignment.
        Optional; same conditions as ``emb_unaligned_npy``.
    emb_meta_csv : str or None
        CSV with per-sample metadata for the embedding arrays.
        Must contain columns ``age`` and ``modality``.  Optional.
    curves_rna_csv : str or None
        CSV training-curve log for the RNA encoder.  Optional.
    curves_xray_csv : str or None
        CSV training-curve log for the X-ray encoder.  Optional.
    curves_mri_csv : str or None
        CSV training-curve log for the MRI encoder.  Optional.
    out_dir : str
        Directory where all output PNGs and the LaTeX table are written.
        Defaults to ``"figures"``.
    """

    preds_unimodal_csv: str
    preds_v3_csv: str
    preds_v35_csv: str

    emb_unaligned_npy: Optional[str]
    emb_aligned_npy: Optional[str]
    emb_meta_csv: Optional[str]  # needs 'age' and 'modality'

    curves_rna_csv: Optional[str]
    curves_xray_csv: Optional[str]
    curves_mri_csv: Optional[str]

    out_dir: str = "figures"


def _ensure_outdir(path: str) -> None:
    """Create *path* (and any missing parent directories) if it does not exist.

    Parameters
    ----------
    path : str
        Directory path to create.
    """
    os.makedirs(path, exist_ok=True)


def _savefig(path: str, dpi: int = 320) -> None:
    """Apply tight layout, save the current matplotlib figure, then close it.

    Parameters
    ----------
    path : str
        Destination file path (extension determines the format, e.g. ``.png``).
    dpi : int, optional
        Output resolution in dots-per-inch.  Defaults to 320.
    """
    plt.tight_layout()
    plt.savefig(path, dpi=dpi, bbox_inches="tight", facecolor=OA_COLORS["bg"])
    plt.close()


def _read_preds(csv_path: str) -> pd.DataFrame:
    """Load a predictions CSV and validate its required columns.

    The CSV must contain at least ``y_true`` (chronological age) and
    ``y_pred`` (model-predicted age).  If a ``modality`` column is absent,
    all rows are assigned the placeholder value ``"Unknown"`` so that
    downstream functions that group by modality still work correctly.

    Parameters
    ----------
    csv_path : str
        Path to the predictions CSV file.

    Returns
    -------
    pandas.DataFrame
        Loaded DataFrame, guaranteed to contain ``y_true``, ``y_pred``, and
        ``modality`` columns.

    Raises
    ------
    ValueError
        If any of the required columns are missing from the file.
    """
    df = pd.read_csv(csv_path)
    required = {"y_true", "y_pred"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{csv_path} missing required columns: {missing}")
    # Provide a sentinel modality so groupby logic is always safe.
    if "modality" not in df.columns:
        df["modality"] = "Unknown"
    return df


# -----------------------------
# Utilities
# -----------------------------
def _grid(ax) -> None:
    """Apply a subtle, print-friendly grid to *ax*.

    The grid is drawn below all data artists (``set_axisbelow``) to avoid
    visual clutter on scatter plots.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axes object to modify in-place.
    """
    ax.grid(True, linewidth=0.6, alpha=0.30, color=OA_COLORS["grid"])
    ax.set_axisbelow(True)


def _identity_line(ax, lo: float, hi: float) -> None:
    """Draw the identity line (y = x) as a dashed reference on *ax*.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Target axes.
    lo : float
        Lower bound of the line segment.
    hi : float
        Upper bound of the line segment.
    """
    ax.plot([lo, hi], [lo, hi], linestyle="--", linewidth=1.4,
            color=OA_COLORS["slate2"], label="Identity (y = x)")


def _data_limits_for_identity(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Compute a safe (lo, hi) range that spans both *x* and *y* arrays.

    The range is used to draw an identity line that fully covers the data
    extent.  If the computed limits are non-finite or degenerate (lo == hi),
    the function falls back to the unit interval [0, 1].

    Parameters
    ----------
    x : numpy.ndarray
        Array of values for one axis (e.g. chronological ages).
    y : numpy.ndarray
        Array of values for the other axis (e.g. predicted ages).

    Returns
    -------
    tuple[float, float]
        ``(lo, hi)`` span covering the combined range of *x* and *y*.
    """
    lo = float(np.nanmin([np.nanmin(x), np.nanmin(y)]))
    hi = float(np.nanmax([np.nanmax(x), np.nanmax(y)]))
    # Guard against all-NaN arrays or perfectly constant data.
    if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
        lo, hi = 0.0, 1.0
    return lo, hi


# -----------------------------
# Metrics
# -----------------------------
def compute_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Compute per-modality regression metrics from a predictions DataFrame.

    Groups the DataFrame by the ``modality`` column and, for each group,
    computes MAE, MSE, RMSE, and R².

    Parameters
    ----------
    df : pandas.DataFrame
        Must contain columns ``y_true``, ``y_pred``, and ``modality``.

    Returns
    -------
    pandas.DataFrame
        One row per modality with columns ``["modality", "N", "MAE", "MSE",
        "RMSE", "R2"]``, sorted by ascending MAE.
    """
    rows = []
    for mod, g in df.groupby("modality"):
        y = g["y_true"].to_numpy()
        yhat = g["y_pred"].to_numpy()
        rows.append({
            "modality": mod,
            "N": len(g),
            "MAE": float(mean_absolute_error(y, yhat)),
            "MSE": float(mean_squared_error(y, yhat)),
            # RMSE is derived from MSE to avoid a second sklearn call.
            "RMSE": float(np.sqrt(mean_squared_error(y, yhat))),
            "R2": float(r2_score(y, yhat)),
        })
    return pd.DataFrame(rows).sort_values("MAE")


def to_latex_table(metrics_df: pd.DataFrame) -> str:
    """Convert a metrics DataFrame to a LaTeX ``tabular`` string.

    Numeric columns are formatted to three decimal places before rendering so
    that the table is ready for direct inclusion in a journal manuscript via
    ``\\input{}``.

    Parameters
    ----------
    metrics_df : pandas.DataFrame
        Output of :func:`compute_metrics` (must have ``MSE``, ``MAE``, ``R2``
        columns).

    Returns
    -------
    str
        A LaTeX-formatted table string (no surrounding ``table`` environment).
    """
    cols = ["modality", "N", "MSE", "MAE", "R2"]
    m = metrics_df.copy()
    # Format floats uniformly to 3 d.p. for a clean tabular column width.
    m["MSE"] = m["MSE"].map(lambda x: f"{x:.3f}")
    m["MAE"] = m["MAE"].map(lambda x: f"{x:.3f}")
    m["R2"] = m["R2"].map(lambda x: f"{x:.3f}")
    return m[cols].to_latex(index=False, escape=True)


def write_metrics_table(df: pd.DataFrame, out_tex_path: str) -> pd.DataFrame:
    """Compute metrics, write a LaTeX table file, and return the metrics DataFrame.

    This is a convenience wrapper that chains :func:`compute_metrics` and
    :func:`to_latex_table` and persists the result to disk.

    Parameters
    ----------
    df : pandas.DataFrame
        Predictions DataFrame with columns ``y_true``, ``y_pred``, and
        ``modality``.
    out_tex_path : str
        Destination path for the ``.tex`` file.

    Returns
    -------
    pandas.DataFrame
        The per-modality metrics table (same as returned by
        :func:`compute_metrics`).
    """
    m = compute_metrics(df)
    tex = to_latex_table(m)
    with open(out_tex_path, "w", encoding="utf-8") as f:
        f.write(tex)
    return m


# -----------------------------
# Figure: conceptual overview
# -----------------------------
def fig_conceptual_overview(out_path: str) -> None:
    """Render a schematic overview of the multimodal organ-age pipeline.

    Horizontal left-to-right layout:
      [Inputs] → [Encoders] → [Contrastive Alignment] → [Fusion Transformer] → Output

    Saved at 2000 px wide (figsize 20×7 @ 100 dpi).

    Parameters
    ----------
    out_path : str
        Destination PNG path.
    """
    import matplotlib.patches as mpatches

    slate = OA_COLORS["slate"]

    fig, ax = plt.subplots(figsize=(20, 7))
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    def box(cx, cy, w, h, text, fc, fontsize=14):
        """Draw a rounded rect centred at (cx, cy) with bold label."""
        r = mpatches.FancyBboxPatch(
            (cx - w / 2, cy - h / 2), w, h,
            boxstyle="round,pad=0.012,rounding_size=0.015",
            linewidth=2.0, edgecolor=slate, facecolor=fc, zorder=3,
        )
        ax.add_patch(r)
        ax.text(cx, cy, text,
                ha="center", va="center", fontsize=fontsize,
                color=slate, fontweight="bold",
                multialignment="center", linespacing=1.5, zorder=4)

    def arrow(x1, y1, x2, y2):
        """Thick filled-head arrow from (x1,y1) → (x2,y2) in axes coords."""
        ax.annotate(
            "", xy=(x2, y2), xytext=(x1, y1),
            arrowprops=dict(arrowstyle="-|>", lw=3.5,
                            color=slate, mutation_scale=28),
            zorder=2,
        )

    # ── Geometry (horizontal, evenly spaced) ──────────────────────────────────
    # GAP = 0.095 between every pair of adjacent box edges throughout.
    # Narrower boxes (BW_s=0.12, tall boxes=0.11) so text fills the box more.
    # All five columns use the same GAP so spacing is visually equal.
    GAP   = 0.095

    BW_s  = 0.12   # small box width  (input / encoder)
    BH_s  = 0.15   # small box height
    ALN_W = 0.11   # alignment box width
    ALN_H = 0.70   # alignment box height (spans all three rows)
    FUS_W = 0.11   # fusion box width
    FUS_H = 0.58   # fusion box height
    OUT_W = 0.11   # output box width
    OUT_H = 0.42   # output box height

    x_inp = 0.02 + BW_s / 2                                # ≈ 0.080
    x_enc = x_inp + BW_s / 2 + GAP + BW_s / 2             # ≈ 0.295
    x_aln = x_enc + BW_s / 2 + GAP + ALN_W / 2            # ≈ 0.505
    x_fus = x_aln + ALN_W / 2 + GAP + FUS_W / 2           # ≈ 0.710
    x_out = x_fus + FUS_W / 2 + GAP + OUT_W / 2           # ≈ 0.915

    # Three modality rows
    y_rna  = 0.77
    y_xray = 0.50
    y_mri  = 0.23

    # ── Input data boxes (left column, stacked) ───────────────────────────────
    box(x_inp, y_rna,  BW_s, BH_s, "GTEx  RNA-seq\n7,378 samples",      "#EEF2FF", fontsize=15)
    box(x_inp, y_xray, BW_s, BH_s, "CheXpert  X-ray\n187,825 samples",  "#FFF7ED", fontsize=15)
    box(x_inp, y_mri,  BW_s, BH_s, "IXI  Brain MRI\n563 samples",       "#F0FDFA", fontsize=15)

    # ── Encoder boxes (second column, stacked) ────────────────────────────────
    box(x_enc, y_rna,  BW_s, BH_s, "MLP\nEncoder",    "#F1F5F9", fontsize=15)
    box(x_enc, y_xray, BW_s, BH_s, "ResNet\nEncoder", "#F1F5F9", fontsize=15)
    box(x_enc, y_mri,  BW_s, BH_s, "ViT\nEncoder",    "#F1F5F9", fontsize=15)

    # ── Contrastive alignment (tall, spans all rows) ──────────────────────────
    box(x_aln, y_xray, ALN_W, ALN_H,
        "Contrastive\nAlignment\n(InfoNCE)",
        "#F0FDF4", fontsize=15)

    # ── Fusion transformer (tall) ─────────────────────────────────────────────
    box(x_fus, y_xray, FUS_W, FUS_H,
        "Fusion\nTransformer\n+\nCross-Modal\nAttention",
        "#FEE2E2", fontsize=15)

    # ── Output box (boxed, matching style of all other nodes) ─────────────────
    box(x_out, y_xray, OUT_W, OUT_H,
        "Organ-Age\nPrediction\n"
        "$\\mu \\pm \\sigma$\n"
        "$\\Delta = \\hat{y} - y_{\\mathrm{chrono}}$",
        "#FEF9C3", fontsize=14)   # soft yellow to visually distinguish output

    # ── Arrows: input → encoder (horizontal per row, equal gap) ───────────────
    for yr in (y_rna, y_xray, y_mri):
        arrow(x_inp + BW_s / 2, yr, x_enc - BW_s / 2, yr)

    # ── Arrows: encoder → alignment (horizontal, fan into tall box) ───────────
    for yr in (y_rna, y_xray, y_mri):
        arrow(x_enc + BW_s / 2, yr, x_aln - ALN_W / 2, yr)

    # ── Arrow: alignment → fusion (horizontal, centre) ────────────────────────
    arrow(x_aln + ALN_W / 2, y_xray, x_fus - FUS_W / 2, y_xray)

    # ── Arrow: fusion → output ────────────────────────────────────────────────
    arrow(x_fus + FUS_W / 2, y_xray, x_out - OUT_W / 2, y_xray)

    _savefig(out_path, dpi=100)


# -----------------------------
# Figure: predicted vs true
# -----------------------------
def fig_pred_vs_true(df: pd.DataFrame, out_path: str, title: str = "") -> None:
    """Scatter plot of model-predicted age versus chronological age.

    When six or fewer modalities are present each is drawn in a distinct
    colour from the publication palette and a legend is added.  For datasets
    with more than six modalities all points are rendered in a single colour
    to avoid palette exhaustion.

    A linear regression fit line and the identity line (y = x) are overlaid
    to make systematic bias immediately visible.  Global MAE and R² are
    embedded in the default title.

    Parameters
    ----------
    df : pandas.DataFrame
        Predictions DataFrame; must have ``y_true``, ``y_pred``, and
        ``modality`` columns.
    out_path : str
        Destination PNG path.
    title : str, optional
        Custom axes title.  If empty the function generates a title that
        includes global MAE and R² values.
    """
    fig, ax = plt.subplots(figsize=(6.5, 5.5))

    mods = df["modality"].unique()
    # Use per-modality colours only when the palette can accommodate all groups.
    if len(mods) <= 6:
        for mod, g in df.groupby("modality"):
            ax.scatter(g["y_true"], g["y_pred"], s=12, alpha=0.50, label=str(mod),
                       linewidths=0)
        ax.legend(frameon=False, fontsize=13)
    else:
        ax.scatter(df["y_true"], df["y_pred"], s=12, alpha=0.50,
                   color=OA_COLORS["blue"], linewidths=0)

    # Fit a linear trend to show any systematic age-dependent bias.
    X = df["y_true"].to_numpy().reshape(-1, 1)
    y = df["y_pred"].to_numpy()
    lr = LinearRegression().fit(X, y)

    xs = np.linspace(df["y_true"].min(), df["y_true"].max(), 200)
    ys = lr.predict(xs.reshape(-1, 1))
    ax.plot(xs, ys, linewidth=2.2, color=OA_COLORS["slate"], label="Regression fit")
    lo, hi = _data_limits_for_identity(xs, ys)
    _identity_line(ax, lo, hi)

    mae = mean_absolute_error(df["y_true"], df["y_pred"])
    r2 = r2_score(df["y_true"], df["y_pred"])
    ax.set_xlabel("Chronological Age (Years)")
    ax.set_ylabel("Predicted Age (Years)")
    ax.set_title(title or f"Predicted vs True Age  (MAE = {mae:.2f} yr,  R² = {r2:.2f})")

    _grid(ax)
    _savefig(out_path)


# -----------------------------
# Figure: age acceleration Δ
# -----------------------------
def fig_age_accel(df: pd.DataFrame, out_path: str, title: str = "") -> None:
    """Histogram of biological age acceleration (Δ = predicted − chronological).

    A dashed vertical reference line at Δ = 0 marks the boundary between
    accelerated (Δ > 0) and decelerated (Δ < 0) ageing.

    Parameters
    ----------
    df : pandas.DataFrame
        Predictions DataFrame with columns ``y_true`` and ``y_pred``.
    out_path : str
        Destination PNG path.
    title : str, optional
        Custom axes title.
    """
    fig, ax = plt.subplots(figsize=(6.5, 4.5))

    # Age acceleration: positive = biologically older than chronological age.
    delta = df["y_pred"] - df["y_true"]
    ax.hist(delta, bins=55, color=OA_COLORS["sage"], alpha=0.88, edgecolor="none")
    ax.axvline(0.0, linewidth=1.6, color=OA_COLORS["slate"], linestyle="--",
               label="Zero (no acceleration)")

    ax.set_xlabel(r"Age Acceleration  $\Delta = \hat{y} - y$  (Years)")
    ax.set_ylabel("Count")
    ax.set_title(title or "Distribution of Biological Age Acceleration")
    ax.legend()

    _grid(ax)
    _savefig(out_path)


# -----------------------------
# Figure: model comparison MAE
# -----------------------------
def fig_model_comparison(mae_dict: Dict[str, float], out_path: str, title: str = "") -> None:
    """Bar chart comparing MAE across model variants.

    Each bar is labelled with its numeric MAE value to avoid readers needing
    to read the y-axis precisely.  Bar colours cycle through the
    publication-safe ``OA_CYCLE`` palette.

    Parameters
    ----------
    mae_dict : dict[str, float]
        Mapping from model name (e.g. ``"Unimodal"``, ``"v3"``, ``"v3.5"``)
        to its mean absolute error in years.
    out_path : str
        Destination PNG path.
    title : str, optional
        Custom axes title.
    """
    fig, ax = plt.subplots(figsize=(6.5, 4.5))

    names = list(mae_dict.keys())
    vals = [mae_dict[k] for k in names]
    # Assign a distinct palette colour to each bar, cycling if needed.
    colors = [OA_CYCLE[i % len(OA_CYCLE)] for i in range(len(names))]
    bars = ax.bar(names, vals, color=colors, alpha=0.88, edgecolor=OA_COLORS["slate"],
                  linewidth=0.8)

    # Value labels on top of bars — positioned 0.05 years above each bar.
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.05,
                f"{v:.2f}", ha="center", va="bottom", fontsize=13,
                color=OA_COLORS["slate"])

    ax.set_ylabel("MAE (Years)")
    ax.set_title(title or "Model Comparison — Mean Absolute Error (lower is better)")
    plt.xticks(rotation=20, ha="right")

    _grid(ax)
    _savefig(out_path)


# -----------------------------
# Figure: training curves
# -----------------------------
def fig_training_curves(curves_csv: str, out_path: str, title: str) -> None:
    """Line plot of training (and optionally validation) loss over epochs/steps.

    The function auto-detects the x-axis column by looking for ``"epoch"``
    first, then ``"step"``.  It plots whichever loss columns are present:
    ``train_loss``, ``val_loss``, or a generic ``loss`` column (the latter
    only when neither of the more specific columns exists).

    Parameters
    ----------
    curves_csv : str
        Path to a CSV log file containing at minimum an ``epoch`` or ``step``
        column and at least one of ``train_loss``, ``val_loss``, or ``loss``.
    out_path : str
        Destination PNG path.
    title : str
        Axes title (e.g. ``"RNA encoder training curves"``).

    Raises
    ------
    ValueError
        If neither ``epoch`` nor ``step`` column is found in the CSV.
    """
    df = pd.read_csv(curves_csv)
    # Prefer epoch-level granularity; fall back to step if epoch is absent.
    xcol = "epoch" if "epoch" in df.columns else ("step" if "step" in df.columns else None)
    if xcol is None:
        raise ValueError(f"{curves_csv} must contain 'epoch' or 'step' column.")

    fig, ax = plt.subplots(figsize=(6.5, 4.5))

    if "train_loss" in df.columns:
        ax.plot(df[xcol], df["train_loss"], label="Train", linewidth=2.2,
                color=OA_COLORS["blue"])
    if "val_loss" in df.columns:
        ax.plot(df[xcol], df["val_loss"], label="Validation", linewidth=2.2,
                color=OA_COLORS["rose"])
    # Only use generic 'loss' column when the more specific columns are absent,
    # to avoid plotting the same series twice.
    if ("loss" in df.columns) and ("train_loss" not in df.columns) and ("val_loss" not in df.columns):
        ax.plot(df[xcol], df["loss"], label="Loss", linewidth=2.2,
                color=OA_COLORS["slate"])

    ax.set_xlabel(xcol.capitalize())
    ax.set_ylabel("Loss")
    ax.set_title(title)
    ax.legend()

    _grid(ax)
    _savefig(out_path)


# -----------------------------
# Figure: UMAP
# -----------------------------
def fig_umap(
    emb: np.ndarray,
    meta: pd.DataFrame,
    out_path: str,
    title: str,
    color_by: str = "age",
    n_neighbors: int = 30,
    min_dist: float = 0.1
) -> None:
    """Project high-dimensional embeddings with UMAP and render a 2-D scatter.

    When ``color_by="age"`` the scatter uses the viridis continuous colormap
    with a colorbar.  For any other column name the categories are mapped to
    the publication discrete palette and a legend is added.

    Parameters
    ----------
    emb : numpy.ndarray, shape (N, D)
        High-dimensional embedding matrix.  Each row is one sample.
    meta : pandas.DataFrame
        Per-sample metadata DataFrame aligned row-for-row with *emb*.
        Must contain the column specified by ``color_by``.
    out_path : str
        Destination PNG path.
    title : str
        Axes title.
    color_by : str, optional
        Name of the ``meta`` column to use for colouring points.
        Use ``"age"`` for a continuous colour scale or any categorical column
        name (e.g. ``"modality"``) for discrete colour coding.
        Defaults to ``"age"``.
    n_neighbors : int, optional
        UMAP neighbourhood size; controls the balance between local and global
        structure.  Defaults to 30.
    min_dist : float, optional
        UMAP minimum distance parameter; smaller values allow tighter cluster
        packing.  Defaults to 0.1.

    Raises
    ------
    ImportError
        If the ``umap-learn`` package is not installed.
    """
    if umap is None:
        raise ImportError("umap-learn not installed. Run: pip install umap-learn")

    # Fix the random seed for reproducible layouts across runs.
    reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, random_state=42)
    proj = reducer.fit_transform(emb)

    fig, ax = plt.subplots(figsize=(6.5, 5.5))

    if color_by == "age":
        # Continuous age variable: viridis colormap + colorbar.
        c = meta["age"].to_numpy()
        sc = ax.scatter(proj[:, 0], proj[:, 1], c=c, cmap="viridis",
                        s=8, alpha=0.75, linewidths=0)
        cb = plt.colorbar(sc, ax=ax)
        cb.set_label("Age (Years)", fontsize=14)
    else:
        # Categorical variable: assign one palette colour per unique category.
        cats = meta[color_by].astype(str)
        uniq = list(pd.unique(cats))
        # Extend palette with slate as a fallback for extra categories.
        palette = OA_CYCLE + [OA_COLORS["slate"]]
        color_map = {u: palette[i % len(palette)] for i, u in enumerate(uniq)}
        for u in uniq:
            idx = np.where(cats.to_numpy() == u)[0]
            ax.scatter(proj[idx, 0], proj[idx, 1], s=8, alpha=0.80,
                       label=u, color=color_map[u], linewidths=0)
        ax.legend(frameon=False, fontsize=12, ncol=2)

    ax.set_title(title)
    ax.set_xlabel("UMAP Dimension 1")
    ax.set_ylabel("UMAP Dimension 2")

    _grid(ax)
    _savefig(out_path)
