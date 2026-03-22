#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""
71_make_organ_age_figures_v1.py

Generates Organ-Age paper-ready figure set from parquet outputs:
- Normative organ-level: predicted vs chronological, age-gap distribution, gap vs chrono
- Calibrated organ-level: full + per-organ PANELS + overlay histograms
- V4 panel: bio-age vs chrono, delta distribution, delta vs chrono, burden + z-score PANEL

Outputs:
C:\Users\busyp\organ-age\figures\organ_age\...
"""

from __future__ import annotations

import os
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

from cycler import cycler


# ============================
# Colorblind-friendly palette (Wong 2011 / Paul Tol)
# ============================
OA_COLORS = {
    "slate":   "#1D3557",   # deep navy (text / axes)
    "slate2":  "#457B9D",   # medium steel blue
    "blue":    "#0072B2",   # CBF blue        (primary scatter)
    "teal":    "#009E73",   # CBF bluish-green (error bars, CI)
    "sage":    "#56B4E9",   # CBF sky blue    (histogram fill)
    "amber":   "#E69F00",   # CBF orange      (secondary series)
    "rose":    "#D55E00",   # CBF vermilion   (third series)
    "purple":  "#CC79A7",   # CBF pink-purple  (fourth series)
    "grid":    "#CFD8DC",   # blue-grey grid
    "bg":      "#FFFFFF",
}

# Cycle for multi-organ overlays: 4 CBF-safe colours
OA_CYCLE = [
    OA_COLORS["blue"],
    OA_COLORS["amber"],
    OA_COLORS["teal"],
    OA_COLORS["rose"],
    OA_COLORS["sage"],
    OA_COLORS["purple"],
]

# One distinct colour per organ for overlay panels
ORGAN_COLORS = OA_CYCLE


# ============================
# Paper-ready styling - serif, large fonts
# ============================
PAPER_STYLE = {
    # --- Non-Arial serif typography ---
    "font.family":      "serif",
    "font.serif":       ["DejaVu Serif", "Georgia", "Times New Roman", "serif"],
    "mathtext.fontset": "dejavuserif",

    # --- Sizes: large enough for two-column layout ---
    "font.size":         16,
    "figure.dpi":        120,
    "savefig.dpi":       320,

    "axes.titlesize":    19,
    "axes.labelsize":    17,
    "axes.titlepad":     10,
    "axes.labelpad":      8,
    "axes.linewidth":     1.2,

    "xtick.labelsize":   15,
    "ytick.labelsize":   15,
    "xtick.major.width":  1.1,
    "ytick.major.width":  1.1,
    "xtick.major.size":   5.0,
    "ytick.major.size":   5.0,

    "legend.fontsize":   14,
    "legend.frameon":    False,
    "figure.titlesize":  20,

    "axes.edgecolor":  OA_COLORS["slate"],
    "text.color":      OA_COLORS["slate"],
    "axes.labelcolor": OA_COLORS["slate"],
    "xtick.color":     OA_COLORS["slate"],
    "ytick.color":     OA_COLORS["slate"],
    "axes.prop_cycle": cycler(color=OA_CYCLE),
}
matplotlib.rcParams.update(PAPER_STYLE)


# ----------------------------
# Paths
# ----------------------------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DATA_DIR = os.path.join(PROJECT_ROOT, "data", "analysis")
OUT_DIR = os.path.join(PROJECT_ROOT, "figures", "organ_age")

NORMATIVE_PATH = os.path.join(DATA_DIR, "organ_age_normative.parquet")
CALIBRATED_PATH = os.path.join(DATA_DIR, "organ_age_calibrated.parquet")

V4_PANEL_PATHS = [
    ("v4_panel_aggressive",  os.path.join(DATA_DIR, "v4_panel_aggressive.parquet")),
    ("v4_panel_balanced",    os.path.join(DATA_DIR, "v4_panel_balanced.parquet")),
    ("v4_panel_conservative",os.path.join(DATA_DIR, "v4_panel_conservative.parquet")),
]


# ----------------------------
# Utils
# ----------------------------
def ensure_dir(path: str) -> None:
    """Create *path* and any missing intermediate directories.

    Parameters
    ----------
    path : str
        Directory to create.  Silently succeeds if it already exists.
    """
    os.makedirs(path, exist_ok=True)


def log(msg: str) -> None:
    """Print *msg* to stdout with immediate flushing.

    Parameters
    ----------
    msg : str
        Message string to print.
    """
    print(msg, flush=True)


def pick_col(df: pd.DataFrame, candidates) -> str | None:
    """Return the first column name from *candidates* that exists in *df*.

    Used for flexible column resolution when parquet files may use different
    naming conventions across dataset versions (e.g. ``"age_chrono"`` vs
    ``"chronological_age"``).

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame whose columns are searched.
    candidates : sequence of str
        Column names to try, in priority order.

    Returns
    -------
    str or None
        The first matching column name, or ``None`` if none are present.
    """
    for c in candidates:
        if c in df.columns:
            return c
    return None


def safe_numeric(series: pd.Series) -> pd.Series:
    """Coerce *series* to numeric, replacing non-parseable values with NaN.

    Parameters
    ----------
    series : pandas.Series
        Input series (may contain strings, mixed types, etc.).

    Returns
    -------
    pandas.Series
        Numeric series; invalid entries become ``float('nan')``.
    """
    return pd.to_numeric(series, errors="coerce")


def clean_xy(df: pd.DataFrame, xcol: str, ycol: str) -> pd.DataFrame:
    """Extract two columns from *df*, coerce both to numeric, and drop NaN rows.

    Parameters
    ----------
    df : pandas.DataFrame
        Source DataFrame.
    xcol : str
        Name of the x-axis column.
    ycol : str
        Name of the y-axis column.

    Returns
    -------
    pandas.DataFrame
        Two-column DataFrame with all rows that have finite values in both
        *xcol* and *ycol*.
    """
    out = df[[xcol, ycol]].copy()
    out[xcol] = safe_numeric(out[xcol])
    out[ycol] = safe_numeric(out[ycol])
    return out.dropna()


def sample_rows(df: pd.DataFrame, max_points: int = 3500, seed: int = 7) -> pd.DataFrame:
    """Return at most *max_points* rows from *df*, sampled reproducibly.

    When ``len(df) <= max_points`` the full DataFrame is returned unchanged
    so no data is discarded unnecessarily.  Downsampling is applied only to
    keep scatter plots legible and rendering time reasonable.

    Parameters
    ----------
    df : pandas.DataFrame
        Source DataFrame.
    max_points : int, optional
        Maximum number of rows to retain.  Defaults to 3500.
    seed : int, optional
        Random seed for reproducible sampling.  Defaults to 7.

    Returns
    -------
    pandas.DataFrame
        Original or down-sampled DataFrame.
    """
    if len(df) <= max_points:
        return df
    return df.sample(n=max_points, random_state=seed)


def tag_title(tag: str) -> str:
    """Convert a file-system-safe dataset *tag* into a human-readable title string.

    ``v4_panel_aggressive`` → ``"V4 Panel: Aggressive"``
    ``calibrated``          → ``"Calibrated"``

    Parameters
    ----------
    tag : str
        Short identifier string used as a filename prefix.

    Returns
    -------
    str
        Pretty-printed title, or an empty string when *tag* is falsy.
    """
    if not tag:
        return ""
    if tag.startswith("v4_panel_"):
        # Strip the common prefix then title-case the remainder.
        return "V4 Panel: " + tag.replace("v4_panel_", "").replace("_", " ").title()
    return tag.replace("_", " ").title()


def _grid(ax) -> None:
    """Apply a subtle, print-friendly background grid to *ax*.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes to modify in-place.
    """
    ax.grid(True, linewidth=0.6, alpha=0.30, color=OA_COLORS["grid"])
    ax.set_axisbelow(True)


def add_identity_line(ax, x, y):
    """Draw the y = x identity line over the combined range of *x* and *y*.

    The function is a no-op when either array is empty or when the computed
    data range is degenerate (non-finite or zero-width).

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Target axes.
    x : array-like
        Values defining one bound of the line's extent.
    y : array-like
        Values defining the other bound of the line's extent.
    """
    if len(x) == 0 or len(y) == 0:
        return
    lo = float(min(np.min(x), np.min(y)))
    hi = float(max(np.max(x), np.max(y)))
    # Skip degenerate ranges to avoid matplotlib warnings.
    if not np.isfinite(lo) or not np.isfinite(hi) or lo == hi:
        return
    ax.plot([lo, hi], [lo, hi], linestyle="--", linewidth=1.4,
            color=OA_COLORS["slate2"], label="y = x")


def savefig(fig, outpath: str, dpi: int = 320) -> None:
    """Save *fig* to *outpath* with tight bounding and then close it.

    Using ``bbox_inches="tight"`` ensures that long suptitles are not clipped
    at the edge of the exported PNG.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        Figure to save.
    outpath : str
        Destination file path.  Extension determines the format.
    dpi : int, optional
        Output resolution.  Defaults to 320 for print-quality rasters.
    """
    # Keep tight bounding so long titles are not clipped in exported PNGs.
    fig.savefig(outpath, dpi=dpi, bbox_inches="tight", pad_inches=0.08,
                facecolor=OA_COLORS["bg"])
    plt.close(fig)


# ----------------------------
# Shared axis-level helpers
# ----------------------------
def _draw_pred_vs_chrono(ax, x, y, color, label=None):
    """Scatter predicted vs. chronological age onto *ax* with identity line.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Target axes.
    x : array-like
        Chronological ages (x-axis values).
    y : array-like
        Model-predicted biological ages (y-axis values).
    color : str
        Matplotlib colour specification for the scatter points.
    label : str or None, optional
        Legend entry for these points.  Defaults to ``None``.
    """
    ax.scatter(x, y, s=14, alpha=0.55, color=color, linewidths=0, label=label)
    add_identity_line(ax, x, y)
    ax.set_xlabel("Chronological Age (Years)")
    ax.set_ylabel("Predicted Bio. Age (Years)")
    _grid(ax)


def _draw_gap_hist(ax, g, color, bins=55):
    """Draw a histogram of age-gap values onto *ax* with a zero-gap reference line.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Target axes.
    g : array-like
        Age-gap values (predicted − chronological).
    color : str
        Fill colour for the histogram bars.
    bins : int, optional
        Number of histogram bins.  Defaults to 55.
    """
    ax.hist(g, bins=bins, color=color, alpha=0.85, edgecolor="none")
    # Dashed vertical line at zero separates accelerated from decelerated ageing.
    ax.axvline(0.0, linewidth=1.5, color=OA_COLORS["slate"], linestyle="--")
    ax.set_xlabel("Age Gap (Pred - Chrono, Years)")
    ax.set_ylabel("Count")
    _grid(ax)


def _draw_gap_vs_chrono(ax, x, y, color, label=None):
    """Scatter age-gap against chronological age onto *ax* with a zero-gap baseline.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Target axes.
    x : array-like
        Chronological ages (x-axis values).
    y : array-like
        Age-gap values (y-axis values).
    color : str
        Matplotlib colour specification for the scatter points.
    label : str or None, optional
        Legend entry for these points.  Defaults to ``None``.
    """
    ax.scatter(x, y, s=14, alpha=0.55, color=color, linewidths=0, label=label)
    # Horizontal dashed reference at Δ = 0 (no acceleration).
    ax.axhline(0.0, linewidth=1.4, color=OA_COLORS["slate2"], linestyle="--")
    ax.set_xlabel("Chronological Age (Years)")
    ax.set_ylabel("Age Gap (Pred - Chrono, Years)")
    _grid(ax)


# ----------------------------
# Individual plot functions (single-organ / population)
# ----------------------------
def plot_pred_vs_chrono(df: pd.DataFrame, chrono_col: str, pred_col: str,
                        title: str, outpath: str) -> None:
    """Render a single scatter of predicted biological age vs. chronological age.

    Parameters
    ----------
    df : pandas.DataFrame
        Source data containing at least *chrono_col* and *pred_col*.
    chrono_col : str
        Column name for the chronological age values.
    pred_col : str
        Column name for the predicted biological age values.
    title : str
        Axes title for the figure.
    outpath : str
        Destination PNG path.
    """
    d = clean_xy(df, chrono_col, pred_col)
    x, y = d[chrono_col].to_numpy(), d[pred_col].to_numpy()

    fig, ax = plt.subplots(figsize=(6.6, 5.2), constrained_layout=True)
    _draw_pred_vs_chrono(ax, x, y, OA_COLORS["blue"])
    ax.set_title(title)
    savefig(fig, outpath)


def plot_gap_hist(df: pd.DataFrame, gap_col: str, title: str,
                  outpath: str, bins: int = 55) -> None:
    """Render a histogram of age-gap values for a single organ or population.

    The function is a no-op when *gap_col* contains no finite values, so
    callers do not need to pre-filter empty slices.

    Parameters
    ----------
    df : pandas.DataFrame
        Source data containing *gap_col*.
    gap_col : str
        Column name for the age-gap (predicted − chronological) values.
    title : str
        Axes title.
    outpath : str
        Destination PNG path.
    bins : int, optional
        Number of histogram bins.  Defaults to 55.
    """
    g = safe_numeric(df[gap_col]).dropna().to_numpy()
    if len(g) == 0:
        return

    fig, ax = plt.subplots(figsize=(6.6, 4.8), constrained_layout=True)
    _draw_gap_hist(ax, g, OA_COLORS["sage"], bins=bins)
    ax.set_title(title)
    savefig(fig, outpath)


def plot_gap_vs_chrono(df: pd.DataFrame, chrono_col: str, gap_col: str,
                       title: str, outpath: str) -> None:
    """Render a scatter of age-gap vs. chronological age for one organ or population.

    Parameters
    ----------
    df : pandas.DataFrame
        Source data containing *chrono_col* and *gap_col*.
    chrono_col : str
        Column name for chronological age values.
    gap_col : str
        Column name for age-gap (predicted − chronological) values.
    title : str
        Axes title.
    outpath : str
        Destination PNG path.
    """
    d = clean_xy(df, chrono_col, gap_col)
    x, y = d[chrono_col].to_numpy(), d[gap_col].to_numpy()

    fig, ax = plt.subplots(figsize=(6.6, 5.2), constrained_layout=True)
    _draw_gap_vs_chrono(ax, x, y, OA_COLORS["amber"])
    ax.set_title(title)
    savefig(fig, outpath)


def plot_hist_generic(df: pd.DataFrame, col: str, title: str, xlabel: str,
                      outpath: str, bins: int = 50) -> None:
    """Render a simple count histogram for any numeric column in *df*.

    Used for single-distribution views of V4 panel metrics such as burden,
    z-score, and abnormal-organ count.  The function is a no-op when the
    column contains no finite values.

    Parameters
    ----------
    df : pandas.DataFrame
        Source data containing *col*.
    col : str
        Column name to histogram.
    title : str
        Axes title.
    xlabel : str
        X-axis label string.
    outpath : str
        Destination PNG path.
    bins : int, optional
        Number of histogram bins.  Defaults to 50.
    """
    v = safe_numeric(df[col]).dropna().to_numpy()
    if len(v) == 0:
        return

    fig, ax = plt.subplots(figsize=(6.6, 4.8), constrained_layout=True)
    ax.hist(v, bins=bins, color=OA_COLORS["teal"], alpha=0.85, edgecolor="none")
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Count")
    ax.set_title(title)
    _grid(ax)
    savefig(fig, outpath)


def plot_scatter_generic(df: pd.DataFrame, xcol: str, ycol: str, title: str,
                         xlabel: str, ylabel: str, outpath: str) -> None:
    """Render a simple two-variable scatter plot for any pair of numeric columns.

    Used for relationships such as burden vs. number of abnormal organs that
    do not require the specialised reference lines of other scatter helpers.

    Parameters
    ----------
    df : pandas.DataFrame
        Source data containing *xcol* and *ycol*.
    xcol : str
        Column name for the x-axis variable.
    ycol : str
        Column name for the y-axis variable.
    title : str
        Axes title.
    xlabel : str
        X-axis label string.
    ylabel : str
        Y-axis label string.
    outpath : str
        Destination PNG path.
    """
    d = clean_xy(df, xcol, ycol)
    x, y = d[xcol].to_numpy(), d[ycol].to_numpy()

    fig, ax = plt.subplots(figsize=(6.6, 5.2), constrained_layout=True)
    ax.scatter(x, y, s=14, alpha=0.55, color=OA_COLORS["rose"], linewidths=0)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    _grid(ax)
    savefig(fig, outpath)


def plot_ci_scatter(df: pd.DataFrame, chrono_col: str, pred_col: str,
                    ci_low_col: str, ci_hi_col: str, title: str, outpath: str,
                    max_points: int = 2500) -> None:
    """Scatter predicted vs. chronological age with per-sample 95% confidence intervals.

    Confidence interval arms are clipped to zero on both sides so that
    asymmetric or poorly ordered CIs do not produce negative error bar lengths.
    The dataset is down-sampled to *max_points* before rendering to keep the
    figure legible.

    Parameters
    ----------
    df : pandas.DataFrame
        Source data containing *chrono_col*, *pred_col*, *ci_low_col*, and
        *ci_hi_col*.
    chrono_col : str
        Column name for chronological age.
    pred_col : str
        Column name for the predicted biological age (centre of CI).
    ci_low_col : str
        Column name for the lower confidence-interval bound.
    ci_hi_col : str
        Column name for the upper confidence-interval bound.
    title : str
        Axes title.
    outpath : str
        Destination PNG path.
    max_points : int, optional
        Maximum number of rows to render (down-sampled when exceeded).
        Defaults to 2500.
    """
    cols = [chrono_col, pred_col, ci_low_col, ci_hi_col]
    d = df[cols].copy()
    for c in cols:
        d[c] = safe_numeric(d[c])
    d = d.dropna()
    if len(d) == 0:
        return
    if len(d) > max_points:
        d = d.sample(n=max_points, random_state=7)

    x  = d[chrono_col].to_numpy()
    y  = d[pred_col].to_numpy()
    lo = d[ci_low_col].to_numpy()
    hi = d[ci_hi_col].to_numpy()

    fig, ax = plt.subplots(figsize=(6.9, 5.4), constrained_layout=True)
    # Clip negative arm lengths that arise when the CI bound crosses the point.
    yerr = np.vstack([np.clip(y - lo, 0, np.inf), np.clip(hi - y, 0, np.inf)])
    ax.errorbar(x, y, yerr=yerr, fmt="o", markersize=3.2,
                alpha=0.45, linewidth=0.8, capsize=0, color=OA_COLORS["teal"])
    add_identity_line(ax, x, y)
    ax.set_xlabel("Chronological Age (Years)")
    ax.set_ylabel("Predicted Biological Age (Years)")
    ax.set_title(title)
    _grid(ax)
    savefig(fig, outpath)


# ----------------------------
# PANEL generators (overlay / 2x2 grid)
# ----------------------------
def plot_pred_vs_chrono_panel(
    df: pd.DataFrame,
    organ_col: str,
    chrono_col: str,
    pred_col: str,
    organs: list[str],
    suptitle: str,
    outpath: str,
) -> None:
    """2x2 panel of scatter plots, one subplot per organ."""
    n = len(organs)
    ncols = 2
    nrows = (n + 1) // 2
    fig, axes = plt.subplots(nrows, ncols, figsize=(13.8, 5.6 * nrows))
    fig.subplots_adjust(top=0.90, hspace=0.32, wspace=0.22)
    axes = np.array(axes).flatten()

    for i, organ in enumerate(organs):
        ax = axes[i]
        dfo = clean_xy(df[df[organ_col] == organ], chrono_col, pred_col)
        d = sample_rows(dfo, max_points=2200, seed=17 + i)
        x, y = d[chrono_col].to_numpy(), d[pred_col].to_numpy()
        color = ORGAN_COLORS[i % len(ORGAN_COLORS)]
        _draw_pred_vs_chrono(ax, x, y, color)
        ax.set_title(organ.replace("_", " ").title(), fontsize=17)

    # hide unused axes
    for j in range(n, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(suptitle, fontsize=18, y=0.98)
    savefig(fig, outpath)


def plot_gap_vs_chrono_panel(
    df: pd.DataFrame,
    organ_col: str,
    chrono_col: str,
    gap_col: str,
    organs: list[str],
    suptitle: str,
    outpath: str,
) -> None:
    """2x2 panel of age-gap vs chronological age, one subplot per organ."""
    n = len(organs)
    ncols = 2
    nrows = (n + 1) // 2
    fig, axes = plt.subplots(nrows, ncols, figsize=(13.8, 5.6 * nrows))
    fig.subplots_adjust(top=0.90, hspace=0.32, wspace=0.22)
    axes = np.array(axes).flatten()

    for i, organ in enumerate(organs):
        ax = axes[i]
        dfo = clean_xy(df[df[organ_col] == organ], chrono_col, gap_col)
        d = sample_rows(dfo, max_points=2200, seed=31 + i)
        x, y = d[chrono_col].to_numpy(), d[gap_col].to_numpy()
        color = ORGAN_COLORS[i % len(ORGAN_COLORS)]
        _draw_gap_vs_chrono(ax, x, y, color)
        ax.set_title(organ.replace("_", " ").title(), fontsize=17)

    for j in range(n, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(suptitle, fontsize=18, y=0.98)
    savefig(fig, outpath)


def plot_gap_hist_panel(
    df: pd.DataFrame,
    organ_col: str,
    gap_col: str,
    organs: list[str],
    suptitle: str,
    outpath: str,
    bins: int = 55,
) -> None:
    """2x2 panel of age-gap histograms, one subplot per organ."""
    n = len(organs)
    ncols = 2
    nrows = (n + 1) // 2
    fig, axes = plt.subplots(nrows, ncols, figsize=(13.8, 5.0 * nrows))
    fig.subplots_adjust(top=0.90, hspace=0.32, wspace=0.22)
    axes = np.array(axes).flatten()

    for i, organ in enumerate(organs):
        ax = axes[i]
        dfo = df[df[organ_col] == organ]
        g = safe_numeric(dfo[gap_col]).dropna().to_numpy()
        color = ORGAN_COLORS[i % len(ORGAN_COLORS)]
        if len(g) == 0:
            ax.set_visible(False)
            continue
        _draw_gap_hist(ax, g, color, bins=bins)
        ax.set_title(organ.replace("_", " ").title(), fontsize=17)

    for j in range(n, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(suptitle, fontsize=18, y=0.98)
    savefig(fig, outpath)


def plot_gap_hist_overlay(
    df: pd.DataFrame,
    organ_col: str,
    gap_col: str,
    organs: list[str],
    title: str,
    outpath: str,
    bins: int = 60,
) -> None:
    """Single axes with all organ gap histograms overlaid and density-normalized."""
    fig, ax = plt.subplots(figsize=(8.8, 6.2), constrained_layout=True)

    for i, organ in enumerate(organs):
        dfo = df[df[organ_col] == organ]
        g = safe_numeric(dfo[gap_col]).dropna().to_numpy()
        if len(g) == 0:
            continue
        color = ORGAN_COLORS[i % len(ORGAN_COLORS)]
        label = organ.replace("_", " ").title()
        ax.hist(
            g,
            bins=bins,
            density=True,
            alpha=0.35,
            color=color,
            edgecolor="none",
            label=label,
        )

    ax.axvline(0.0, linewidth=1.6, color=OA_COLORS["slate"], linestyle="--",
               label="Zero gap")
    ax.set_xlabel("Age Gap (Predicted - Chronological, Years)")
    ax.set_ylabel("Density")
    ax.set_title(title)
    ax.legend(loc="upper right", fontsize=12, ncol=2)
    _grid(ax)
    savefig(fig, outpath)


def plot_pred_vs_chrono_overlay(
    df: pd.DataFrame,
    organ_col: str,
    chrono_col: str,
    pred_col: str,
    organs: list[str],
    title: str,
    outpath: str,
) -> None:
    """Overlay predicted-vs-chronological scatter for multiple organs on one axes.

    Each organ is plotted in a distinct colour from ``ORGAN_COLORS`` and with
    reduced alpha so overplotting remains readable.  A single shared identity
    line is drawn using the combined range of all organs' data.

    Parameters
    ----------
    df : pandas.DataFrame
        Full dataset containing *organ_col*, *chrono_col*, and *pred_col*.
    organ_col : str
        Column name that identifies which organ each row belongs to.
    chrono_col : str
        Column name for chronological age.
    pred_col : str
        Column name for predicted biological age.
    organs : list[str]
        Ordered list of organ names to overlay; each element is one series.
    title : str
        Axes title.
    outpath : str
        Destination PNG path.
    """
    fig, ax = plt.subplots(figsize=(8.8, 6.2), constrained_layout=True)

    for i, organ in enumerate(organs):
        dfo = clean_xy(df[df[organ_col] == organ], chrono_col, pred_col)
        # Down-sample dense organs to prevent the scatter from becoming opaque.
        dfo = sample_rows(dfo, max_points=2500, seed=101 + i)
        if len(dfo) == 0:
            continue
        color = ORGAN_COLORS[i % len(ORGAN_COLORS)]
        label = organ.replace("_", " ").title()
        ax.scatter(dfo[chrono_col], dfo[pred_col], s=12, alpha=0.28,
                   linewidths=0, color=color, label=label)

    # Draw identity line over the full combined range of all selected organs.
    all_xy = clean_xy(df[df[organ_col].isin(organs)], chrono_col, pred_col)
    if len(all_xy):
        add_identity_line(ax, all_xy[chrono_col].to_numpy(), all_xy[pred_col].to_numpy())
    ax.set_xlabel("Chronological Age (Years)")
    ax.set_ylabel("Predicted Bio. Age (Years)")
    ax.set_title(title)
    ax.legend(loc="upper left", ncol=2, fontsize=12)
    _grid(ax)
    savefig(fig, outpath)


def plot_gap_vs_chrono_overlay(
    df: pd.DataFrame,
    organ_col: str,
    chrono_col: str,
    gap_col: str,
    organs: list[str],
    title: str,
    outpath: str,
) -> None:
    """Overlay age-gap vs. chronological age scatter for multiple organs on one axes.

    Each organ is drawn in a distinct colour; a horizontal dashed line at
    Δ = 0 serves as a reference for zero age-gap.

    Parameters
    ----------
    df : pandas.DataFrame
        Full dataset containing *organ_col*, *chrono_col*, and *gap_col*.
    organ_col : str
        Column name that identifies which organ each row belongs to.
    chrono_col : str
        Column name for chronological age.
    gap_col : str
        Column name for the age-gap (predicted − chronological) values.
    organs : list[str]
        Ordered list of organ names to overlay.
    title : str
        Axes title.
    outpath : str
        Destination PNG path.
    """
    fig, ax = plt.subplots(figsize=(8.8, 6.2), constrained_layout=True)

    for i, organ in enumerate(organs):
        dfo = clean_xy(df[df[organ_col] == organ], chrono_col, gap_col)
        # Down-sample to prevent over-plotting in dense organs.
        dfo = sample_rows(dfo, max_points=2500, seed=131 + i)
        if len(dfo) == 0:
            continue
        color = ORGAN_COLORS[i % len(ORGAN_COLORS)]
        label = organ.replace("_", " ").title()
        ax.scatter(dfo[chrono_col], dfo[gap_col], s=12, alpha=0.28,
                   linewidths=0, color=color, label=label)

    ax.axhline(0.0, linewidth=1.5, color=OA_COLORS["slate"], linestyle="--", label="Zero gap")
    ax.set_xlabel("Chronological Age (Years)")
    ax.set_ylabel("Age Gap (Pred - Chrono, Years)")
    ax.set_title(title)
    ax.legend(loc="upper right", ncol=2, fontsize=12)
    _grid(ax)
    savefig(fig, outpath)


def plot_calibrated_overlay_summary(
    df: pd.DataFrame,
    organ_col: str,
    chrono_col: str,
    pred_col: str,
    gap_col: str,
    organs: list[str],
    outpath: str,
) -> None:
    """Compact calibrated summary with overlays.

    Uses a 2x2 layout with a bottom-row histogram panel to avoid the visual
    squashing that occurs in a very wide 1x3 strip.
    """
    fig = plt.figure(figsize=(13.5, 9.2))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 0.92], hspace=0.42, wspace=0.24)
    ax_pred = fig.add_subplot(gs[0, 0])
    ax_gap = fig.add_subplot(gs[0, 1])
    ax_hist = fig.add_subplot(gs[1, :])

    for i, organ in enumerate(organs):
        color = ORGAN_COLORS[i % len(ORGAN_COLORS)]
        label = organ.replace("_", " ").title()

        # Downsample dense organs so overlay comparisons remain legible.
        d_pred = clean_xy(df[df[organ_col] == organ], chrono_col, pred_col)
        d_pred = sample_rows(d_pred, max_points=2200, seed=170 + i)
        ax_pred.scatter(
            d_pred[chrono_col], d_pred[pred_col], s=11, alpha=0.26,
            linewidths=0, color=color, label=label
        )

        d_gap = clean_xy(df[df[organ_col] == organ], chrono_col, gap_col)
        d_gap = sample_rows(d_gap, max_points=2200, seed=190 + i)
        ax_gap.scatter(
            d_gap[chrono_col], d_gap[gap_col], s=11, alpha=0.26,
            linewidths=0, color=color, label=label
        )

        g = safe_numeric(df[df[organ_col] == organ][gap_col]).dropna().to_numpy()
        if len(g):
            ax_hist.hist(
                g, bins=55, density=True, alpha=0.30, color=color,
                edgecolor="none", label=label
            )

    all_xy = clean_xy(df[df[organ_col].isin(organs)], chrono_col, pred_col)
    if len(all_xy):
        add_identity_line(ax_pred, all_xy[chrono_col].to_numpy(), all_xy[pred_col].to_numpy())
    ax_pred.set_title("Predicted vs Chronological")
    ax_pred.set_xlabel("Chronological Age (Years)")
    ax_pred.set_ylabel("Predicted Bio. Age (Years)")
    _grid(ax_pred)

    ax_gap.axhline(0.0, linewidth=1.5, color=OA_COLORS["slate"], linestyle="--")
    ax_gap.set_title("Age Gap vs Chronological")
    ax_gap.set_xlabel("Chronological Age (Years)")
    ax_gap.set_ylabel("Age Gap (Pred - Chrono, Years)")
    _grid(ax_gap)

    ax_hist.axvline(0.0, linewidth=1.6, color=OA_COLORS["slate"], linestyle="--")
    ax_hist.set_title("Age Gap Distribution", pad=10)
    ax_hist.set_xlabel("Age Gap (Pred - Chrono, Years)")
    ax_hist.set_ylabel("Density")
    _grid(ax_hist)

    handles, labels = ax_hist.get_legend_handles_labels()
    if handles:
        fig.legend(
            handles,
            labels,
            loc="upper center",
            ncol=min(len(labels), 4),
            frameon=False,
            bbox_to_anchor=(0.5, 0.975),
            fontsize=12,
        )
    fig.suptitle("Calibrated Organ-Age Overlays - Key Organs", fontsize=19, y=0.985)
    savefig(fig, outpath)


def plot_v4_summary_panel(
    df: pd.DataFrame,
    tag: str,
    chrono_col: str | None,
    pred_col: str | None,
    gap_col: str | None,
    z_col: str | None,
    burden_col: str | None,
    n_abn_col: str | None,
    outpath: str,
) -> None:
    """2x2 combined V4 panel: bio-age vs chrono, delta hist, z-score dist, burden scatter."""
    fig, axes = plt.subplots(2, 2, figsize=(13, 10), constrained_layout=True)
    pretty = tag_title(tag)

    # Top-left: bio-age vs chrono
    ax = axes[0, 0]
    if chrono_col and pred_col:
        d = clean_xy(df, chrono_col, pred_col)
        x, y = d[chrono_col].to_numpy(), d[pred_col].to_numpy()
        _draw_pred_vs_chrono(ax, x, y, OA_COLORS["blue"])
        ax.set_title("Bio-Age vs Chronological Age")
    else:
        ax.set_visible(False)

    # Top-right: delta histogram
    ax = axes[0, 1]
    if gap_col:
        g = safe_numeric(df[gap_col]).dropna().to_numpy()
        if len(g):
            _draw_gap_hist(ax, g, OA_COLORS["sage"])
            ax.set_title("Age-Gap Distribution")
    else:
        ax.set_visible(False)

    # Bottom-left: Z-score distribution
    ax = axes[1, 0]
    if z_col:
        v = safe_numeric(df[z_col]).dropna().to_numpy()
        if len(v):
            ax.hist(v, bins=55, color=OA_COLORS["amber"], alpha=0.85, edgecolor="none")
            ax.axvline(0.0, linewidth=1.4, color=OA_COLORS["slate"], linestyle="--")
            ax.set_xlabel("Z-Score (Weighted)")
            ax.set_ylabel("Count")
            ax.set_title("Organ Z-Score Distribution")
            _grid(ax)
    else:
        ax.set_visible(False)

    # Bottom-right: burden vs n_abnormal or delta vs chrono fallback
    ax = axes[1, 1]
    if burden_col and n_abn_col:
        d = clean_xy(df, n_abn_col, burden_col)
        x, y = d[n_abn_col].to_numpy(), d[burden_col].to_numpy()
        ax.scatter(x, y, s=14, alpha=0.55, color=OA_COLORS["rose"], linewidths=0)
        ax.set_xlabel("Number of Abnormal Organs")
        ax.set_ylabel("Burden (Absolute Z)")
        ax.set_title("Organ Burden vs Abnormal Count")
        _grid(ax)
    elif chrono_col and gap_col:
        d = clean_xy(df, chrono_col, gap_col)
        x, y = d[chrono_col].to_numpy(), d[gap_col].to_numpy()
        _draw_gap_vs_chrono(ax, x, y, OA_COLORS["amber"])
        ax.set_title("Age-Gap vs Chronological Age")
    else:
        ax.set_visible(False)

    fig.suptitle(f"{pretty} - Summary Panel", fontsize=20)
    savefig(fig, outpath)


# ----------------------------
# Dataset-specific figure sets
# ----------------------------
def figures_for_normative(df: pd.DataFrame, tag: str, out_subdir: str) -> None:
    """Generate the full normative-model figure set from a predictions DataFrame.

    Produces population-level scatter, gap histogram, and gap-vs-chrono plots,
    then per-organ figures for the top 12 organs by sample count, and finally
    per-modality/source breakdowns when those grouping columns are present.

    Column names are resolved flexibly via :func:`pick_col` to accommodate
    different parquet schema versions.  If no pre-computed gap column exists
    it is derived as ``predicted − chronological`` on-the-fly.

    Parameters
    ----------
    df : pandas.DataFrame
        Normative model predictions parquet loaded into a DataFrame.
    tag : str
        Short identifier string used as a prefix for all output file names
        (e.g. ``"normative"``).
    out_subdir : str
        Directory where all PNG files for this dataset are written.
    """
    ensure_dir(out_subdir)

    chrono_col  = pick_col(df, ["age_chrono", "chrono_age", "chronological_age", "age"])
    pred_col    = pick_col(df, ["age_pred", "pred_age", "predicted_age", "y_pred"])
    gap_col     = pick_col(df, ["organ_age_delta", "age_gap", "delta", "age_delta"])
    organ_col   = pick_col(df, ["organ", "tissue"])
    modality_col = pick_col(df, ["modality"])
    source_col  = pick_col(df, ["source", "dataset"])

    if gap_col is None and chrono_col and pred_col:
        df = df.copy()
        df["_age_gap_tmp"] = safe_numeric(df[pred_col]) - safe_numeric(df[chrono_col])
        gap_col = "_age_gap_tmp"

    pretty = tag_title(tag)

    # Population-level figures
    if chrono_col and pred_col:
        plot_pred_vs_chrono(
            df, chrono_col, pred_col,
            title=f"{pretty}: Predicted vs Chronological\n(All Organs)",
            outpath=os.path.join(out_subdir, f"{tag}_pred_vs_chrono_all.png"),
        )

    if gap_col:
        plot_gap_hist(
            df, gap_col,
            title=f"{pretty}: Age-Gap Distribution\n(All Organs)",
            outpath=os.path.join(out_subdir, f"{tag}_gap_hist_all.png"),
        )

    if chrono_col and gap_col:
        plot_gap_vs_chrono(
            df, chrono_col, gap_col,
            title=f"{pretty}: Age Gap vs Chronological\n(All Organs)",
            outpath=os.path.join(out_subdir, f"{tag}_gap_vs_chrono_all.png"),
        )

    # Per-organ (individual + panels)
    if organ_col and chrono_col and pred_col:
        counts = df[organ_col].value_counts()
        top_organs = list(counts.head(12).index)

        # Individual files (kept for completeness)
        for organ in top_organs:
            dfo = df[df[organ_col] == organ]
            safe_name = str(organ).replace(" ", "_").replace("/", "_")
            plot_pred_vs_chrono(
                dfo, chrono_col, pred_col,
                title=f"{pretty}: Predicted vs Chronological - {organ}",
                outpath=os.path.join(out_subdir, f"{tag}_pred_vs_chrono_{safe_name}.png"),
            )
            if gap_col:
                plot_gap_hist(
                    dfo, gap_col,
                    title=f"{pretty}: Age-Gap Distribution - {organ}",
                    outpath=os.path.join(out_subdir, f"{tag}_gap_hist_{safe_name}.png"),
                )
                plot_gap_vs_chrono(
                    dfo, chrono_col, gap_col,
                    title=f"{pretty}: Age Gap vs Chronological - {organ}",
                    outpath=os.path.join(out_subdir, f"{tag}_gap_vs_chrono_{safe_name}.png"),
                )

    # Modality-specific figures
    for col, label in [(modality_col, "Modality"), (source_col, "Source")]:
        if col and col in df.columns:
            cats = list(df[col].dropna().unique())
            if 1 < len(cats) <= 6 and chrono_col and pred_col:
                for c in cats:
                    dfc = df[df[col] == c]
                    safe_c = str(c).replace(" ", "_").replace("/", "_")
                    plot_pred_vs_chrono(
                        dfc, chrono_col, pred_col,
                        title=f"{pretty}: Predicted vs Chronological ({label} = {c})",
                        outpath=os.path.join(
                            out_subdir, f"{tag}_pred_vs_chrono_{label.lower()}_{safe_c}.png"),
                    )


def figures_for_calibrated(df: pd.DataFrame, tag: str, out_subdir: str) -> None:
    """Generate the full calibrated-model figure set from a predictions DataFrame.

    In addition to the basic scatter/histogram/gap plots produced by
    :func:`figures_for_normative`, this function also renders:

    * A CI scatter plot (when lower and upper CI columns are present).
    * Per-organ 2x2 panel PNGs for a curated set of key organs
      (brain, heart, lung, liver).
    * Overlay comparisons (all key organs on one axes) for both scatter
      and gap views.
    * A calibrated overlay summary panel (2x2 grid with a wide bottom row).
    * An overlaid density histogram of the age-gap for up to 8 organs.

    Parameters
    ----------
    df : pandas.DataFrame
        Calibrated model predictions parquet loaded into a DataFrame.
    tag : str
        Short identifier string used as a prefix for all output file names
        (e.g. ``"calibrated"``).
    out_subdir : str
        Directory where all PNG files for this dataset are written.
    """
    ensure_dir(out_subdir)

    chrono_col = pick_col(df, ["age_chrono", "chrono_age", "chronological_age", "age"])
    pred_col   = pick_col(df, ["age_pred_cal", "age_pred", "pred_age", "predicted_age", "y_pred"])
    gap_col    = pick_col(df, ["age_delta_cal", "organ_age_delta", "age_gap", "delta", "age_delta"])
    ci_low_col = pick_col(df, ["ci_lower", "lower_ci", "ci_low"])
    ci_hi_col  = pick_col(df, ["ci_upper", "upper_ci", "ci_high"])
    organ_col  = pick_col(df, ["organ", "tissue"])

    if gap_col is None and chrono_col and pred_col:
        df = df.copy()
        df["_age_gap_tmp"] = safe_numeric(df[pred_col]) - safe_numeric(df[chrono_col])
        gap_col = "_age_gap_tmp"

    pretty = tag_title(tag)

    # Population-level
    if chrono_col and pred_col:
        plot_pred_vs_chrono(
            df, chrono_col, pred_col,
            title=f"{pretty}: Predicted vs Chronological\n(All Organs)",
            outpath=os.path.join(out_subdir, f"{tag}_pred_vs_chrono_all.png"),
        )

    if gap_col:
        plot_gap_hist(
            df, gap_col,
            title=f"{pretty}: Age-Gap Distribution\n(All Organs)",
            outpath=os.path.join(out_subdir, f"{tag}_gap_hist_all.png"),
        )

    if chrono_col and gap_col:
        plot_gap_vs_chrono(
            df, chrono_col, gap_col,
            title=f"{pretty}: Age Gap vs Chronological\n(All Organs)",
            outpath=os.path.join(out_subdir, f"{tag}_gap_vs_chrono_all.png"),
        )

    if chrono_col and pred_col and ci_low_col and ci_hi_col:
        plot_ci_scatter(
            df, chrono_col, pred_col, ci_low_col, ci_hi_col,
            title=f"{pretty}: Predicted vs Chronological\nwith 95% CI (Sampled)",
            outpath=os.path.join(out_subdir, f"{tag}_pred_vs_chrono_with_ci.png"),
        )

    # Per-organ panels + overlays
    if organ_col and chrono_col and pred_col:
        counts = df[organ_col].value_counts()
        top_organs = list(counts.head(12).index)

        # --- Paper-ready 2x2 panel PNGs ---
        panel_organs = [o for o in ["brain", "brain_cortex", "heart", "lung", "liver"] if o in top_organs][:4]
        if len(panel_organs) >= 2:
            plot_pred_vs_chrono_panel(
                df, organ_col, chrono_col, pred_col,
                organs=panel_organs,
                suptitle=f"{pretty}: Predicted vs Chronological - Key Organs",
                outpath=os.path.join(out_subdir, f"{tag}_pred_vs_chrono_4organ_panel.png"),
            )
            plot_pred_vs_chrono_overlay(
                df, organ_col, chrono_col, pred_col,
                organs=panel_organs,
                title=f"{pretty}: Predicted vs Chronological Overlay",
                outpath=os.path.join(out_subdir, f"{tag}_pred_vs_chrono_overlay.png"),
            )
            if gap_col:
                plot_gap_vs_chrono_panel(
                    df, organ_col, chrono_col, gap_col,
                    organs=panel_organs,
                    suptitle=f"{pretty}: Age-Gap vs Chronological - Key Organs",
                    outpath=os.path.join(out_subdir, f"{tag}_gap_vs_chrono_4organ_panel.png"),
                )
                plot_gap_vs_chrono_overlay(
                    df, organ_col, chrono_col, gap_col,
                    organs=panel_organs,
                    title=f"{pretty}: Age-Gap vs Chronological Overlay",
                    outpath=os.path.join(out_subdir, f"{tag}_gap_vs_chrono_overlay.png"),
                )
                plot_gap_hist_panel(
                    df, organ_col, gap_col,
                    organs=panel_organs,
                    suptitle=f"{pretty}: Age-Gap Distributions - Key Organs",
                    outpath=os.path.join(out_subdir, f"{tag}_gap_hist_4organ_panel.png"),
                )
                plot_calibrated_overlay_summary(
                    df, organ_col, chrono_col, pred_col, gap_col,
                    organs=panel_organs,
                    outpath=os.path.join(out_subdir, f"{tag}_overlay_summary_key_organs.png"),
                )

        # --- Overlay histogram (all top organs on one axes) ---
        if gap_col and len(top_organs) >= 2:
            overlay_organs = [o for o in top_organs if len(
                safe_numeric(df[df[organ_col] == o][gap_col]).dropna()) >= 30][:8]
            if overlay_organs:
                plot_gap_hist_overlay(
                    df, organ_col, gap_col,
                    organs=overlay_organs,
                    title=f"{pretty}: Age-Gap Overlay - All Organs",
                    outpath=os.path.join(out_subdir, f"{tag}_gap_hist_overlay.png"),
                )

        # Individual per-organ files (kept for completeness)
        for organ in top_organs:
            dfo = df[df[organ_col] == organ]
            safe_name = str(organ).replace(" ", "_").replace("/", "_")
            plot_pred_vs_chrono(
                dfo, chrono_col, pred_col,
                title=f"{pretty}: Predicted vs Chronological - {organ}",
                outpath=os.path.join(out_subdir, f"{tag}_pred_vs_chrono_{safe_name}.png"),
            )
            if gap_col:
                plot_gap_hist(
                    dfo, gap_col,
                    title=f"{pretty}: Age-Gap Distribution - {organ}",
                    outpath=os.path.join(out_subdir, f"{tag}_gap_hist_{safe_name}.png"),
                )
                plot_gap_vs_chrono(
                    dfo, chrono_col, gap_col,
                    title=f"{pretty}: Age Gap vs Chronological - {organ}",
                    outpath=os.path.join(out_subdir, f"{tag}_gap_vs_chrono_{safe_name}.png"),
                )


def figures_for_v4_panel(df: pd.DataFrame, tag: str, out_subdir: str) -> None:
    """Generate the full V4 panel figure set from a panel predictions DataFrame.

    The V4 panel data includes patient-level summary statistics (weighted
    bio-age, delta, z-score, burden, organ counts) rather than individual
    organ predictions.  This function produces:

    * A combined 2x2 summary panel PNG (the primary publication figure).
    * Individual histograms and scatter plots for each available metric.

    All column names are resolved flexibly via :func:`pick_col` to support
    the aggressive, balanced, and conservative panel variants.

    Parameters
    ----------
    df : pandas.DataFrame
        V4 panel parquet (one row per patient) loaded into a DataFrame.
    tag : str
        Short identifier string used as a prefix for output file names
        (e.g. ``"v4_panel_aggressive"``).
    out_subdir : str
        Directory where all PNG files for this panel are written.
    """
    ensure_dir(out_subdir)

    chrono_col  = pick_col(df, ["age_chrono", "chrono_age", "chronological_age", "age"])
    pred_col    = pick_col(df, ["bio_age_weighted", "bio_age", "age_pred", "pred_age", "predicted_age"])
    gap_col     = pick_col(df, ["delta_weighted", "age_delta", "age_gap", "delta"])
    z_col       = pick_col(df, ["z_weighted", "z", "organ_age_zscore", "zscore"])
    n_abn_col   = pick_col(df, ["n_abnormal_organs", "n_abnormal", "abnormal_organs"])
    burden_col  = pick_col(df, ["burden_abs_z", "burden", "abs_z_burden"])
    n_organs_col = pick_col(df, ["n_organs", "n_total_organs"])

    if gap_col is None and chrono_col and pred_col:
        df = df.copy()
        df["_age_gap_tmp"] = safe_numeric(df[pred_col]) - safe_numeric(df[chrono_col])
        gap_col = "_age_gap_tmp"

    pretty = tag_title(tag)

    # --- Combined summary panel (publication-ready single PNG) ---
    plot_v4_summary_panel(
        df, tag, chrono_col, pred_col, gap_col, z_col, burden_col, n_abn_col,
        outpath=os.path.join(out_subdir, f"{tag}_summary_panel.png"),
    )

    # Individual figures (kept for detailed inspection)
    if chrono_col and pred_col:
        plot_pred_vs_chrono(
            df, chrono_col, pred_col,
            title=f"{pretty}: Bio-Age vs Chronological",
            outpath=os.path.join(out_subdir, f"{tag}_bioage_vs_chrono.png"),
        )

    if gap_col:
        plot_gap_hist(
            df, gap_col,
            title=f"{pretty}: Delta Distribution",
            outpath=os.path.join(out_subdir, f"{tag}_delta_hist.png"),
        )
        if chrono_col:
            plot_gap_vs_chrono(
                df, chrono_col, gap_col,
                title=f"{pretty}: Delta vs Chronological",
                outpath=os.path.join(out_subdir, f"{tag}_delta_vs_chrono.png"),
            )

    if burden_col:
        plot_hist_generic(
            df, burden_col,
            title=f"{pretty}: Burden Distribution",
            xlabel="Burden (Absolute Z)",
            outpath=os.path.join(out_subdir, f"{tag}_burden_hist.png"),
        )

    if n_abn_col:
        plot_hist_generic(
            df, n_abn_col,
            title=f"{pretty}: Abnormal Organs (Count) Distribution",
            xlabel="Number of Abnormal Organs",
            outpath=os.path.join(out_subdir, f"{tag}_n_abnormal_hist.png"),
            bins=40,
        )

    if z_col:
        plot_hist_generic(
            df, z_col,
            title=f"{pretty}: Z-Score Distribution",
            xlabel="Z-Score (Weighted)",
            outpath=os.path.join(out_subdir, f"{tag}_z_hist.png"),
        )

    if burden_col and n_abn_col:
        plot_scatter_generic(
            df, n_abn_col, burden_col,
            title=f"{pretty}: Burden vs Abnormal Organs",
            xlabel="Number of Abnormal Organs",
            ylabel="Burden (Absolute Z)",
            outpath=os.path.join(out_subdir, f"{tag}_burden_vs_n_abnormal.png"),
        )

    if n_organs_col:
        plot_hist_generic(
            df, n_organs_col,
            title=f"{pretty}: Organs Available (Count) Distribution",
            xlabel="Number of Organs Available",
            outpath=os.path.join(out_subdir, f"{tag}_n_organs_hist.png"),
            bins=30,
        )


# ----------------------------
# Main
# ----------------------------
def main() -> None:
    """Entry point: load parquet files and dispatch to per-dataset figure generators.

    Iterates over the three expected input parquet files:

    1. ``organ_age_normative.parquet``  → :func:`figures_for_normative`
    2. ``organ_age_calibrated.parquet`` → :func:`figures_for_calibrated`
    3. ``v4_panel_{aggressive,balanced,conservative}.parquet``
       → :func:`figures_for_v4_panel` (one call per variant)

    Missing files are skipped with a ``[SKIP]`` log message rather than
    raising an error, so the script can be run on partial pipeline outputs.

    All figures are written under ``figures/organ_age/``.
    """
    ensure_dir(OUT_DIR)
    stamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log(f"[INFO] Organ-Age Figures v1 | {stamp}")
    log(f"[INFO] Project root: {PROJECT_ROOT}")
    log(f"[INFO] Output dir:  {OUT_DIR}")

    if os.path.exists(NORMATIVE_PATH):
        log(f"\n[LOAD] {NORMATIVE_PATH}")
        df_norm = pd.read_parquet(NORMATIVE_PATH)
        figures_for_normative(df_norm, tag="normative", out_subdir=OUT_DIR)
    else:
        log(f"\n[SKIP] Missing: {NORMATIVE_PATH}")

    if os.path.exists(CALIBRATED_PATH):
        log(f"\n[LOAD] {CALIBRATED_PATH}")
        df_cal = pd.read_parquet(CALIBRATED_PATH)
        figures_for_calibrated(df_cal, tag="calibrated", out_subdir=OUT_DIR)
    else:
        log(f"\n[SKIP] Missing: {CALIBRATED_PATH}")

    subdir = os.path.join(OUT_DIR, "v4_panels")
    ensure_dir(subdir)

    for tag, path in V4_PANEL_PATHS:
        if os.path.exists(path):
            log(f"\n[LOAD] {path}")
            df_panel = pd.read_parquet(path)
            figures_for_v4_panel(df_panel, tag=tag, out_subdir=subdir)
        else:
            log(f"\n[SKIP] Missing: {path}")

    log(f"\n[DONE] Figures written to: {OUT_DIR}")


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=UserWarning)
    main()
