"""
v4_visualize.py
===============
Standalone visualization suite for the v4 OrganAge pipeline.

Given a subject ID this script generates three publication-ready figures
saved under ``figures/vitalis_v4/<subject_id>/``:

  1. **radar.png** – polar radar chart of calibrated organ z-scores.
  2. **delta_bar.png** – bar chart of per-organ age deltas
     (organ age minus chronological age).
  3. **ci_plot.png** – horizontal error-bar plot showing calibrated organ
     ages with 90 % confidence intervals, annotated with the chronological
     age reference line.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# -----------------------------
# Load calibrated + panel data
# -----------------------------

def load_calibrated(path):
    """
    Load the calibrated organ-age parquet.

    Parameters
    ----------
    path : str
        Path to the calibrated ``.parquet`` file.

    Returns
    -------
    pd.DataFrame
        Calibrated organ-age table.
    """
    df = pd.read_parquet(path)
    return df


def load_panel(path):
    """
    Load a v4 panel parquet and index it by ``subject_id``.

    Parameters
    ----------
    path : str
        Path to the panel ``.parquet`` file.

    Returns
    -------
    pd.DataFrame
        Panel table indexed by ``subject_id``.
    """
    df = pd.read_parquet(path)
    return df.set_index("subject_id")


# -----------------------------
# Visualization Helpers
# -----------------------------

def ensure_dir(path: Path):
    """
    Create ``path`` (and any missing parents) if it does not already exist.

    Parameters
    ----------
    path : Path
        Directory to create.
    """
    path.mkdir(parents=True, exist_ok=True)


def radar_plot(organs, values, outpath):
    """
    values = z-scores (calibrated)
    """
    N = len(organs)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    values = values.tolist()
    values += values[:1]  # close loop
    angles += angles[:1]

    fig = plt.figure(figsize=(8, 8))
    ax = plt.subplot(111, polar=True)

    ax.plot(angles, values, linewidth=2)
    ax.fill(angles, values, alpha=0.25)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(organs, fontsize=10)
    ax.set_yticklabels([])

    ax.set_title("Organ Z-Scores (Calibrated)", fontsize=16, pad=20)
    plt.savefig(outpath, dpi=300, bbox_inches='tight')
    plt.close()


def delta_bar_plot(organs, deltas, outpath):
    """
    Render a bar chart of per-organ age deltas (organ age - chronological age).

    Bars are coloured blue for negative deltas (biologically younger) and red
    for positive deltas (biologically older).

    Parameters
    ----------
    organs : list[str]
        Ordered list of organ names for the x-axis labels.
    deltas : array-like of float
        Per-organ delta values (years) in the same order as ``organs``.
    outpath : Path
        Destination path for the saved PNG figure.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(organs, deltas, color=["#1f77b4" if d < 0 else "#d62728" for d in deltas])

    ax.axhline(0, color='black', linewidth=1)
    ax.set_ylabel("Δ (Organ Age – Chronological Age)", fontsize=12)
    ax.set_title("Organ Age Deltas", fontsize=16)
    plt.xticks(rotation=45, ha="right")

    plt.savefig(outpath, dpi=300, bbox_inches='tight')
    plt.close()


def ci_plot(chrono_age, organs, cal_ages, ci_low, ci_high, outpath):
    """
    Render a horizontal error-bar plot of calibrated organ ages with CIs.

    A vertical dashed reference line marks the subject's chronological age.

    Parameters
    ----------
    chrono_age : float
        Subject's chronological age in years.
    organs : list[str]
        Ordered list of organ names for the y-axis labels.
    cal_ages : array-like of float
        Calibrated (isotonic-corrected) predicted age per organ.
    ci_low : array-like of float
        Lower bound of the confidence interval per organ.
    ci_high : array-like of float
        Upper bound of the confidence interval per organ.
    outpath : Path
        Destination path for the saved PNG figure.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    y = np.arange(len(organs))

    ax.errorbar(
        cal_ages,
        y,
        xerr=[cal_ages - ci_low, ci_high - cal_ages],
        fmt="o",
        ecolor="gray",
        elinewidth=2,
        capsize=4,
        markersize=6,
        color="blue",
        label="Calibrated Organ Age ± 95% CI",
    )

    ax.axvline(chrono_age, color="red", linestyle="--", linewidth=2, label="Chronological Age")

    ax.set_yticks(y)
    ax.set_yticklabels(organs)
    ax.set_xlabel("Age")
    ax.set_title("Organ Ages with Confidence Intervals", fontsize=16)
    ax.legend()

    plt.tight_layout()
    plt.savefig(outpath, dpi=300)
    plt.close()


# -----------------------------
# Main Visualization Pipeline
# -----------------------------

def main():
    """
    Command-line entry point for the v4 visualization suite.

    Loads calibrated organ-age data and the subject panel, then generates
    the radar chart, delta bar chart, and confidence-interval plot for the
    specified subject.
    """
    parser = argparse.ArgumentParser(description="OrganAge-v4 Visualization Suite")
    parser.add_argument("--subject", type=str, required=True, help="Subject ID to visualize")
    parser.add_argument(
        "--calibrated",
        type=str,
        default="data/analysis/organ_age_calibrated.parquet"
    )
    parser.add_argument(
        "--panel",
        type=str,
        default="data/analysis/v4_panel_balanced.parquet"
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default="figures/vitalis_v4"
    )

    args = parser.parse_args()

    # Load data
    df_cal = load_calibrated(args.calibrated)
    df_panel = load_panel(args.panel)

    subj = args.subject
    if subj not in df_panel.index:
        raise ValueError(f"[V4 VIS] Subject '{subj}' not found in panel.")

    # Filter calibrated table to subject organs
    df_sub = df_cal[df_cal["subject_id"] == subj].copy()
    if df_sub.empty:
        raise RuntimeError(f"[V4 VIS] No calibrated entries for subject '{subj}'")

    outdir = Path(args.outdir) / subj
    ensure_dir(outdir)

    # Sort for consistency
    df_sub = df_sub.sort_values("organ")

    organs = df_sub["organ"].tolist()
    z = df_sub["zscore_cal"].to_numpy()
    deltas = df_sub["age_delta_cal"].to_numpy()
    cal_ages = df_sub["age_pred_cal"].to_numpy()
    ci_low = df_sub["ci_lower"].to_numpy()
    ci_high = df_sub["ci_upper"].to_numpy()

    chrono = float(df_sub["age_chrono"].mean())

    # 1. Radar chart
    radar_plot(
        organs,
        z,
        outdir / "radar.png"
    )

    # 2. Bar plot of deltas
    delta_bar_plot(
        organs,
        deltas,
        outdir / "delta_bar.png"
    )

    # 3. Confidence interval plot
    ci_plot(
        chrono,
        organs,
        cal_ages,
        ci_low,
        ci_high,
        outdir / "ci_plot.png"
    )

    print(f"[V4 VIS] Saved all plots to {outdir}")


if __name__ == "__main__":
    main()
