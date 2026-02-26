"""
v4_plot_one_subject_panel.py
============================
Generate a single-subject organ age-delta bar chart from the calibrated
organ-age table.

For the requested subject the script:
  1. Loads the calibrated parquet and filters to the matching subject (with
     case-insensitive substring fallback for partial ID matches).
  2. Aggregates repeated measurements per organ by mean.
  3. Sorts organs by delta (most accelerated first).
  4. Renders a bar chart where:
       - Rose-coloured bars indicate organs with a positive delta
         (biologically older than expected).
       - Blue bars indicate organs with a negative delta (younger).
       - Each bar is annotated with the calibrated z-score.
  5. Saves the figure to ``--out`` (default:
     ``figures/vitalis_v4/1117F_panel.png``).

Typical usage::

    python v4_plot_one_subject_panel.py --subject_id GTEX-1117F \\
        --calibrated_path data/analysis/organ_age_calibrated.parquet
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt


OA_COLORS = {
    "slate": "#1D3557",
    "blue": "#0072B2",
    "teal": "#009E73",
    "amber": "#E69F00",
    "rose": "#D55E00",
    "grid": "#CFD8DC",
    "bg": "#FFFFFF",
}

PAPER_STYLE = {
    "font.family": "serif",
    "font.serif": ["DejaVu Serif", "Georgia", "Times New Roman", "serif"],
    "mathtext.fontset": "dejavuserif",
    "font.size": 16,
    "axes.titlesize": 18,
    "axes.labelsize": 16,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "axes.linewidth": 1.2,
    "axes.edgecolor": OA_COLORS["slate"],
    "text.color": OA_COLORS["slate"],
    "axes.labelcolor": OA_COLORS["slate"],
    "xtick.color": OA_COLORS["slate"],
    "ytick.color": OA_COLORS["slate"],
    "savefig.dpi": 320,
    "figure.dpi": 120,
}
matplotlib.rcParams.update(PAPER_STYLE)


def _grid(ax) -> None:
    """
    Apply a subtle vertical grid to ``ax`` for readability.

    Enables y-axis grid lines with low opacity and ensures bars are drawn
    on top of the grid.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Target axes object.
    """
    ax.grid(True, axis="y", linewidth=0.6, alpha=0.30, color=OA_COLORS["grid"])
    ax.set_axisbelow(True)


def main() -> None:
    """
    Entry point: parse CLI arguments, load calibrated data, find the subject,
    aggregate per organ, and render the annotated delta bar chart.
    """
    parser = argparse.ArgumentParser(description="Plot one-subject organ-age delta panel.")
    parser.add_argument("--calibrated_path", default="data/analysis/organ_age_calibrated.parquet")
    parser.add_argument("--subject_id", required=True)
    parser.add_argument("--out", default="figures/vitalis_v4/1117F_panel.png")
    args = parser.parse_args()

    df = pd.read_parquet(args.calibrated_path)
    df["subject_id"] = df["subject_id"].astype(str)

    sub = df[df["subject_id"].str.lower() == args.subject_id.lower()].copy()
    if sub.empty:
        sub = df[df["subject_id"].str.contains(args.subject_id, case=False, na=False)].copy()
    if sub.empty:
        raise SystemExit(f"[ERR] No rows found for subject_id={args.subject_id}")

    # Aggregate repeated rows so each organ is represented once.
    per = (
        sub.groupby("organ", dropna=False)
        .agg(
            age_chrono=("age_chrono", "mean"),
            age_delta_cal=("age_delta_cal", "mean"),
            zscore_cal=("zscore_cal", "mean"),
        )
        .reset_index()
        .sort_values("age_delta_cal", ascending=False)
    )

    age_chrono = float(per["age_chrono"].mean())
    x = np.arange(len(per))
    y = per["age_delta_cal"].to_numpy()
    z = per["zscore_cal"].to_numpy()

    # Blue indicates decelerated organs (negative delta), rose indicates accelerated.
    colors = [OA_COLORS["rose"] if d > 0 else OA_COLORS["blue"] for d in y]

    fig, ax = plt.subplots(figsize=(11.0, 6.8), constrained_layout=True)
    ax.bar(x, y, color=colors, alpha=0.90, edgecolor="none")
    ax.axhline(0.0, linestyle="--", linewidth=1.4, color=OA_COLORS["slate"])

    ax.set_xticks(x)
    ax.set_xticklabels(per["organ"].astype(str), rotation=32, ha="right")
    ax.set_ylabel("Age gap (Predicted - Chronological, years)")
    ax.set_title(f"Subject {sub['subject_id'].iloc[0]} | Chronological age ~ {age_chrono:.1f}")
    _grid(ax)

    for i, (delta, zi) in enumerate(zip(y, z)):
        if not np.isfinite(zi):
            continue
        offset = 0.22 if delta >= 0 else -0.22
        ax.text(
            i,
            delta + offset,
            f"z={zi:.2f}",
            ha="center",
            va="bottom" if delta >= 0 else "top",
            fontsize=11,
            color=OA_COLORS["slate"],
        )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=320, bbox_inches="tight", pad_inches=0.05, facecolor=OA_COLORS["bg"])
    plt.close(fig)
    print("[OK] Wrote:", out_path)


if __name__ == "__main__":
    main()
