"""
v4_report_isef.py
=================
Generate a poster-ready, two-page ISEF-style PDF report for a single subject.

This report is optimised for science-fair and academic poster contexts. It
uses a clean white background with purple/orange accent colours and includes
a plain-English explanation panel alongside the quantitative results.

Page 1 – Poster-friendly summary:
  * Left column: quantitative global metrics (chronological age, biological
    age, delta, z-score, organ/abnormal counts, burden).
  * Right column: plain-language explanation of how to interpret the profile.
  * Per-organ table with a reduced column set for compact poster inclusion.

Page 2 – Visualization panel (same three plots as clinical/tech reports but
          with the ISEF colour scheme).

Output is written to ``reports/v4/isef/OrganAge_isef_<subject_id>.pdf``.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


CAL_PATH = "data/analysis/organ_age_calibrated.parquet"
PANEL_PATH = "data/analysis/v4_panel_balanced.parquet"


def load_data(cal_path: str, panel_path: str):
    """
    Load calibrated organ-age table and subject-level panel.

    Parameters
    ----------
    cal_path : str
        Path to the calibrated organ-age ``.parquet``.
    panel_path : str
        Path to the subject-level panel ``.parquet``; indexed by
        ``subject_id`` on return.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        ``(df_cal, df_panel)`` where ``df_panel`` is indexed by
        ``subject_id``.
    """
    df_cal = pd.read_parquet(cal_path)
    df_panel = pd.read_parquet(panel_path).set_index("subject_id")
    return df_cal, df_panel


def aggregate_organs(df_sub: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate multiple samples per organ for a single subject into one row
    per organ by taking the mean of numeric fields.

    Parameters
    ----------
    df_sub : pd.DataFrame
        Subset of the calibrated table for a single subject.

    Returns
    -------
    pd.DataFrame
        One row per organ, rounded to 2 decimal places and sorted
        alphabetically by organ name.
    """
    agg = (
        df_sub.groupby("organ", dropna=False)
        .agg(
            n_samples=("organ", "size"),
            age_chrono=("age_chrono", "mean"),
            age_pred_cal=("age_pred_cal", "mean"),
            age_delta_cal=("age_delta_cal", "mean"),
            zscore_cal=("zscore_cal", "mean"),
            ci_lower=("ci_lower", "mean"),
            ci_upper=("ci_upper", "mean"),
        )
        .reset_index()
        .sort_values("organ")
    )
    num_cols = [
        "age_chrono",
        "age_pred_cal",
        "age_delta_cal",
        "zscore_cal",
        "ci_lower",
        "ci_upper",
    ]
    agg[num_cols] = agg[num_cols].round(2)
    return agg


def radar_plot(ax, organs, zscores):
    """
    Draw an ISEF-styled polar radar chart of per-organ calibrated z-scores.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        A polar subplot.
    organs : list[str]
        Organ names for the spoke labels.
    zscores : np.ndarray
        Calibrated z-score per organ, aligned with ``organs``.
    """
    n = len(organs)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
    angles += angles[:1]
    values = zscores.tolist() + zscores[:1].tolist()

    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.plot(angles, values, linewidth=2, color="#7b2cff")
    ax.fill(angles, values, alpha=0.3, color="#c09bff")
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(organs, fontsize=9)
    ax.set_yticklabels([])
    ax.set_title("Organ Z-Scores (Calibrated)", fontsize=12, pad=15)


def delta_bar_plot(ax, organs, deltas):
    """
    Draw an ISEF-styled bar chart of per-organ age deltas.

    Negative deltas are coloured green (biologically younger); positive
    deltas are coloured red (biologically older).

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        A standard Cartesian subplot.
    organs : list[str]
        Organ names for the x-axis labels.
    deltas : array-like of float
        Age delta per organ (calibrated organ age minus chronological age).
    """
    colors = ["#2ca02c" if d < 0 else "#d62728" for d in deltas]
    ax.bar(organs, deltas, color=colors)
    ax.axhline(0, color="black", linewidth=1)
    ax.set_ylabel("Δ (Organ Age – Chronological Age)", fontsize=10)
    ax.set_xticklabels(organs, rotation=45, ha="right", fontsize=9)
    ax.set_title("Organ Age Deltas", fontsize=12)


def ci_plot(ax, chrono_age, organs, cal_ages, ci_low, ci_high):
    """
    Draw a horizontal error-bar plot of calibrated organ ages with CIs.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        A standard Cartesian subplot.
    chrono_age : float
        Subject's chronological age, drawn as an orange dashed reference.
    organs : list[str]
        Organ names for the y-axis labels.
    cal_ages : np.ndarray
        Calibrated organ age estimates.
    ci_low : np.ndarray
        Lower CI bound per organ.
    ci_high : np.ndarray
        Upper CI bound per organ.
    """
    y = np.arange(len(organs))
    ax.errorbar(
        cal_ages,
        y,
        xerr=[cal_ages - ci_low, ci_high - cal_ages],
        fmt="o",
        ecolor="#999999",
        elinewidth=2,
        capsize=4,
        markersize=5,
        color="#1f77b4",
        label="Calibrated Organ Age ± 95% CI",
    )
    ax.axvline(chrono_age, color="#ff7f0e", linestyle="--", linewidth=2, label="Chronological Age")
    ax.set_yticks(y)
    ax.set_yticklabels(organs, fontsize=9)
    ax.set_xlabel("Age (years)", fontsize=10)
    ax.set_title("Organ Ages with Confidence Intervals", fontsize=12)
    ax.legend(fontsize=8)


def build_report(subject_id: str,
                 cal_path: str = CAL_PATH,
                 panel_path: str = PANEL_PATH,
                 out_dir: str = "reports/v4/isef"):
    """
    Build and save the two-page ISEF-style poster PDF report.

    Parameters
    ----------
    subject_id : str
        Unique subject identifier present in both ``cal_path`` and
        ``panel_path``.
    cal_path : str
        Path to the calibrated organ-age parquet.
    panel_path : str
        Path to the subject-level panel parquet.
    out_dir : str
        Directory where the PDF will be written.

    Raises
    ------
    ValueError
        If ``subject_id`` is not found in the panel index.
    RuntimeError
        If no calibrated rows exist for ``subject_id``.
    """

    df_cal, df_panel = load_data(cal_path, panel_path)
    if subject_id not in df_panel.index:
        raise ValueError(f"Subject '{subject_id}' not found in panel {panel_path}")

    df_sub = df_cal[df_cal["subject_id"] == subject_id].copy()
    if df_sub.empty:
        raise RuntimeError(f"No calibrated rows for subject '{subject_id}'")

    df_org = aggregate_organs(df_sub)
    row = df_panel.loc[subject_id]

    chrono = float(df_org["age_chrono"].mean())
    bio = float(row["bio_age_weighted"])
    delta_w = float(row["delta_weighted"])
    z_w = float(row["z_weighted"])
    burden = float(row["burden_abs_z"])
    n_org = int(row["n_organs"])
    n_abn = int(row["n_abnormal_organs"])

    organs = df_org["organ"].tolist()
    zscores = df_org["zscore_cal"].to_numpy()
    deltas = df_org["age_delta_cal"].to_numpy()
    cal_ages = df_org["age_pred_cal"].to_numpy()
    ci_low = df_org["ci_lower"].to_numpy()
    ci_high = df_org["ci_upper"].to_numpy()

    out_dir_path = Path(out_dir)
    out_dir_path.mkdir(parents=True, exist_ok=True)
    out_pdf = out_dir_path / f"OrganAge_isef_{subject_id}.pdf"

    with PdfPages(out_pdf) as pdf:
        # Page 1: poster-friendly summary
        fig, ax = plt.subplots(figsize=(8.5, 11))
        ax.axis("off")
        fig.patch.set_facecolor("white")

        title = f"OrganAge-v4 – Subject-Level Organ Aging Profile\nSubject: {subject_id}"
        fig.suptitle(title, fontsize=18, color="#7b2cff", y=0.96)

        # left column: numeric summary
        fig.text(0.05, 0.80, "Global Metrics", fontsize=14, weight="bold", color="#7b2cff")
        summary = (
            f"Chronological age: {chrono:.1f} years\n"
            f"Composite biological age (weighted): {bio:.1f} years\n"
            f"Composite delta (bio - chrono): {delta_w:.1f} years\n"
            f"Composite z-score: {z_w:.2f}\n"
            f"Organs modeled: {n_org}\n"
            f"Organs with |z| ≥ 2: {n_abn}\n"
            f"Mean |organ z| (organ-age burden): {burden:.2f}"
        )
        fig.text(0.05, 0.57, summary, fontsize=11)

        # right column: explanatory box
        fig.text(
            0.55,
            0.80,
            "How to read this profile:",
            fontsize=14,
            weight="bold",
            color="#ff7f0e",
        )
        explanation = (
            "• Each organ's biological age is predicted from multi-modal\n"
            "  embeddings (transcriptomics + imaging) and calibrated\n"
            "  against organ-specific normative curves.\n"
            "• Negative Δ indicates an organ is biologically younger than\n"
            "  expected for this chronological age; positive Δ indicates\n"
            "  accelerated aging.\n"
            "• Z-scores express this deviation in units of standard deviation\n"
            "  relative to the healthy reference distribution."
        )
        fig.text(0.55, 0.57, explanation, fontsize=9)

        # organ table for quick poster inclusion
        cols = [
            "organ",
            "age_pred_cal",
            "age_delta_cal",
            "zscore_cal",
            "ci_lower",
            "ci_upper",
        ]
        tbl_df = df_org[cols]
        table_ax = fig.add_axes([0.05, 0.05, 0.9, 0.45])
        table_ax.axis("off")

        table = table_ax.table(
            cellText=tbl_df.values,
            colLabels=tbl_df.columns,
            loc="center",
            cellLoc="center",
        )
        table.auto_set_font_size(False)
        table.set_fontsize(8)
        table.scale(1, 1.3)

        pdf.savefig(fig)
        plt.close(fig)

        # Page 2: visual layout for poster
        fig = plt.figure(figsize=(8.5, 11))
        fig.patch.set_facecolor("white")
        gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.1])

        ax1 = fig.add_subplot(gs[0, 0], polar=True)
        radar_plot(ax1, organs, zscores)

        ax2 = fig.add_subplot(gs[0, 1])
        delta_bar_plot(ax2, organs, deltas)

        ax3 = fig.add_subplot(gs[1, :])
        ci_plot(ax3, chrono, organs, cal_ages, ci_low, ci_high)

        fig.suptitle("OrganAge-v4 – Visualization Panel (Ready for Poster)",
                     fontsize=16, y=0.98, color="#7b2cff")
        pdf.savefig(fig)
        plt.close(fig)

    print(f"[ISEF] Saved report -> {out_pdf}")


def main():
    """
    Command-line entry point for the ISEF-style report generator.

    Parses subject ID, data paths, and output directory from CLI arguments
    and delegates to ``build_report``.
    """
    parser = argparse.ArgumentParser(description="Generate ISEF-style OrganAge-v4 report.")
    parser.add_argument("--subject", type=str, required=True)
    parser.add_argument("--calibrated", type=str, default=CAL_PATH)
    parser.add_argument("--panel", type=str, default=PANEL_PATH)
    parser.add_argument("--outdir", type=str, default="reports/v4/isef")
    args = parser.parse_args()

    build_report(args.subject, args.calibrated, args.panel, args.outdir)


if __name__ == "__main__":
    main()
