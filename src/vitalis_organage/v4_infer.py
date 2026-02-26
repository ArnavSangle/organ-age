"""
v4_infer.py
===========
Inference and reporting utilities for the v4 OrganAge pipeline.

Loads the calibrated organ-age table (output of ``calibrate_v4.py``) and
exposes two summary views:

  * **Subject-level** (``summarize_subject``): per-organ calibrated age,
    delta, z-score, and confidence intervals for a single subject, with
    organs flagged as abnormal when ``|z| >= ABNORMAL_Z``.

  * **Global** (``summarize_global``): cohort-wide statistics per organ
    including mean/std of delta, mean/std of z-score, and fraction of
    samples meeting the abnormality threshold.

Both summaries can optionally be exported to CSV via CLI flags.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd


ABNORMAL_Z = 2.0  # threshold for flagging organ as abnormal


def load_calibrated_table(path: str) -> pd.DataFrame:
    """
    Load the calibrated organ-age parquet and validate that all required
    columns are present.

    Parameters
    ----------
    path : str
        File-system path to the calibrated organ-age ``.parquet`` file.

    Returns
    -------
    pd.DataFrame
        The loaded table with guaranteed column presence.

    Raises
    ------
    RuntimeError
        If any expected column is absent from the file.
    """
    df = pd.read_parquet(path)
    expected_cols = [
        "subject_id",
        "organ",
        "modality",
        "source",
        "age_chrono",
        "age_pred",
        "organ_age_delta",
        "organ_age_zscore",
        "age_pred_cal",
        "age_delta_cal",
        "zscore_cal",
        "ci_lower",
        "ci_upper",
    ]
    missing = [c for c in expected_cols if c not in df.columns]
    if missing:
        raise RuntimeError(f"[V4 INFER] Missing columns in calibrated table: {missing}")
    return df


def summarize_subject(df: pd.DataFrame, subject_id: str) -> pd.DataFrame:
    """
    For a given subject_id, aggregate per organ:
      - chronological age (mean)
      - calibrated organ age (mean)
      - calibrated delta (mean)
      - calibrated z-score (mean)
      - CI lower / upper (mean)
      - number of samples
      - abnormal flag (|z| >= ABNORMAL_Z)
    """
    df_sub = df[df["subject_id"] == subject_id].copy()
    if df_sub.empty:
        raise ValueError(f"[V4 INFER] No rows found for subject_id='{subject_id}'")

    grouped = (
        df_sub.groupby("organ", dropna=False)
        .agg(
            n_samples=("organ", "size"),
            age_chrono=("age_chrono", "mean"),
            organ_age_cal=("age_pred_cal", "mean"),
            delta_cal=("age_delta_cal", "mean"),
            z_cal=("zscore_cal", "mean"),
            ci_lower=("ci_lower", "mean"),
            ci_upper=("ci_upper", "mean"),
        )
        .reset_index()
    )

    # Flag abnormal organs
    grouped["flag_abnormal"] = grouped["z_cal"].abs() >= ABNORMAL_Z

    # Sort: abnormal first, then by |z|
    grouped = grouped.sort_values(
        by=["flag_abnormal", "z_cal"],
        ascending=[False, False],
        ignore_index=True,
    )

    return grouped


def summarize_global(df: pd.DataFrame) -> pd.DataFrame:
    """
    Global snapshot across all subjects:
      - per organ: mean/std of delta & z, fraction abnormal.
    """
    def frac_abnormal(z):
        """
        Compute the fraction of z-scores whose absolute value meets or
        exceeds ``ABNORMAL_Z``.

        Parameters
        ----------
        z : pd.Series
            Series of z-score values (may contain NaN).

        Returns
        -------
        float
            Fraction in [0, 1], or ``np.nan`` if ``z`` is empty after
            dropping NaN values.
        """
        z = z.dropna()
        if z.empty:
            return np.nan
        return (z.abs() >= ABNORMAL_Z).mean()

    grouped = (
        df.groupby("organ", dropna=False)
        .agg(
            n_samples=("organ", "size"),
            delta_mean=("age_delta_cal", "mean"),
            delta_std=("age_delta_cal", "std"),
            z_mean=("zscore_cal", "mean"),
            z_std=("zscore_cal", "std"),
            frac_abnormal=("zscore_cal", frac_abnormal),
        )
        .reset_index()
    )

    # order by fraction abnormal descending
    grouped = grouped.sort_values(
        by="frac_abnormal",
        ascending=False,
        ignore_index=True,
    )

    return grouped


def main():
    """
    Command-line entry point for the v4 inference / reporting script.

    Loads the calibrated table, prints a global organ-level summary, and
    (optionally) a per-subject summary.  Either or both summaries can be
    written to CSV with the ``--out_global_csv`` / ``--out_subject_csv`` flags.
    """
    parser = argparse.ArgumentParser(description="V4 organ-age inference / reporting")
    parser.add_argument(
        "--calibrated_path",
        type=str,
        default="data/analysis/organ_age_calibrated.parquet",
        help="Path to calibrated organ-age table.",
    )
    parser.add_argument(
        "--subject_id",
        type=str,
        default=None,
        help="If provided, output per-organ summary for this subject.",
    )
    parser.add_argument(
        "--out_subject_csv",
        type=str,
        default=None,
        help="Optional: path to save subject-level summary CSV.",
    )
    parser.add_argument(
        "--out_global_csv",
        type=str,
        default=None,
        help="Optional: path to save global organ-level summary CSV.",
    )

    args = parser.parse_args()

    # 1) Load calibrated table
    print("[V4 INFER] Loading calibrated table:", args.calibrated_path)
    df = load_calibrated_table(args.calibrated_path)
    print("[V4 INFER] Shape:", df.shape)

    # 2) Optional: global summary
    global_summary = summarize_global(df)
    print("\n[V4 INFER] Global organ-level summary (top rows):")
    print(global_summary.head(15).to_string(index=False))

    if args.out_global_csv:
        out_global = Path(args.out_global_csv)
        out_global.parent.mkdir(parents=True, exist_ok=True)
        global_summary.to_csv(out_global, index=False)
        print("[V4 INFER] Saved global summary ->", out_global)

    # 3) Optional: subject-level summary
    if args.subject_id is not None:
        print(f"\n[V4 INFER] Subject-level summary for subject_id='{args.subject_id}':")
        subj_summary = summarize_subject(df, args.subject_id)
        print(subj_summary.to_string(index=False))

        if args.out_subject_csv:
            out_subj = Path(args.out_subject_csv)
            out_subj.parent.mkdir(parents=True, exist_ok=True)
            subj_summary.to_csv(out_subj, index=False)
            print("[V4 INFER] Saved subject summary ->", out_subj)


if __name__ == "__main__":
    main()
