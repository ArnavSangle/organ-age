"""
v4_compare_schemes.py
=====================
Comparison utility for the three v4 panel weighting schemes: balanced,
aggressive, and conservative.

For each scheme this script prints descriptive statistics (biological age,
delta, z-score, abnormal-organ counts, and burden) and then computes
pairwise Pearson correlations between all key metrics across all pairs of
schemes.

Typical usage::

    python v4_compare_schemes.py

Assumes that all three panel parquets already exist under
``data/analysis/``.
"""
import pandas as pd
import numpy as np

from pathlib import Path


def load_panel(path):
    """
    Load a v4 panel parquet and sort by subject_id for reproducible ordering.

    Parameters
    ----------
    path : pathlib.Path or str
        Path to the panel ``.parquet`` file.

    Returns
    -------
    pd.DataFrame
        Panel DataFrame sorted by ``subject_id``.
    """
    df = pd.read_parquet(path)
    df = df.copy()
    df = df.sort_values("subject_id").reset_index(drop=True)
    return df


def describe_scheme(df, name):
    """
    Print descriptive statistics for a single weighting scheme panel.

    Outputs subject count and distribution summaries for biological age,
    delta, z-score, number of abnormal organs, and burden (mean absolute
    z-score).

    Parameters
    ----------
    df : pd.DataFrame
        Panel DataFrame for a single weighting scheme.
    name : str
        Human-readable scheme name used in the printed header.
    """
    print(f"\n=== {name.upper()} SCHEME ===")
    print("Subjects:", len(df))

    print("Biological Age Stats:")
    print(df["bio_age_weighted"].describe())

    print("\nDelta Stats (bio_age - chrono):")
    print(df["delta_weighted"].describe())

    print("\nZ-Score Stats:")
    print(df["z_weighted"].describe())

    print("\n# Abnormal Organs:")
    print(df["n_abnormal_organs"].value_counts().sort_index())

    print("\nBurden abs(Z) Stats:")
    print(df["burden_abs_z"].describe())


def compute_correlations(dfA, dfB, nameA, nameB):
    """
    Print pairwise Pearson correlations between two scheme panels.

    Merges the two panels on ``subject_id`` and prints Pearson r for
    ``bio_age_weighted``, ``delta_weighted``, ``z_weighted``, and
    ``burden_abs_z``.

    Parameters
    ----------
    dfA : pd.DataFrame
        Panel DataFrame for the first scheme.
    dfB : pd.DataFrame
        Panel DataFrame for the second scheme.
    nameA : str
        Label for the first scheme (used in printed output).
    nameB : str
        Label for the second scheme (used in printed output).
    """
    merged = pd.merge(
        dfA,
        dfB,
        on="subject_id",
        suffixes=(f"_{nameA}", f"_{nameB}")
    )

    print(f"\n=== CORRELATION: {nameA.upper()} vs {nameB.upper()} ===")
    print("bio_age_weighted corr:",
          merged[f"bio_age_weighted_{nameA}"].corr(merged[f"bio_age_weighted_{nameB}"]))

    print("delta_weighted corr:",
          merged[f"delta_weighted_{nameA}"].corr(merged[f"delta_weighted_{nameB}"]))

    print("z_weighted corr:",
          merged[f"z_weighted_{nameA}"].corr(merged[f"z_weighted_{nameB}"]))

    print("burden_abs_z corr:",
          merged[f"burden_abs_z_{nameA}"].corr(merged[f"burden_abs_z_{nameB}"]))


def main():
    """
    Load all three scheme panels, describe each, and print pairwise correlations.
    """
    # Load all 3 schemes
    p_bal = Path("data/analysis/v4_panel_balanced.parquet")
    p_agg = Path("data/analysis/v4_panel_aggressive.parquet")
    p_con = Path("data/analysis/v4_panel_conservative.parquet")

    df_bal = load_panel(p_bal)
    df_agg = load_panel(p_agg)
    df_con = load_panel(p_con)

    # Describe each scheme
    describe_scheme(df_bal, "balanced")
    describe_scheme(df_agg, "aggressive")
    describe_scheme(df_con, "conservative")

    # Correlations between schemes
    compute_correlations(df_bal, df_agg, "balanced", "aggressive")
    compute_correlations(df_bal, df_con, "balanced", "conservative")
    compute_correlations(df_agg, df_con, "aggressive", "conservative")


if __name__ == "__main__":
    main()
