"""
v4_panel.py
===========
Build subject-level OrganAge composite panels from the calibrated organ-age
table (output of ``calibrate_v4.py``).

Three weight schemes are provided (``balanced``, ``aggressive``,
``conservative``) that assign different importance weights to each organ
when computing the weighted composite biological age.

For every subject with at least one calibrated organ the script produces:
  - ``bio_age_weighted``    : weighted mean of per-organ calibrated ages
  - ``delta_weighted``      : composite biological age minus chronological age
  - ``z_weighted``          : weighted mean z-score
  - ``n_organs``            : number of organs with valid calibrated ages
  - ``n_abnormal_organs``   : organs with ``|z| >= ABNORMAL_Z``
  - ``burden_abs_z``        : mean absolute z-score across organs
  - ``top3_old_organs``     : three organs with largest positive delta
  - ``top3_young_organs``   : three organs with largest negative delta

Output is saved as ``data/analysis/v4_panel_<scheme>.parquet``.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd


# -----------------------------------------------------------
# Weight schemes
# -----------------------------------------------------------

WEIGHTS_SCHEMES = {
    "balanced": {
        "brain":           1.3,
        "brain_cortex":    1.3,
        "heart":           1.3,
        "liver":           1.3,
        "kidney":          1.3,
        "skeletal_muscle": 1.0,
        "colon":           1.0,
        "lung":            1.0,
        "adipose":         0.9,
        "skin":            0.9,
        "whole_blood":     0.9,
    },
    "aggressive": {
        "brain":           1.5,
        "brain_cortex":    1.5,
        "heart":           1.5,
        "liver":           1.5,
        "kidney":          1.5,
        "skeletal_muscle": 1.0,
        "colon":           1.0,
        "lung":            1.0,
        "adipose":         0.7,
        "skin":            0.7,
        "whole_blood":     0.7,
    },
    "conservative": {
        "brain":           1.15,
        "brain_cortex":    1.15,
        "heart":           1.15,
        "liver":           1.15,
        "kidney":          1.15,
        "skeletal_muscle": 1.0,
        "colon":           1.0,
        "lung":            1.0,
        "adipose":         0.95,
        "skin":            0.95,
        "whole_blood":     0.9,
    },
}


ABNORMAL_Z = 2.0


def load_calibrated(path: str) -> pd.DataFrame:
    """
    Load the calibrated organ-age parquet and validate required columns.

    Parameters
    ----------
    path : str
        Path to the calibrated organ-age ``.parquet`` file.

    Returns
    -------
    pd.DataFrame
        Validated calibrated organ-age table.

    Raises
    ------
    RuntimeError
        If any required column is missing.
    """
    df = pd.read_parquet(path)
    expected = [
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
    missing = [c for c in expected if c not in df.columns]
    if missing:
        raise RuntimeError(f"[V4 PANEL] Missing columns in calibrated table: {missing}")
    return df


def get_weights_for_scheme(scheme: str) -> dict:
    """
    Return the organ-weight dictionary for the requested weighting scheme.

    Parameters
    ----------
    scheme : str
        One of ``'balanced'``, ``'aggressive'``, or ``'conservative'``.

    Returns
    -------
    dict
        Mapping of organ name -> float weight.

    Raises
    ------
    ValueError
        If ``scheme`` is not one of the defined keys in ``WEIGHTS_SCHEMES``.
    """
    if scheme not in WEIGHTS_SCHEMES:
        raise ValueError(
            f"[V4 PANEL] Unknown weight scheme '{scheme}'. "
            f"Available: {list(WEIGHTS_SCHEMES.keys())}"
        )
    return WEIGHTS_SCHEMES[scheme]


def summarize_subject_with_weights(df_sub: pd.DataFrame, weights: dict):
    """
    Given all rows for a single subject, compute:

      - composite biological age (weighted mean of organ_age_cal)
      - composite delta, z
      - organ-wise stats
      - burden metrics

    Returns a dict, or None if the subject has no valid calibrated organs.
    """
    if df_sub.empty:
        return None

    # Aggregate per organ
    per_organ = (
        df_sub.groupby("organ", dropna=False)
        .agg(
            n_samples=("organ", "size"),
            age_chrono=("age_chrono", "mean"),
            organ_age_cal=("age_pred_cal", "mean"),
            delta_cal=("age_delta_cal", "mean"),
            z_cal=("zscore_cal", "mean"),
        )
        .reset_index()
    )

    # Chronological age (should be same across organs; use mean)
    age_chrono = float(per_organ["age_chrono"].mean())

    # Assign weights
    def organ_weight(row):
        """
        Look up the weight for ``row['organ']``, defaulting to 1.0 if not
        explicitly listed in the weight scheme.

        Parameters
        ----------
        row : pd.Series
            A single row from ``per_organ`` containing at least ``'organ'``.

        Returns
        -------
        float
            The weight assigned to this organ.
        """
        org = row["organ"]
        return float(weights.get(org, 1.0))

    per_organ["weight"] = per_organ.apply(organ_weight, axis=1)

    # Keep only organs with finite calibrated age
    mask_valid = np.isfinite(per_organ["organ_age_cal"].to_numpy())
    per_organ_valid = per_organ[mask_valid].copy()
    if per_organ_valid.empty:
        # nothing calibrated for this subject → skip
        return None

    # Composite biological age (weighted mean of organ_age_cal)
    w = per_organ_valid["weight"].to_numpy()
    organ_age = per_organ_valid["organ_age_cal"].to_numpy()
    z = per_organ_valid["z_cal"].to_numpy()

    w_sum = w.sum()
    if w_sum <= 0:
        return None

    bio_age = float((w * organ_age).sum() / w_sum)
    composite_delta = float(bio_age - age_chrono)
    composite_z = float((w * z).sum() / w_sum)

    # Burden metrics
    abs_z = np.abs(z)
    burden_abs_z = float(abs_z.mean())
    n_organs = int(per_organ_valid.shape[0])
    n_abnormal = int((abs_z >= ABNORMAL_Z).sum())

    # Identify top 3 oldest / youngest organs (by delta)
    per_organ_sorted_old = per_organ_valid.sort_values(
        by="delta_cal", ascending=False, ignore_index=True
    )
    per_organ_sorted_young = per_organ_valid.sort_values(
        by="delta_cal", ascending=True, ignore_index=True
    )

    def pack_top3(df_top):
        """
        Serialise the top-3 organs and their deltas into a pipe-delimited string.

        Parameters
        ----------
        df_top : pd.DataFrame
            Up to 3 rows with at least ``'organ'`` and ``'delta_cal'``
            columns.

        Returns
        -------
        str
            Pipe-separated string of ``'organ:delta'`` pairs, e.g.
            ``'liver:4.21|heart:3.10|kidney:1.55'``.
        """
        return "|".join(
            f"{row.organ}:{row.delta_cal:.2f}"
            for _, row in df_top.iterrows()
        )

    top3_old_str = pack_top3(per_organ_sorted_old.head(3))
    top3_young_str = pack_top3(per_organ_sorted_young.head(3))

    out = {
        "subject_id": str(df_sub["subject_id"].iloc[0]),
        "age_chrono": age_chrono,
        "bio_age_weighted": bio_age,
        "delta_weighted": composite_delta,
        "z_weighted": composite_z,
        "n_organs": n_organs,
        "n_abnormal_organs": n_abnormal,
        "burden_abs_z": burden_abs_z,
        "top3_old_organs": top3_old_str,
        "top3_young_organs": top3_young_str,
    }

    return out


def build_panel(df: pd.DataFrame, scheme: str) -> pd.DataFrame:
    """
    Build a subject-level panel DataFrame using the specified weighting scheme.

    Iterates over unique subjects in ``df``, calls
    ``summarize_subject_with_weights`` for each, and collects the results into
    a sorted DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Calibrated organ-age table (all subjects combined).
    scheme : str
        Name of the weighting scheme to use (see ``WEIGHTS_SCHEMES``).

    Returns
    -------
    pd.DataFrame
        One row per subject with composite biological age metrics.
    """
    weights = get_weights_for_scheme(scheme)
    print(f"[V4 PANEL] Using scheme '{scheme}' with weights:\n{weights}\n")

    rows = []
    skipped = 0

    for subj_id, df_sub in df.groupby("subject_id", sort=False):
        summary = summarize_subject_with_weights(df_sub, weights)
        if summary is None:
            skipped += 1
            continue
        rows.append(summary)

    print(f"[V4 PANEL] Subjects with valid organs: {len(rows)}")
    print(f"[V4 PANEL] Subjects skipped (no valid calibrated organs): {skipped}")

    panel_df = pd.DataFrame(rows)
    panel_df = panel_df.sort_values(by="subject_id", ignore_index=True)
    return panel_df


def main():
    """
    Command-line entry point for the v4 panel builder.

    Loads the calibrated organ-age table, builds the subject-level panel with
    the chosen weighting scheme, and writes the resulting parquet to disk.
    """
    parser = argparse.ArgumentParser(description="Build v4 subject-level OrganAge panels.")
    parser.add_argument(
        "--calibrated_path",
        type=str,
        default="data/analysis/organ_age_calibrated.parquet",
        help="Path to calibrated organ-age table.",
    )
    parser.add_argument(
        "--scheme",
        type=str,
        default="balanced",
        choices=list(WEIGHTS_SCHEMES.keys()),
        help="Weight scheme to use for composite biological age.",
    )
    parser.add_argument(
        "--out_path",
        type=str,
        default=None,
        help=(
            "Output parquet path for subject-level panels. "
            "Default: data/analysis/v4_panel_<scheme>.parquet"
        ),
    )
    args = parser.parse_args()

    print("[V4 PANEL] Loading calibrated table:", args.calibrated_path)
    df = load_calibrated(args.calibrated_path)
    print("[V4 PANEL] Calibrated shape:", df.shape)

    panel_df = build_panel(df, scheme=args.scheme)
    print("[V4 PANEL] Panel shape:", panel_df.shape)
    print("[V4 PANEL] Head:\n", panel_df.head().to_string(index=False))

    if args.out_path is None:
        out_path = Path(f"data/analysis/v4_panel_{args.scheme}.parquet")
    else:
        out_path = Path(args.out_path)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    panel_df.to_parquet(out_path)
    print("[V4 PANEL] Saved panel ->", out_path)


if __name__ == "__main__":
    main()
