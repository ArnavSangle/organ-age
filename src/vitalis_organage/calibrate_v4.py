"""
calibrate_v4.py
===============
Post-hoc isotonic calibration of raw v4.5 organ-age predictions.

Takes the ``organ_age_normative.parquet`` table produced by
``build_organ_age_targets.py`` and, for each organ independently:

  1. Fits a monotonic isotonic regression mapping raw model predictions
     (``age_pred``) to chronological age using only the designated healthy
     reference cohort (e.g. GTEx, IXI).
  2. Applies the fitted regressor to all samples of that organ to produce
     calibrated predictions (``age_pred_cal``).
  3. Computes calibrated delta (``age_delta_cal``), z-score (``zscore_cal``),
     and a constant-width 90 % confidence interval (``ci_lower``, ``ci_upper``)
     derived from the residual distribution of the reference cohort.

Output is saved to ``data/analysis/organ_age_calibrated.parquet``.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression


# -------------------------------------------------------------
# Utility: isotonic calibration for monotonic remapping
# -------------------------------------------------------------

def calibrate_isotonic(x_ref, y_ref, x_all):
    """
    Fit isotonic regression y ≈ iso(x) on reference cohort.
    Return calibrated predictions for x_all.
    """
    ir = IsotonicRegression(out_of_bounds="clip")
    ir.fit(x_ref, y_ref)
    y_cal = ir.predict(x_all)
    return y_cal, ir


# -------------------------------------------------------------
# Uncertainty estimation: quantile bandwidths
# -------------------------------------------------------------

def estimate_uncertainty(residuals, confidence=0.90):
    """
    Given residual distribution (reference cohort),
    compute lower/upper quantiles for CI.
    """
    alpha = (1 - confidence) / 2
    lo = np.quantile(residuals, alpha)
    hi = np.quantile(residuals, 1 - alpha)
    return lo, hi


# -------------------------------------------------------------
# ORG-level calibration (POSitional indexing only)
# -------------------------------------------------------------

def calibrate_per_organ(df: pd.DataFrame,
                        healthy_sources,
                        min_ref_n=50) -> pd.DataFrame:
    """
    Take organ_age_normative table and:
      - calibrate age_pred -> age_pred_cal via isotonic regression
      - compute delta, z-score, and constant CI band per organ

    Uses numpy masks + df.iloc[...] exclusively for assignments
    to avoid any index alignment weirdness.
    """
    df = df.copy()

    # Add new columns
    df["age_pred_cal"] = np.nan
    df["age_delta_cal"] = np.nan
    df["zscore_cal"] = np.nan
    df["ci_lower"] = np.nan
    df["ci_upper"] = np.nan

    # Precompute arrays + column positions
    organ_arr = df["organ"].to_numpy()
    source_arr = df["source"].to_numpy()
    age_chrono_arr = df["age_chrono"].to_numpy(dtype="float32")
    age_pred_arr = df["age_pred"].to_numpy(dtype="float32")

    age_pred_cal_col = df.columns.get_loc("age_pred_cal")
    age_delta_cal_col = df.columns.get_loc("age_delta_cal")
    zscore_cal_col = df.columns.get_loc("zscore_cal")
    ci_lower_col = df.columns.get_loc("ci_lower")
    ci_upper_col = df.columns.get_loc("ci_upper")

    healthy_set = set(healthy_sources)

    organs = sorted(pd.unique(df["organ"].dropna()))
    print(f"[CALIB] Found {len(organs)} organs:", organs)

    for organ in organs:
        # Mask over all rows for this organ
        mask_organ = (organ_arr == organ)
        if not mask_organ.any():
            continue

        # Reference cohort mask
        mask_ref = (
            mask_organ
            & np.isin(source_arr, list(healthy_set))
            & np.isfinite(age_chrono_arr)
            & np.isfinite(age_pred_arr)
        )
        ref_idx = np.flatnonzero(mask_ref)
        n_ref = ref_idx.size

        if n_ref < min_ref_n:
            print(f"[CALIB] organ={organ:20s} | REF N={n_ref:4d} < {min_ref_n}, skipping.")
            continue

        print(f"[CALIB] organ={organ:20s} | REF N={n_ref:4d} | calibrating...")

        x_ref = age_pred_arr[ref_idx]        # model predictions
        y_ref = age_chrono_arr[ref_idx]      # true chronological age

        # ---- 1) Isotonic calibration y ≈ iso(x) ----
        pred_cal_ref, ir = calibrate_isotonic(x_ref, y_ref, x_ref)

        # Apply to ALL valid rows of this organ
        mask_all = (
            mask_organ
            & np.isfinite(age_chrono_arr)
            & np.isfinite(age_pred_arr)
        )
        all_idx = np.flatnonzero(mask_all)
        if all_idx.size == 0:
            continue

        x_all = age_pred_arr[all_idx]
        y_all = age_chrono_arr[all_idx]

        pred_cal_all = ir.predict(x_all)

        # ---- 2) Delta + z-score ----
        delta_all = pred_cal_all - y_all

        # residuals in reference cohort (for std / CI)
        ref_resid = y_ref - pred_cal_ref
        std_ref = ref_resid.std(ddof=1)
        if std_ref < 1e-6 or not np.isfinite(std_ref):
            z_all = np.zeros_like(delta_all)
            print(f"[CALIB] organ={organ:20s} | std_resid ~0; z-scores set to 0.")
        else:
            z_all = delta_all / std_ref

        # ---- 3) Confidence intervals (constant band per organ) ----
        lo_off, hi_off = estimate_uncertainty(ref_resid, confidence=0.90)

        ci_lo_all = pred_cal_all + lo_off
        ci_hi_all = pred_cal_all + hi_off

        # ---- 4) Safety checks ----
        n_all = all_idx.size
        if not (len(pred_cal_all) == len(delta_all) == len(z_all) == len(ci_lo_all) == len(ci_hi_all) == n_all):
            raise RuntimeError(
                f"[CALIB] Length mismatch in organ '{organ}': "
                f"n_all={n_all}, len(pred_cal_all)={len(pred_cal_all)}, "
                f"len(delta_all)={len(delta_all)}, len(z_all)={len(z_all)}"
            )

        # ---- 5) Positional assignment into df ----
        df.iloc[all_idx, age_pred_cal_col] = pred_cal_all
        df.iloc[all_idx, age_delta_cal_col] = delta_all
        df.iloc[all_idx, zscore_cal_col] = z_all
        df.iloc[all_idx, ci_lower_col] = ci_lo_all
        df.iloc[all_idx, ci_upper_col] = ci_hi_all

        print(
            f"[CALIB] {organ:20s} | "
            f"Δμ={delta_all.mean():.2f}, Δσ={delta_all.std():.2f}, "
            f"zμ={z_all.mean():.2f}"
        )

    return df


# -------------------------------------------------------------
# MAIN
# -------------------------------------------------------------

def main():
    """
    Command-line entry point for the v4 calibration pipeline.

    Loads the normative organ-age table, runs ``calibrate_per_organ`` with the
    supplied healthy-source list, and writes the calibrated table to disk.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str,
                        default="data/analysis/organ_age_normative.parquet")
    parser.add_argument("--healthy_sources", type=str, default="gtex_v10,ixi")
    parser.add_argument("--out", type=str,
                        default="data/analysis/organ_age_calibrated.parquet")
    parser.add_argument("--min_ref_n", type=int, default=50)
    args = parser.parse_args()

    healthy = [x.strip() for x in args.healthy_sources.split(",") if x.strip()]

    print("[CALIB] Loading:", args.input)
    df = pd.read_parquet(args.input)
    print("[CALIB] df:", df.shape)

    df_cal = calibrate_per_organ(
        df,
        healthy_sources=healthy,
        min_ref_n=args.min_ref_n,
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df_cal.to_parquet(out_path)

    print("[CALIB] Saved calibrated organ-age table →", out_path)
    print("[CALIB] Done.")


if __name__ == "__main__":
    main()
