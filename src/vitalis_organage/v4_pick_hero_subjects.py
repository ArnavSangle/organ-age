"""
v4_pick_hero_subjects.py
========================
Identify "hero" subjects suitable for use as illustrative case studies in
figures and presentations.

A hero subject is defined as one who:
  * Has measurements for at least 7 distinct organs (comprehensive profiling).
  * Has a small mean absolute organ-age delta (minimal deviation from expected),
    making them biologically typical and useful as a reference example.

The script:
  1. Loads the calibrated organ-age table from
     ``data/analysis/organ_age_calibrated.parquet``.
  2. Computes per-subject organ count and mean absolute delta.
  3. Filters to subjects with >= 7 organs.
  4. Sorts by (organ count descending, mean absolute delta ascending).
  5. Prints the top 20 candidates and saves the full ranked list to
     ``data/analysis/v4_hero_candidates.csv``.
"""
from pathlib import Path
import pandas as pd


def main():
    """
    Entry point: load calibrated data, rank subjects by suitability as hero
    cases, print the top 20, and save the full candidate table to CSV.
    """
    data_path = Path("data/analysis/organ_age_calibrated.parquet")
    print(f"[HERO] Loading calibrated organ-age table from {data_path} ...")
    df = pd.read_parquet(data_path)
    print("[HERO] Shape:", df.shape)

    # Ensure the columns we need exist
    required = {"subject_id", "organ", "organ_age_delta"}
    missing = required - set(df.columns)
    if missing:
        raise RuntimeError(f"[HERO] Missing required columns: {missing}")

    # Compute per-subject stats
    grp = df.groupby("subject_id")
    n_organs = grp["organ"].nunique()
    mean_abs_delta = grp["organ_age_delta"].apply(
        lambda s: (s.abs()).mean()
    )

    summary = (
        pd.DataFrame({
            "n_organs": n_organs,
            "mean_abs_delta": mean_abs_delta,
        })
        .reset_index()
    )

    # Filter to subjects with at least, say, 7 organs measured
    summary = summary[summary["n_organs"] >= 7]

    # Sort by (many organs, small mean_abs_delta)
    summary = summary.sort_values(
        ["n_organs", "mean_abs_delta"],
        ascending=[False, True]
    )

    print("\n[HERO] Top 20 candidate subjects with many organs and modest deviation:")
    print(summary.head(20).to_string(index=False))

    out_path = Path("data/analysis/v4_hero_candidates.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out_path, index=False)
    print(f"\n[HERO] Saved full candidate table -> {out_path}")


if __name__ == "__main__":
    main()
