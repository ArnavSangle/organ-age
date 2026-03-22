# scripts/61_gtex_pca.py
"""
encode_gtex_pca_v1.py

Reduce the high-dimensional GTEx RNA expression features
(~59 k genes) to a compact PCA representation suitable for downstream
age regression and fusion experiments.

Pipeline:
  1. Load ``data/processed/unified_gtex.parquet`` (produced by
     build_unified_dataset_v1.py).
  2. Standardise the feature matrix with StandardScaler.
  3. Fit a PCA with ``--n_components`` components (default 1024).
  4. Replace the 'features' column with the projected vectors.
  5. Save the compressed parquet to ``data/processed/unified_gtex_pca.parquet``.
  6. Persist the fitted scaler + PCA object to ``models/gtex_pca.joblib``
     for reproducibility and inference-time encoding.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import joblib


ROOT = Path(".")
PROCESSED = ROOT / "data" / "processed"
MODELS = ROOT / "models"


def main():
    """
    Parse CLI arguments, load GTEx features, standardise, apply PCA, then
    save the compressed parquet and the fitted PCA / scaler artefacts.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=str,
        default=str(PROCESSED / "unified_gtex.parquet"),
        help="Path to original GTEx unified parquet",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(PROCESSED / "unified_gtex_pca.parquet"),
        help="Path to PCA-compressed GTEx parquet",
    )
    parser.add_argument(
        "--n_components",
        type=int,
        default=1024,
        help="Number of PCA components",
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    MODELS.mkdir(parents=True, exist_ok=True)

    print(f"[PCA] Loading GTEx from {input_path}")
    df = pd.read_parquet(input_path)

    if "features" not in df.columns:
        raise ValueError("Expected 'features' column in GTEx parquet.")

    # Build feature matrix
    print("[PCA] Building feature matrix...")
    feats = df["features"].tolist()
    X = np.asarray(feats, dtype=np.float32)
    print(f"[PCA] X shape: {X.shape}")

    # Standardize features
    print("[PCA] Standardizing features...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Fit PCA
    print(f"[PCA] Fitting PCA with n_components={args.n_components}...")
    pca = PCA(n_components=args.n_components, random_state=42)
    X_pca = pca.fit_transform(X_scaled)
    print(f"[PCA] X_pca shape: {X_pca.shape}")

    # Replace features column
    df = df.copy()
    df["features"] = [x.astype("float32") for x in X_pca]

    print(f"[PCA] Saving PCA-compressed GTEx to {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path)

    # Save PCA + scaler for reproducibility
    pca_path = MODELS / "gtex_pca.joblib"
    print(f"[PCA] Saving PCA + scaler to {pca_path}")
    joblib.dump({"scaler": scaler, "pca": pca}, pca_path)

    print("[PCA] Done.")


if __name__ == "__main__":
    main()
