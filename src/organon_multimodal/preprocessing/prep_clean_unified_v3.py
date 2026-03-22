"""
prep_clean_unified_v3.py

Pre-process and compress the unified multimodal dataset (RNA / X-ray / MRI)
into per-modality 64-dimensional PCA embeddings.

Steps per modality:
  1. Filter rows by modality label.
  2. Drop samples with any non-finite feature values.
  3. Standardize features with StandardScaler.
  4. Reduce to at most 64 principal components with PCA.
  5. Write per-modality embedding columns (emb_<tag>_*) back to a single
     concatenated parquet file saved to v3_output/data/v3_aligned_base.parquet.

Expected input:  data/processed/unified_all.parquet
Expected output: v3_output/data/v3_aligned_base.parquet
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

OUT = Path("v3_output")
DATA = OUT / "data"
DATA.mkdir(parents=True, exist_ok=True)


def build_modality_embeddings(df_all, modality_name: str, emb_tag: str):
    """
    Filter df by modality, extract 'features', scale + PCA → 64D embeddings.

    Assumes:
      - df_all has columns: ['sample_id','subject_id','age','sex','organ','modality','source','features']
      - `features` is a list/array per row
      - modality column values are 'rna', 'xray', 'mri'
    """
    df_mod = df_all[df_all["modality"] == modality_name].copy()
    if df_mod.empty:
        print(f"[V3 CLEAN] WARNING: no rows for modality '{modality_name}'")
        return df_mod

    print(f"[V3 CLEAN] Processing modality='{modality_name}', N={len(df_mod)}")

    # Turn list-of-arrays into a 2D array
    feats_list = df_mod["features"].to_list()
    X = np.stack(feats_list).astype("float32")  # (N, D)
    print(f"[V3 CLEAN] {modality_name}: raw feature shape = {X.shape}")

    # 1) Drop rows with any non-finite value in features
    finite_mask = np.isfinite(X).all(axis=1)
    n_total = X.shape[0]
    n_bad = (~finite_mask).sum()
    if n_bad > 0:
        print(f"[V3 CLEAN] {modality_name}: dropping {n_bad}/{n_total} rows with non-finite features")
        X = X[finite_mask]
        df_mod = df_mod.iloc[finite_mask].reset_index(drop=True)

    if X.shape[0] == 0:
        print(f"[V3 CLEAN] WARNING: modality '{modality_name}' has 0 clean rows after filtering")
        return df_mod

    print(f"[V3 CLEAN] {modality_name}: clean feature shape = {X.shape}")

    # 2) Standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    print(f"[V3 CLEAN] {modality_name}: scaled")

    # 3) PCA – limit by both samples and feature dimension
    max_components = 64
    n_features = X_scaled.shape[1]
    n_samples = X_scaled.shape[0]
    n_components = min(max_components, n_features, n_samples)
    print(f"[V3 CLEAN] {modality_name}: PCA to {n_components} dims (samples={n_samples}, features={n_features})")

    pca = PCA(n_components=n_components)
    Z = pca.fit_transform(X_scaled)  # (N, K)

    # 4) Make absolutely sure embeddings are finite
    Z = np.nan_to_num(Z, nan=0.0, posinf=0.0, neginf=0.0)

    for j in range(Z.shape[1]):
        df_mod[f"emb_{emb_tag}_{j}"] = Z[:, j]

    return df_mod


def main():
    """
    Entry point: load unified_all.parquet, build per-modality PCA embeddings
    for RNA, X-ray, and MRI, then concatenate and save the result.
    """
    print("[V3 CLEAN] Loading unified data...")
    df = pd.read_parquet("data/processed/unified_all.parquet")
    print("[V3 CLEAN] Loaded:", df.shape)
    print("[V3 CLEAN] Unique modalities:", df["modality"].unique())

    # Your modalities are 'rna', 'xray', 'mri'
    df_rna = build_modality_embeddings(df, "rna",  "rna")
    df_xray = build_modality_embeddings(df, "xray", "xray")
    df_mri = build_modality_embeddings(df, "mri",  "mri")

    # Concatenate only non-empty pieces
    parts = [d for d in [df_rna, df_xray, df_mri] if d is not None and not d.empty]
    if not parts:
        raise RuntimeError("[V3 CLEAN] No modality data found – check df['modality'] values.")

    df_out = pd.concat(parts, axis=0)
    df_out = df_out.sort_index()

    out_path = DATA / "v3_aligned_base.parquet"
    df_out.to_parquet(out_path)
    print("[V3 CLEAN] Saved:", out_path)

    emb_cols = [c for c in df_out.columns if c.startswith("emb_")]
    print("[V3 CLEAN] Number of embedding columns:", len(emb_cols))
    print("[V3 CLEAN] First few embedding cols:", emb_cols[:10])


if __name__ == "__main__":
    main()
