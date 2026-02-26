"""
viz_umap_unimodal_v3.py

Visualise each modality (RNA / X-ray / MRI) separately in its own 2-D UMAP
projection, using only the embedding columns specific to that modality
(e.g. emb_rna_* for RNA).

One PNG per modality is saved to v3_output/plots/:
  umap_<modality>_age_v3.png

Rows with non-finite embedding values are dropped before the reduction.
"""

import pandas as pd
import numpy as np
import umap
import matplotlib.pyplot as plt
from pathlib import Path

OUT = Path("v3_output")
PLOT = OUT / "plots"
PLOT.mkdir(parents=True, exist_ok=True)


def run_umap(df, modality_name: str):
    """
    Run UMAP for a single modality.

    Uses only the embedding block for that modality:
      rna  -> emb_rna_*
      xray -> emb_xray_*
      mri  -> emb_mri_*
    """
    if df.shape[0] == 0:
        print(f"[UMAP] Skipping {modality_name}: 0 samples in input df")
        return

    # Pick only embedding columns for THIS modality
    emb_prefix = f"emb_{modality_name}_"
    emb_cols = [c for c in df.columns if c.startswith(emb_prefix)]

    if not emb_cols:
        print(f"[UMAP] Skipping {modality_name}: no embedding columns starting with '{emb_prefix}'")
        return

    X = df[emb_cols].to_numpy(dtype="float32")

    # Safety: drop rows with non-finite embeddings (should be rare now)
    finite_mask = np.isfinite(X).all(axis=1)
    n_total = X.shape[0]
    n_bad = (~finite_mask).sum()
    if n_bad > 0:
        print(f"[UMAP] {modality_name}: dropping {n_bad}/{n_total} rows with non-finite embeddings")
        X = X[finite_mask]
        df = df.iloc[finite_mask].reset_index(drop=True)

    if X.shape[0] == 0:
        print(f"[UMAP] Skipping {modality_name}: 0 samples after dropping non-finite rows")
        return

    print(f"[UMAP] Running {modality_name}: X shape = {X.shape}")

    reducer = umap.UMAP(n_components=2, random_state=42)
    U = reducer.fit_transform(X)

    plt.figure(figsize=(6, 6))
    sc = plt.scatter(U[:, 0], U[:, 1], s=3, c=df["age"].values, cmap="viridis")
    plt.title(f"{modality_name} UMAP (age)")
    plt.colorbar(sc, label="Age")
    plt.tight_layout()
    out_path = PLOT / f"umap_{modality_name}_age_v3.png"
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"[UMAP] Saved {modality_name} plot -> {out_path}")


def main():
    """
    Load v3_aligned_base.parquet, then call run_umap() for every unique
    modality present in the data, producing one age-coloured UMAP plot each.
    """
    df_path = OUT / "data" / "v3_aligned_base.parquet"
    print(f"[UMAP] Loading {df_path} ...")
    df = pd.read_parquet(df_path)

    print("[UMAP] Modality counts:")
    print(df["modality"].value_counts())

    for modality in sorted(df["modality"].unique()):
        sub = df[df["modality"] == modality]
        run_umap(sub, modality)


if __name__ == "__main__":
    main()
