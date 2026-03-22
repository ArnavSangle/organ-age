"""
viz_umap_multimodal_aligned_v3.py - UMAP visualisation for v3.5 contrastive-aligned embeddings.

Loads the contrastive-aligned multimodal table
(``v3_output/data/v3_aligned_contrastive.parquet``), stacks all ``z_*``
columns into a single matrix, reduces it to 2-D with UMAP, and saves two
scatter plots:

  1. Colour-coded by chronological **age** (viridis colour map).
  2. Colour-coded by **modality** (tab10: RNA=0, Xray=1, MRI=2).

Outputs are written to ``v3_output/plots/``.
"""

import pandas as pd
import numpy as np
import umap
import matplotlib.pyplot as plt
from pathlib import Path

OUT = Path("v3_output")
DATA = OUT / "data"
PLOT = OUT / "plots"
PLOT.mkdir(parents=True, exist_ok=True)


def main():
    """
    Load the contrastive-aligned parquet, run UMAP dimensionality reduction,
    and save scatter plots coloured by age and by modality.

    The UMAP reducer uses ``n_components=2`` and ``random_state=42`` for
    reproducibility.  NaN / inf values in the embedding matrix are replaced
    with 0.0 before fitting.
    """
    df_path = DATA / "v3_aligned_contrastive.parquet"
    print(f"[UMAP v3.5] Loading {df_path} ...")
    df = pd.read_parquet(df_path)

    print("[UMAP v3.5] Modality counts:")
    print(df["modality"].value_counts())

    # Use only aligned embeddings: z_rna_*, z_xray_*, z_mri_*
    emb_cols = [c for c in df.columns if c.startswith("z_")]
    if not emb_cols:
        raise RuntimeError("[UMAP v3.5] No aligned embedding columns (z_*) found in dataframe")

    print(f"[UMAP v3.5] Using {len(emb_cols)} embedding columns")
    X = df[emb_cols].to_numpy(dtype="float32")

    # Rows will have NaNs where a given modality block doesn't exist -> fill with 0
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    print(f"[UMAP v3.5] X shape = {X.shape}")

    reducer = umap.UMAP(n_components=2, random_state=42)
    U = reducer.fit_transform(X)

    # --- Plot by age ---
    plt.figure(figsize=(6, 6))
    sc = plt.scatter(U[:, 0], U[:, 1], s=3, c=df["age"].values, cmap="viridis")
    plt.colorbar(sc, label="Age")
    plt.title("V3.5 Multimodal UMAP (age, contrastive aligned)")
    plt.tight_layout()
    age_path = PLOT / "umap_multimodal_age_v3_contrastive.png"
    plt.savefig(age_path, dpi=200)
    plt.close()
    print("[UMAP v3.5] Saved age plot ->", age_path)

    # --- Plot by modality ---
    mod_map = {"rna": 0, "xray": 1, "mri": 2}
    C = df["modality"].map(mod_map).fillna(-1).to_numpy()

    plt.figure(figsize=(6, 6))
    sc2 = plt.scatter(U[:, 0], U[:, 1], s=3, c=C, cmap="tab10")
    plt.title("V3.5 Multimodal UMAP (modality, contrastive aligned)")
    plt.tight_layout()
    mod_path = PLOT / "umap_multimodal_modality_v3_contrastive.png"
    plt.savefig(mod_path, dpi=200)
    plt.close()
    print("[UMAP v3.5] Saved modality plot ->", mod_path)


if __name__ == "__main__":
    main()
