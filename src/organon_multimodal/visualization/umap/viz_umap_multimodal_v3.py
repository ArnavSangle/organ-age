"""
viz_umap_multimodal_v3.py

Visualise all three modalities (RNA / X-ray / MRI) together in a single
2-D UMAP projection computed from the concatenated per-modality embedding
columns (emb_rna_*, emb_xray_*, emb_mri_*) stored in v3_aligned_base.parquet.

Missing modality blocks for a given sample are filled with zeros before
the UMAP reduction so that all rows have the same dimensionality.

Produces two PNG plots in v3_output/plots/:
  - umap_multimodal_age_v3.png      (scatter coloured by age)
  - umap_multimodal_modality_v3.png (scatter coloured by modality)
"""

import pandas as pd
import numpy as np
import umap
import matplotlib.pyplot as plt
from pathlib import Path

OUT = Path("v3_output")
PLOT = OUT / "plots"
PLOT.mkdir(parents=True, exist_ok=True)


def main():
    """
    Load v3_aligned_base.parquet, compute a joint 2-D UMAP over all modality
    embedding columns, and save scatter plots coloured by age and by modality.
    """
    df_path = OUT / "data" / "v3_aligned_base.parquet"
    print(f"[UMAP] Loading {df_path} ...")
    df = pd.read_parquet(df_path)

    print("[UMAP] Modality counts:")
    print(df["modality"].value_counts())

    # Use all embedding columns (rna + xray + mri)
    emb_cols = [c for c in df.columns if c.startswith("emb_")]
    if not emb_cols:
        raise RuntimeError("[UMAP] No embedding columns (emb_*) found in dataframe")

    X = df[emb_cols].to_numpy(dtype="float32")

    # IMPORTANT: rows have NaNs for other-modality blocks → fill them with 0
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    print(f"[UMAP] Multimodal X shape = {X.shape}")

    reducer = umap.UMAP(n_components=2, random_state=42)
    U = reducer.fit_transform(X)

    # --- Plot by age ---
    plt.figure(figsize=(6, 6))
    sc = plt.scatter(U[:, 0], U[:, 1], s=3, c=df["age"].values, cmap="viridis")
    plt.colorbar(sc, label="Age")
    plt.title("V3 Multimodal UMAP (age)")
    plt.tight_layout()
    age_path = PLOT / "umap_multimodal_age_v3.png"
    plt.savefig(age_path, dpi=200)
    plt.close()
    print(f"[UMAP] Saved age plot -> {age_path}")

    # --- Plot by modality ---
    # Your modalities are 'rna', 'xray', 'mri'
    mod_map = {"rna": 0, "xray": 1, "mri": 2}
    C = df["modality"].map(mod_map).fillna(-1).to_numpy()

    plt.figure(figsize=(6, 6))
    sc2 = plt.scatter(U[:, 0], U[:, 1], s=3, c=C, cmap="tab10")
    plt.title("V3 Multimodal UMAP (modality)")
    plt.tight_layout()
    mod_path = PLOT / "umap_multimodal_modality_v3.png"
    plt.savefig(mod_path, dpi=200)
    plt.close()
    print(f"[UMAP] Saved modality plot -> {mod_path}")


if __name__ == "__main__":
    main()
