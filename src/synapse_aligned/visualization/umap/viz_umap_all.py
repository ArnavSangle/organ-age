"""
viz_umap_all.py - Unified UMAP visualisation script for all embedding versions.

Produces UMAP scatter plots for three distinct embedding views:

  1. **v3 unimodal** - one plot per modality (RNA, X-ray, MRI) using the raw
     PCA ``emb_<modality>_*`` columns from ``v3_aligned_base.parquet``.
  2. **v3 multimodal** - all ``emb_*`` columns concatenated from the base table,
     giving a single mixed-modality plot.
  3. **v3.5 aligned multimodal** - all ``z_*`` contrastive-aligned columns from
     ``v3_aligned_contrastive.parquet``.

Each view produces two PNG files saved to the appropriate figures directory:
  - ``<prefix>_age.png``      - points coloured by chronological age (viridis).
  - ``<prefix>_modality.png`` - points coloured by modality (tab10).

Directory layout (relative to the project root, auto-detected via ``__file__``)
---------------------------------------------------------------------------------
  data/processed/aligned/        - input parquet files
  figures/organon_multimodal/umap/ - outputs for v3 unimodal & multimodal views
  figures/synapse_aligned/umap/   - outputs for v3.5 aligned view
"""

import pandas as pd
import numpy as np
import umap
import matplotlib.pyplot as plt
from pathlib import Path

# ============================================================
# Directory structure (correct for your project)
# ============================================================

# Script is located at:
# organ-age/src/synapse_aligned/visualization/umap/viz_umap_all.py
# So project root is 4 levels up.
ROOT = Path(__file__).resolve().parents[4]

DATA = ROOT / "data" / "processed" / "aligned"
FIG_ORG = ROOT / "figures" / "organon_multimodal" / "umap"
FIG_SYN = ROOT / "figures" / "synapse_aligned" / "umap"

# Ensure figure folders exist
FIG_ORG.mkdir(parents=True, exist_ok=True)
FIG_SYN.mkdir(parents=True, exist_ok=True)

print("[INFO] Unified UMAP script running from:", Path(__file__).resolve())
print("[INFO] Project root detected as:", ROOT)
print("[INFO] Reading data from:", DATA)
print("[INFO] Saving outputs to:")
print("       -", FIG_ORG)
print("       -", FIG_SYN)



# ============================================================
# Helper: run UMAP and save 2 plots
# ============================================================

def run_umap_and_save(X, df, out_prefix, out_dir):
    """
    Fit a 2-D UMAP on ``X`` and save age and modality scatter plots.

    Parameters
    ----------
    X : np.ndarray of shape (N, D)
        Embedding matrix to reduce.  Should already have NaN / inf values
        replaced (e.g. via ``np.nan_to_num``).
    df : pd.DataFrame
        Metadata dataframe aligned row-for-row with ``X``.  Must contain
        ``"age"`` (numeric) and ``"modality"`` (string: "rna", "xray", "mri")
        columns.
    out_prefix : str
        Short label used as the stem of the two output filenames, e.g.
        ``"unimodal_rna_v3"``.
    out_dir : pathlib.Path
        Directory in which to write the two PNG files.  Must already exist.

    Outputs
    -------
    Writes two files to ``out_dir``:
        ``<out_prefix>_age.png``      - scatter coloured by age (dpi=300).
        ``<out_prefix>_modality.png`` - scatter coloured by modality (dpi=300).
    """
    reducer = umap.UMAP(n_components=2, random_state=42)
    U = reducer.fit_transform(X)

    # ---- plot by age ----
    plt.figure(figsize=(6,6))
    sc = plt.scatter(U[:,0], U[:,1], s=3, c=df["age"], cmap="viridis")
    plt.colorbar(sc, label="Age")
    plt.title(f"{out_prefix} (Age)")
    plt.tight_layout()
    plt.savefig(out_dir / f"{out_prefix}_age.png", dpi=300)
    plt.close()

    # ---- plot by modality ----
    mod_map = {"rna":0, "xray":1, "mri":2}
    C = df["modality"].map(mod_map).fillna(-1).to_numpy()

    plt.figure(figsize=(6,6))
    sc2 = plt.scatter(U[:,0], U[:,1], s=3, c=C, cmap="tab10")
    plt.title(f"{out_prefix} (Modality)")
    plt.tight_layout()
    plt.savefig(out_dir / f"{out_prefix}_modality.png", dpi=300)
    plt.close()

    print(f"[UMAP] Saved {out_prefix} → {out_dir}")


# ============================================================
# MAIN
# ============================================================

def main():
    """
    Orchestrate all three UMAP visualisation passes (v3 unimodal, v3 multimodal,
    v3.5 contrastive-aligned) and write all output PNGs to their respective
    figure directories.
    """
    # --------------------------------------------------------
    # 1) v3 unimodal (emb_rna_*, emb_xray_*, emb_mri_*)
    # --------------------------------------------------------
    df_base = pd.read_parquet(DATA / "v3_aligned_base.parquet")

    for mod in ["rna", "xray", "mri"]:
        cols = [c for c in df_base.columns if c.startswith(f"emb_{mod}_")]
        df_mod = df_base[df_base["modality"] == mod]
        if len(df_mod) == 0 or len(cols) == 0:
            continue

        X = df_mod[cols].to_numpy(dtype="float32")
        X = np.nan_to_num(X)
        run_umap_and_save(X, df_mod, f"unimodal_{mod}_v3", FIG_ORG)

    # --------------------------------------------------------
    # 2) v3 multimodal (concatenate all emb_ blocks)
    # --------------------------------------------------------
    cols_multi = [c for c in df_base.columns if c.startswith("emb_")]
    X = df_base[cols_multi].to_numpy(dtype="float32")
    X = np.nan_to_num(X)
    run_umap_and_save(X, df_base, "multimodal_v3", FIG_ORG)

    # --------------------------------------------------------
    # 3) v3.5 aligned multimodal (z_* blocks)
    # --------------------------------------------------------
    df_con = pd.read_parquet(DATA / "v3_aligned_contrastive.parquet")
    cols_con = [c for c in df_con.columns if c.startswith("z_")]
    X = df_con[cols_con].to_numpy(dtype="float32")
    X = np.nan_to_num(X)
    run_umap_and_save(X, df_con, "multimodal_v3_5_aligned", FIG_SYN)

    print("[UMAP] All UMAP visualizations completed.")


if __name__ == "__main__":
    main()
