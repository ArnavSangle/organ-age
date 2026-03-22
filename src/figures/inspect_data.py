"""Quick data inspection for 3D figure generation."""
import pandas as pd
import numpy as np
import os

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

def check(label, path):
    full = os.path.join(ROOT, path)
    if not os.path.exists(full):
        print(f"  MISSING: {path}")
        return
    if path.endswith(".parquet"):
        df = pd.read_parquet(full)
        print(f"  {label}: shape={df.shape}, cols={list(df.columns)[:12]}")
        if 'modality' in df.columns:
            print(f"    modality counts: {df['modality'].value_counts().to_dict()}")
        if 'age' in df.columns:
            print(f"    age range: {df['age'].min():.1f} – {df['age'].max():.1f}")
        # Check for embedding columns (list/array type)
        for col in df.columns:
            if df[col].dtype == object:
                try:
                    sample = df[col].iloc[0]
                    if hasattr(sample, '__len__') and not isinstance(sample, str):
                        print(f"    embedding col '{col}': len={len(sample)}")
                        break
                except Exception:
                    pass
    elif path.endswith(".npy"):
        arr = np.load(full, allow_pickle=False)
        print(f"  {label}: shape={arr.shape}, dtype={arr.dtype}, range=[{arr.min():.3f}, {arr.max():.3f}]")
    elif path.endswith(".csv"):
        df = pd.read_csv(full)
        print(f"  {label}: shape={df.shape}, cols={list(df.columns)[:12]}")

print("=== Aligned parquets ===")
check("aligned_all",         "data/processed/aligned/aligned_all.parquet")
check("aligned_rna",         "data/processed/aligned/aligned_rna.parquet")
check("aligned_chexpert",    "data/processed/aligned/aligned_chexpert.parquet")
check("aligned_ixi",         "data/processed/aligned/aligned_ixi.parquet")
check("v3_base",             "data/processed/aligned/v3_aligned_base.parquet")
check("v3_contrastive",      "data/processed/aligned/v3_aligned_contrastive.parquet")

print("\n=== Embeddings ===")
check("chexpert_embeddings",  "data/interim/chexpert/embeddings_compressed/chexpert_embeddings.npy")

print("\n=== Latent IG ===")
for organ in ["brain_cortex", "liver", "lung"]:
    check(f"latent_ig_{organ}", f"analysis/v4_5_ig_latent/latent_ig_{organ}.npy")

print("\n=== Gene importance ===")
for f in os.listdir(os.path.join(ROOT, "analysis/v4_5_gene_importance")):
    check(f, f"analysis/v4_5_gene_importance/{f}")

print("\n=== IXI NIfTI (first 3) ===")
ixi_dir = os.path.join(ROOT, "data/raw/ixi/T1")
if os.path.exists(ixi_dir):
    files = sorted(os.listdir(ixi_dir))[:3]
    for f in files:
        size = os.path.getsize(os.path.join(ixi_dir, f)) / 1e6
        print(f"  {f}: {size:.1f} MB")
