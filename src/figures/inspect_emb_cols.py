"""Check embedding column layout in v3_base and v3_contrastive."""
import pandas as pd, os

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

for name, path in [
    ("v3_base",        "data/processed/aligned/v3_aligned_base.parquet"),
    ("v3_contrastive", "data/processed/aligned/v3_aligned_contrastive.parquet"),
]:
    df = pd.read_parquet(os.path.join(ROOT, path), columns=None)
    meta = ['sample_id', 'subject_id', 'age', 'sex', 'organ', 'modality', 'source', 'features', 'embedding']
    emb_cols = [c for c in df.columns if c not in meta]
    print(f"{name}: {len(emb_cols)} embedding cols, first={emb_cols[:4]}, last={emb_cols[-4:]}")
    # Check for NaN in first emb col
    import numpy as np
    sample = df[emb_cols[0]]
    print(f"  dtype={sample.dtype}, NaN={sample.isna().sum()}, range=[{sample.min():.3f},{sample.max():.3f}]")
