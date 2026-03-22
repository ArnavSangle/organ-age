"""Detailed column inspection for v3_contrastive."""
import pandas as pd, os, numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

df = pd.read_parquet(os.path.join(ROOT, "data/processed/aligned/v3_aligned_contrastive.parquet"))
meta = ['sample_id','subject_id','age','sex','organ','modality','source','features','embedding']
all_cols = [c for c in df.columns if c not in meta]

# group by prefix
from collections import Counter
prefixes = Counter(c.rsplit('_', 1)[0] for c in all_cols)
print("Column prefixes:", sorted(prefixes.items()))
print(f"Total emb cols: {len(all_cols)}")

# Check which cols have NaN per modality
for mod in ['rna', 'xray', 'mri']:
    sub = df[df['modality'] == mod]
    # Find a prefix group that is fully non-NaN for this modality
    for prefix, _ in sorted(prefixes.items()):
        pcols = [c for c in all_cols if c.startswith(prefix + '_') or c == prefix][:1]
        if pcols:
            nan_rate = sub[pcols[0]].isna().mean()
            if nan_rate < 0.01:
                print(f"  modality={mod}: prefix '{prefix}' → NaN rate {nan_rate:.3f}")
                break
