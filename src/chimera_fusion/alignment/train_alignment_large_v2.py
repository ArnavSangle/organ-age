#!/usr/bin/env python
"""
73_2_alignment_large.py

Goal:
- Build a shared low-dimensional "aligned" embedding for all modalities
  (RNA, X-ray, MRI) using the unified_all.parquet table.
- Uses a random projection from truncated features (first D_TRUNC dims)
  into a shared latent space (OUT_DIM).
- Saves:
    data/processed/aligned_all.parquet
    data/processed/aligned_rna.parquet
    data/processed/aligned_chexpert.parquet
    data/processed/aligned_ixi.parquet

This script is intentionally *robust*:
- It does NOT depend on any pre-existing aligned_*.parquet files.
- It ignores --epochs and --batch_size arguments (kept only for CLI compatibility).
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def build_parser() -> argparse.ArgumentParser:
    """Construct and return the CLI argument parser for this script."""
    p = argparse.ArgumentParser()
    p.add_argument("--run_id", type=str, default="v2",
                   help="Run identifier (kept for bookkeeping, not strictly required).")
    p.add_argument("--epochs", type=int, default=60,
                   help="Unused (kept only for CLI compatibility).")
    p.add_argument("--batch_size", type=int, default=2048,
                   help="Streaming batch size for projection.")
    p.add_argument("--seed", type=int, default=42,
                   help="Random seed for projection matrix.")
    return p


def main():
    """
    Load unified_all.parquet, compute a random-projection alignment of all
    feature vectors into an OUT_DIM-dimensional shared space, attach the
    resulting 'embedding' column, and save the global and per-modality
    aligned parquet files under ``data/processed/``.
    """
    args = build_parser().parse_args()

    root = Path(__file__).resolve().parents[1]
    proc = root / "data" / "processed"
    proc.mkdir(parents=True, exist_ok=True)

    unified_path = proc / "unified_all.parquet"
    if not unified_path.exists():
        raise FileNotFoundError(
            f"Expected {unified_path} from script 70_build_unified_dataset.py, "
            f"but it does not exist."
        )

    print(f"[ALIGN] Run ID: {args.run_id}")
    print(f"[ALIGN] Loading {unified_path} ...")
    df = pd.read_parquet(unified_path)

    if "features" not in df.columns:
        raise ValueError("unified_all.parquet must contain a 'features' column of lists.")

    feats = df["features"].tolist()
    N = len(feats)
    print(f"[ALIGN] Loaded unified_all with N={N} rows.")

    # ---- Truncate / pad features to a manageable fixed dimension -----------------
    # We do NOT use the full ~59k-dim RNA features here for alignment, for
    # performance reasons. Instead, we truncate every feature vector to
    # the first D_TRUNC entries, padding with zeros if shorter.
    D_TRUNC = 512       # effective input dimensionality for alignment
    OUT_DIM = 64        # aligned latent dimensionality
    BATCH = args.batch_size

    print(f"[ALIGN] Using D_TRUNC={D_TRUNC}, OUT_DIM={OUT_DIM}, batch_size={BATCH}")

    # Build random projection matrix (Johnson–Lindenstrauss-style)
    rng = np.random.default_rng(args.seed)
    proj = rng.normal(
        loc=0.0,
        scale=1.0 / np.sqrt(D_TRUNC),
        size=(D_TRUNC, OUT_DIM),
    ).astype("float32")

    # Prepare output storage
    aligned = np.zeros((N, OUT_DIM), dtype="float32")

    def to_trunc_vec(v: list[float]) -> np.ndarray:
        """Convert a variable-length feature list -> fixed D_TRUNC vector."""
        arr = np.asarray(v, dtype="float32")
        if arr.shape[0] >= D_TRUNC:
            return arr[:D_TRUNC]
        out = np.zeros(D_TRUNC, dtype="float32")
        out[: arr.shape[0]] = arr
        return out

    print("[ALIGN] Computing aligned embeddings via streaming random projection...")
    start_idx = 0
    while start_idx < N:
        end_idx = min(start_idx + BATCH, N)
        batch_feats = feats[start_idx:end_idx]

        # Build a (batch_size, D_TRUNC) block
        X_block = np.zeros((end_idx - start_idx, D_TRUNC), dtype="float32")
        for i, feat in enumerate(batch_feats):
            X_block[i] = to_trunc_vec(feat)

        # Project: (B, D_TRUNC) @ (D_TRUNC, OUT_DIM) -> (B, OUT_DIM)
        Z_block = X_block @ proj
        aligned[start_idx:end_idx] = Z_block

        if (start_idx // BATCH) % 10 == 0:
            print(f"  [ALIGN] Processed {end_idx}/{N} rows...")

        start_idx = end_idx

    print("[ALIGN] Finished computing aligned embeddings.")
    # Attach as a new column of lists
    df["embedding"] = [aligned[i].tolist() for i in range(N)]

    # Save global aligned table
    aligned_all_path = proc / "aligned_all.parquet"
    print(f"[ALIGN] Saving global aligned table -> {aligned_all_path}")
    df.to_parquet(aligned_all_path, index=False)

    # Also save per-modality splits for convenience
    if "modality" not in df.columns:
        print("[ALIGN] WARNING: 'modality' column not found; "
              "cannot write per-modality aligned files.")
        return

    mod_map = {
        "rna": "aligned_rna.parquet",
        "xray": "aligned_chexpert.parquet",
        "mri": "aligned_ixi.parquet",
    }

    for mod, fname in mod_map.items():
        sub = df[df["modality"] == mod].copy()
        if sub.empty:
            print(f"[ALIGN] WARNING: No rows with modality='{mod}', skipping {fname}.")
            continue
        out_path = proc / fname
        print(f"[ALIGN] Saving {mod} aligned table -> {out_path} "
              f"(N={len(sub)})")
        sub.to_parquet(out_path, index=False)

    print("[ALIGN] Done. You now have:")
    print(f"  - {aligned_all_path}")
    for mod, fname in mod_map.items():
        print(f"  - {proc / fname}")


if __name__ == "__main__":
    main()
