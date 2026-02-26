"""
eval_fusion_transformer_cross_v3.py - Evaluation script for the CrossFusion model (v3.5).

Loads a trained ``CrossFusion`` transformer checkpoint and evaluates it on the
full contrastive-aligned dataset, reporting MSE and MAE both overall and
broken down by modality (RNA, X-ray, MRI).

The CrossFusion class is imported directly from the sibling training script
(``train_fusion_transformer_cross_v3.py``) by inserting the training directory
into ``sys.path`` at runtime.

Workflow
--------
1. Add the training directory to ``sys.path`` so ``CrossFusion`` can be imported.
2. Load ``v3_aligned_contrastive.parquet`` from ``data/processed/aligned/``.
3. Build per-modality and combined arrays, clean non-finite values.
4. Load the saved model weights and run batched inference.
5. Print MSE and MAE for ALL samples, and separately for each modality.

CLI arguments
-------------
--run_id      : Run tag used to locate the model file (default "v3_cross").
--device      : "cuda", "cpu", or "auto" (default "auto").
--batch_size  : Inference batch size (default 4096).
"""

import argparse
from pathlib import Path
import sys

# --- Make sure the training folder (with fusion_transformer_cross_v3.py) is on sys.path ---
THIS_DIR = Path(__file__).resolve().parent          # .../src/synapse_aligned/evaluation
TRAINING_DIR = THIS_DIR.parent / "training"         # .../src/synapse_aligned/training

if str(TRAINING_DIR) not in sys.path:
    sys.path.append(str(TRAINING_DIR))

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

# import the CrossFusion model from the training folder
from train_fusion_transformer_cross_v3 import CrossFusion

# ---- Paths ----
DATA = Path("data/processed/aligned")

MODELS_ROOT = Path("models/synapse_aligned")
FUSION_DIR = MODELS_ROOT / "fusion_cross_v3_5"
MODEL_DIR = FUSION_DIR / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)



def _filter_block(df_mod: pd.DataFrame, cols, name: str):
    """
    Extract the embedding matrix and age labels from a modality subset,
    dropping any rows that contain non-finite values in X or y.

    Parameters
    ----------
    df_mod : pd.DataFrame
        Rows belonging to a single modality.
    cols : list[str]
        Column names of the aligned embedding block (e.g. ``z_rna_*``).
    name : str
        Human-readable modality label used only for diagnostic print output.

    Returns
    -------
    X : np.ndarray of shape (N_clean, len(cols)), dtype float32
    y : np.ndarray of shape (N_clean,), dtype float32
    """
    X = df_mod[cols].to_numpy(dtype="float32")
    y = df_mod["age"].to_numpy(dtype="float32")

    mask_X = np.isfinite(X).all(axis=1)
    mask_y = np.isfinite(y)
    mask = mask_X & mask_y

    n_total = X.shape[0]
    n_keep = int(mask.sum())
    n_drop = n_total - n_keep
    if n_drop > 0:
        print(f"[V3 CROSS EVAL] {name}: dropping {n_drop}/{n_total} rows with non-finite X/y")

    X = X[mask]
    y = y[mask]
    return X, y


def build_full_dataset(df: pd.DataFrame):
    """
    Build cleaned per-modality arrays and a combined array from the aligned
    contrastive parquet table.

    Parameters
    ----------
    df : pd.DataFrame
        Full aligned contrastive table containing ``z_rna_*``, ``z_xray_*``,
        and ``z_mri_*`` columns plus an ``age`` column and a ``modality``
        column.

    Returns
    -------
    tuple of four (X, y) pairs:
        ``(Xr, yr)`` - RNA block (N_rna, D), age array
        ``(Xx, yx)`` - X-ray block (N_xray, D), age array
        ``(Xm, ym)`` - MRI block (N_mri, D), age array
        ``(X_all, y_all)`` - vertically concatenated arrays for all modalities

    Raises
    ------
    RuntimeError
        If any of the ``z_*`` column groups are missing from the dataframe.
    """
    print("[V3 CROSS EVAL] df shape:", df.shape)
    print("[V3 CROSS EVAL] modalities:\n", df["modality"].value_counts())

    cols_r = [c for c in df.columns if c.startswith("z_rna_")]
    cols_x = [c for c in df.columns if c.startswith("z_xray_")]
    cols_m = [c for c in df.columns if c.startswith("z_mri_")]

    if not (cols_r and cols_x and cols_m):
        raise RuntimeError("[V3 CROSS EVAL] Missing some z_* columns in aligned contrastive table.")

    df_r = df[df["modality"] == "rna"].copy()
    df_x = df[df["modality"] == "xray"].copy()
    df_m = df[df["modality"] == "mri"].copy()

    Xr, yr = _filter_block(df_r, cols_r, "rna")
    Xx, yx = _filter_block(df_x, cols_x, "xray")
    Xm, ym = _filter_block(df_m, cols_m, "mri")

    X = np.concatenate([Xr, Xx, Xm], axis=0)
    y = np.concatenate([yr, yx, ym], axis=0)

    print("[V3 CROSS EVAL] After cleaning:")
    print("  X shape:", X.shape)
    print("  y shape:", y.shape)

    return (Xr, yr), (Xx, yx), (Xm, ym), (X, y)


def evaluate_block(model, device, X, y, name: str, batch_size: int = 4096):
    """Evaluate in batches to avoid giant single forward passes on GPU."""
    ds = TensorDataset(
        torch.tensor(X, dtype=torch.float32),
        torch.tensor(y, dtype=torch.float32),
    )
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False)

    sq_errs = []
    abs_errs = []

    model.eval()
    with torch.no_grad():
        for xb, yb in dl:
            xb, yb = xb.to(device), yb.to(device)
            xb = torch.nan_to_num(xb, nan=0.0, posinf=0.0, neginf=0.0)
            yb = torch.nan_to_num(yb, nan=0.0, posinf=0.0, neginf=0.0)

            pred = model(xb)
            diff = pred - yb
            sq_errs.append((diff ** 2).detach().cpu())
            abs_errs.append(diff.abs().detach().cpu())

    sq_errs = torch.cat(sq_errs)
    abs_errs = torch.cat(abs_errs)

    mse = float(sq_errs.mean().item())
    mae = float(abs_errs.mean().item())

    print(f"{name:<4} | N={len(y):7d} | MSE={mse:8.3f} | MAE={mae:6.3f}")
    return mse, mae


def main():
    """
    Parse CLI arguments, load the CrossFusion model, and print evaluation
    metrics (MSE, MAE) for the full dataset and each individual modality.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_id", type=str, default="v3_cross")
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--batch_size", type=int, default=4096)
    args = parser.parse_args()

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    print("[V3 CROSS EVAL] Using device:", device)

    # v3.5 contrastive-aligned table
    df = pd.read_parquet(DATA / "v3_aligned_contrastive.parquet")

    (Xr, yr), (Xx, yx), (Xm, ym), (X_all, y_all) = build_full_dataset(df)

    D = X_all.shape[1]  # should be 256
    model = CrossFusion(D=D).to(device)

    model_path = MODEL_DIR / f"fusion_cross_{args.run_id}.pt"
    print("[V3 CROSS EVAL] Loading model:", model_path)
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)

    print("\n=== Overall Cross-Fusion Metrics (v3.5 contrastive) ===")
    evaluate_block(model, device, X_all, y_all, "ALL", batch_size=args.batch_size)

    print("\n=== Per-modality Metrics (v3.5 contrastive) ===")
    evaluate_block(model, device, Xr, yr, "rna", batch_size=args.batch_size)
    evaluate_block(model, device, Xx, yx, "xray", batch_size=args.batch_size)
    evaluate_block(model, device, Xm, ym, "mri", batch_size=args.batch_size)


if __name__ == "__main__":
    main()
