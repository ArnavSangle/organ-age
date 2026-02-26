"""
train_alignment_ultra_v3.py - Supervised alignment projector training (v3).

Trains a single shared Projector (MLP) across all three modalities (RNA,
X-ray, MRI) using a supervised MSE objective: the projector is asked to
produce an embedding whose mean over the feature dimension approximates the
sample's chronological age.

This acts as a lightweight sanity-check / warm-up alignment step, ensuring
that the shared embedding space at least encodes age-relevant structure before
the contrastive alignment stage.

Workflow
--------
1. Load ``v3_aligned_base.parquet`` and extract per-modality PCA embeddings.
2. Concatenate all modalities into a single matrix; shuffle rows.
3. Train a two-layer MLP Projector (Linear -> ReLU -> Linear) that maps each
   64-d embedding to another 64-d representation, supervised by the MSE between
   the per-dimension mean of the output and the true age.
4. Save the trained projector weights to ``v3_output/models/alignment_v3.pt``.

Note: the batch size is 4096 and gradient clipping is applied at norm 5.0.
"""

import pandas as pd
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path

OUT = Path("v3_output")
DATA = OUT / "data"


def get_matrix(df: pd.DataFrame, prefix: str):
    """
    Extract embedding matrix and ages for a given prefix.
    Returns (X, ages) as float32 numpy arrays.
    If no columns match the prefix, returns empty arrays.
    """
    cols = [c for c in df.columns if c.startswith(prefix)]
    if not cols:
        print(f"[V3 ALIGN] WARNING: no columns found for prefix '{prefix}'")
        return (
            np.empty((0, 0), dtype="float32"),
            np.empty((0,), dtype="float32"),
        )

    X = df[cols].to_numpy(dtype="float32")
    ages = df["age"].to_numpy(dtype="float32")

    # Drop rows with NaN ages
    mask = ~np.isnan(ages)
    X = X[mask]
    ages = ages[mask]

    # Replace any NaN/inf in embeddings
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    return X, ages


def main():
    """
    Entry point for supervised alignment projector training.

    Loads base embeddings from all three modalities, concatenates them, trains
    an MLP Projector using MSE against chronological age, and saves the final
    weights to disk.
    """
    OUT.mkdir(exist_ok=True)
    (OUT / "models").mkdir(parents=True, exist_ok=True)

    print("[V3 ALIGN] Loading base embeddings...")
    df = pd.read_parquet(DATA / "v3_aligned_base.parquet")
    print(f"[V3 ALIGN] Loaded base table: shape={df.shape}")

    # --- Extract per-modality embeddings ---
    Xr, Ar = get_matrix(df, "emb_rna_")
    Xx, Ax = get_matrix(df, "emb_xray_")
    Xm, Am = get_matrix(df, "emb_mri_")

    matrices = []
    ages_list = []

    for name, Xmod, Amod in [
        ("rna", Xr, Ar),
        ("xray", Xx, Ax),
        ("mri", Xm, Am),
    ]:
        if Xmod.size == 0:
            print(f"[V3 ALIGN] WARNING: modality '{name}' has 0 samples.")
            continue
        # Sanity: embeddings should be 2D
        if Xmod.ndim != 2:
            raise ValueError(f"[V3 ALIGN] {name} embeddings are not 2D: {Xmod.shape}")
        matrices.append(Xmod)
        ages_list.append(Amod)
        print(f"[V3 ALIGN] {name}: X shape={Xmod.shape}, ages shape={Amod.shape}")

    if not matrices:
        raise RuntimeError("[V3 ALIGN] No modality embeddings found; nothing to align.")

    # --- Concatenate all modalities ---
    X = np.concatenate(matrices, axis=0)
    ages = np.concatenate(ages_list, axis=0)
    print(f"[V3 ALIGN] Concatenated X shape={X.shape}, ages shape={ages.shape}")

    # One more global clean, just in case
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    ages = np.nan_to_num(ages, nan=0.0, posinf=0.0, neginf=0.0)

    # Optional: shuffle
    idx = np.arange(len(ages))
    np.random.shuffle(idx)
    X = X[idx]
    ages = ages[idx]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[V3 ALIGN] Using device: {device}")

    # --- Convert to tensors ---
    X_t = torch.from_numpy(X).float()
    y_t = torch.from_numpy(ages).float()

    # Sanity check on feature dim (expect 64)
    feat_dim = X_t.shape[1]
    print(f"[V3 ALIGN] Feature dim: {feat_dim}")
    if feat_dim != 64:
        print(f"[V3 ALIGN] WARNING: expected 64-dim embeddings, got {feat_dim}")

    ds = TensorDataset(X_t, y_t)
    dl = DataLoader(ds, batch_size=4096, shuffle=True, drop_last=False)

    # --- Projector model ---
    class Projector(nn.Module):
        """
        Lightweight two-layer MLP that projects embeddings into a shared space.

        Used here with a supervised MSE objective: the per-dimension mean of
        the output vector is treated as a scalar age proxy.
        """

        def __init__(self, in_dim=64, hidden=128, out_dim=64):
            """
            Initialise the projector.

            Parameters
            ----------
            in_dim : int
                Input feature dimension (default 64 for PCA-reduced embeddings).
            hidden : int
                Width of the hidden layer (default 128).
            out_dim : int
                Output embedding dimension (default 64).
            """
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(in_dim, hidden),
                nn.ReLU(),
                nn.Linear(hidden, out_dim),
            )

        def forward(self, x):
            """
            Forward pass through the MLP.

            Parameters
            ----------
            x : torch.Tensor
                Input tensor of shape ``(B, in_dim)``.

            Returns
            -------
            torch.Tensor
                Output tensor of shape ``(B, out_dim)``.
            """
            return self.net(x)

    P = Projector(in_dim=feat_dim, hidden=128, out_dim=64).to(device)
    opt = torch.optim.AdamW(P.parameters(), lr=1e-4, weight_decay=1e-4)
    mse = nn.MSELoss()

    print("[V3 ALIGN] Training projector...")
    num_epochs = 200
    for epoch in range(num_epochs):
        P.train()
        losses = []
        for xb, yb in dl:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)

            opt.zero_grad()

            pred = P(xb)
            # Collapse embedding to scalar age proxy via mean over dims
            pred_scalar = pred.mean(dim=1)

            # Clean any NaN/inf before loss
            pred_scalar = torch.nan_to_num(pred_scalar, nan=0.0, posinf=0.0, neginf=0.0)
            yb = torch.nan_to_num(yb, nan=0.0, posinf=0.0, neginf=0.0)

            loss = mse(pred_scalar, yb)

            # Guard against NaN/inf loss
            if torch.isnan(loss) or torch.isinf(loss):
                print("[V3 ALIGN] Warning: NaN/inf loss detected; skipping batch.")
                continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(P.parameters(), max_norm=5.0)
            opt.step()

            losses.append(loss.item())

        if len(losses) == 0:
            print(f"[V3 ALIGN] [epoch {epoch+1:02d}] all batches skipped (NaN/inf).")
        else:
            avg_loss = float(np.mean(losses))
            print(f"[V3 ALIGN] [epoch {epoch+1:02d}] loss={avg_loss:.6f}")

    out_path = OUT / "models" / "alignment_v3.pt"
    torch.save(P.state_dict(), out_path)
    print("[V3 ALIGN] Saved:", out_path)


if __name__ == "__main__":
    main()
