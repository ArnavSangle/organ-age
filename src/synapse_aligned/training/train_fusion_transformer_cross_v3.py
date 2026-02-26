"""
train_fusion_transformer_cross_v3.py - Cross-fusion transformer training (v3.5).

Trains a ``CrossFusion`` transformer model that regresses biological age from
the contrastive-aligned multimodal embeddings (``z_rna_*``, ``z_xray_*``,
``z_mri_*``) produced by ``train_alignment_contrastive_v3.py``.

The model treats each sample's aligned 256-d vector as a single-token sequence
and passes it through a 4-layer Transformer Encoder, then a two-layer MLP
regression head that outputs a scalar age prediction.

Workflow
--------
1. Load ``v3_aligned_contrastive.parquet``.
2. Build and clean per-modality arrays; concatenate into a single dataset.
3. Split 90 / 10 into train and validation sets.
4. Train ``CrossFusion`` with AdamW + gradient clipping; track validation MSE.
5. Save the best-validation-MSE checkpoint to ``v3_output/models/``.

CLI arguments
-------------
--run_id      : Tag appended to the saved model filename (default "v3_cross").
--epochs      : Number of training epochs (default 40).
--batch_size  : Mini-batch size (default 256).
--device      : "cuda", "cpu", or "auto" (default "auto").
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


OUT = Path("v3_output")
DATA = OUT / "data"
MODELS = OUT / "models"


class CrossFusion(nn.Module):
    def __init__(self, D: int):
        """
        Simple transformer-based fusion over a single-token sequence.

        Input:  x of shape (B, D)
        We treat it as (B, 1, D) for the encoder.
        """
        super().__init__()
        layer = nn.TransformerEncoderLayer(
            d_model=D,
            nhead=8,
            dim_feedforward=512,
            activation="gelu",
            batch_first=True,  # (B, T, D)
        )
        self.attn = nn.TransformerEncoder(layer, num_layers=4)
        self.head = nn.Sequential(
            nn.Linear(D, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Run the transformer encoder and regression head.

        Parameters
        ----------
        x : torch.Tensor
            Aligned embedding tensor of shape ``(B, D)``.

        Returns
        -------
        torch.Tensor
            Scalar age predictions of shape ``(B,)``.
        """
        # x: (B, D) -> (B, 1, D)
        x = x.unsqueeze(1)
        h = self.attn(x).squeeze(1)  # (B, D)
        return self.head(h).squeeze(1)  # (B,)


def _filter_block(df_mod: pd.DataFrame, cols: list[str], name: str):
    """
    From a modality-specific dataframe, extract X (z_* block) and y (age),
    and drop any rows with non-finite values.
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
        print(f"[V3 CROSS FUSION] {name}: dropping {n_drop}/{n_total} rows with non-finite X/y")

    X = X[mask]
    y = y[mask]

    return X, y


def build_dataset(df: pd.DataFrame):
    """
    Build (X_train, y_train, X_val, y_val) from v3_aligned_contrastive.parquet
    using the aligned embeddings:

        rna rows -> z_rna_*
        xray rows -> z_xray_*
        mri rows -> z_mri_*
    """
    print("[V3 CROSS FUSION] df shape:", df.shape)
    print("[V3 CROSS FUSION] modalities:\n", df["modality"].value_counts())

    cols_r = [c for c in df.columns if c.startswith("z_rna_")]
    cols_x = [c for c in df.columns if c.startswith("z_xray_")]
    cols_m = [c for c in df.columns if c.startswith("z_mri_")]

    if not (cols_r and cols_x and cols_m):
        raise RuntimeError("[V3 CROSS FUSION] Missing some z_* columns in aligned contrastive table.")

    # Filter per modality
    df_r = df[df["modality"] == "rna"].copy()
    df_x = df[df["modality"] == "xray"].copy()
    df_m = df[df["modality"] == "mri"].copy()

    Xr, yr = _filter_block(df_r, cols_r, "rna")
    Xx, yx = _filter_block(df_x, cols_x, "xray")
    Xm, ym = _filter_block(df_m, cols_m, "mri")

    # Concatenate all modalities vertically -> (N_total, D)
    X = np.concatenate([Xr, Xx, Xm], axis=0)
    y = np.concatenate([yr, yx, ym], axis=0)

    print("[V3 CROSS FUSION] After cleaning:")
    print("  X shape:", X.shape)
    print("  y shape:", y.shape)

    # Final global non-finite check (paranoia)
    mask_X = np.isfinite(X).all(axis=1)
    mask_y = np.isfinite(y)
    mask = mask_X & mask_y
    if mask.sum() != X.shape[0]:
        n_drop = X.shape[0] - int(mask.sum())
        print(f"[V3 CROSS FUSION] Global: dropping {n_drop} rows with non-finite values")
        X = X[mask]
        y = y[mask]

    # Shuffle once and create train/val split (e.g., 90/10)
    N = X.shape[0]
    idx = np.random.permutation(N)
    X = X[idx]
    y = y[idx]

    split = int(0.9 * N)
    X_train, X_val = X[:split], X[split:]
    y_train, y_val = y[:split], y[split:]

    print(f"[V3 CROSS FUSION] Train size = {X_train.shape[0]}, Val size = {X_val.shape[0]}")

    return X_train, y_train, X_val, y_val


def main():
    """
    Parse CLI arguments, build the dataset, train CrossFusion, and save the
    best model checkpoint.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_id", type=str, default="v3_cross")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--device", type=str, default="auto")  # "cuda", "cpu", or "auto"
    args = parser.parse_args()

    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    print("[V3 CROSS FUSION] Using device:", device)

    # Load aligned contrastive embeddings
    df = pd.read_parquet(DATA / "v3_aligned_contrastive.parquet")

    # Build dataset (with cleaning)
    X_train, y_train, X_val, y_val = build_dataset(df)

    # Convert to tensors
    X_train = torch.tensor(X_train, dtype=torch.float32)
    y_train = torch.tensor(y_train, dtype=torch.float32)
    X_val = torch.tensor(X_val, dtype=torch.float32)
    y_val = torch.tensor(y_val, dtype=torch.float32)

    train_ds = TensorDataset(X_train, y_train)
    val_ds = TensorDataset(X_val, y_val)

    train_dl = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=2048, shuffle=False)

    D = X_train.shape[1]  # should be 256
    model = CrossFusion(D=D).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-4)
    loss_fn = nn.MSELoss()

    print("[V3 CROSS FUSION] Training...")
    best = float("inf")

    for epoch in range(1, args.epochs + 1):
        # ---- Train ----
        model.train()
        train_losses = []
        for xb, yb in train_dl:
            xb, yb = xb.to(device), yb.to(device)

            # Belt-and-suspenders: clean any stray NaNs in batch
            xb = torch.nan_to_num(xb, nan=0.0, posinf=0.0, neginf=0.0)
            yb = torch.nan_to_num(yb, nan=0.0, posinf=0.0, neginf=0.0)

            pred = model(xb)
            loss = loss_fn(pred, yb)

            if not torch.isfinite(loss):
                print("[V3 CROSS FUSION] Non-finite loss detected, stopping training.")
                print("  batch loss:", loss)
                return

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()
            train_losses.append(loss.item())

        train_mse = float(np.mean(train_losses))

        # ---- Val ----
        model.eval()
        with torch.no_grad():
            vloss = []
            for xb, yb in val_dl:
                xb, yb = xb.to(device), yb.to(device)
                xb = torch.nan_to_num(xb, nan=0.0, posinf=0.0, neginf=0.0)
                yb = torch.nan_to_num(yb, nan=0.0, posinf=0.0, neginf=0.0)
                vloss.append(loss_fn(model(xb), yb).item())
            val_mse = float(np.mean(vloss))

        print(f"[epoch {epoch:02d}] train MSE={train_mse:.2f}  val MSE={val_mse:.2f}")

        # Save best model
        if val_mse < best and np.isfinite(val_mse):
            best = val_mse
            MODELS.mkdir(parents=True, exist_ok=True)
            out_path = MODELS / f"fusion_cross_{args.run_id}.pt"
            torch.save(model.state_dict(), out_path)
            print(f"[V3 CROSS FUSION] Saved BEST model (val MSE={best:.2f}) -> {out_path}")


if __name__ == "__main__":
    main()
