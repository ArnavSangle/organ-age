#!/usr/bin/env python
"""
72_2_train_multimodal_fusion_v2.py

Train a multimodal fusion age regressor on the aligned embedding
produced by 73_2_alignment_large.py (aligned_all.parquet).

- Input:  data/processed/aligned_all.parquet
          columns: [..., "age", "embedding" (list[float, OUT_DIM])]
- Model:  simple MLP regressor on the embedding
- Output:
    experiments/run_<run_id>/models/fusion_regressor_v2.pt
    experiments/run_<run_id>/fusion_curves_v2.pkl
"""

import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


# ------------------------- Model definition ------------------------- #

class FusionRegressor(nn.Module):
    """
    Two-hidden-layer MLP age regressor applied to aligned embeddings.

    Architecture:
      in_dim -> hidden_dim -> ReLU -> Dropout
             -> hidden_dim2 -> ReLU -> Dropout
             -> 1

    Designed to operate on the OUT_DIM-dimensional embeddings produced by
    train_alignment_large_v2.py.
    """

    def __init__(self, in_dim: int, hidden_dim: int = 256, hidden_dim2: int = 128, dropout: float = 0.1):
        """
        Args:
            in_dim:      Dimensionality of the aligned embedding input.
            hidden_dim:  Width of the first hidden layer (default 256).
            hidden_dim2: Width of the second hidden layer (default 128).
            dropout:     Dropout probability applied after each hidden layer.
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim2, 1),
        )

    def forward(self, x):
        """Return predicted scalar age for each sample in the batch."""
        return self.net(x).squeeze(-1)


# ------------------------- Helpers ------------------------- #

def build_parser() -> argparse.ArgumentParser:
    """Construct and return the CLI argument parser for this script."""
    p = argparse.ArgumentParser()
    p.add_argument("--run_id", type=str, default="v2",
                   help="Run identifier (for output dirs).")
    p.add_argument("--device", type=str, default="cuda",
                   help="Device: 'cuda' or 'cpu'.")
    p.add_argument("--batch_size", type=int, default=256,
                   help="Batch size for training.")
    p.add_argument("--epochs", type=int, default=40,
                   help="Number of training epochs.")
    p.add_argument("--lr", type=float, default=1e-3,
                   help="Learning rate.")
    p.add_argument("--weight_decay", type=float, default=1e-4,
                   help="Weight decay for Adam.")
    p.add_argument("--val_fraction", type=float, default=0.15,
                   help="Fraction of data to use for validation.")
    p.add_argument("--seed", type=int, default=42,
                   help="Random seed.")
    return p


def set_seed(seed: int):
    """Seed Python, NumPy, and PyTorch (CPU + all GPUs) for reproducibility."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def prepare_data(aligned_path: Path, val_fraction: float = 0.15):
    """
    Load aligned_all.parquet, extract the 'embedding' column as a dense
    float32 matrix, shuffle, and split into train / val arrays.

    Returns:
        (X_train, y_train): numpy arrays for training.
        (X_val,   y_val):   numpy arrays for validation.
        D: embedding dimensionality (int).

    Raises:
        FileNotFoundError: if *aligned_path* does not exist.
        ValueError:        if required columns are absent or embeddings differ
                           in length across rows.
    """
    if not aligned_path.exists():
        raise FileNotFoundError(
            f"Expected {aligned_path}. Run 73_2_alignment_large.py first."
        )

    print(f"[FUSION] Loading aligned data from {aligned_path}")
    df = pd.read_parquet(aligned_path)

    if "embedding" not in df.columns:
        raise ValueError("aligned_all.parquet must contain an 'embedding' column.")

    if "age" not in df.columns:
        raise ValueError("aligned_all.parquet must contain an 'age' column.")

    # Drop any rows with missing embedding or age
    df = df.dropna(subset=["embedding", "age"])
    emb_list = df["embedding"].tolist()
    ages = df["age"].to_numpy(dtype=np.float32)

    # Convert list-of-lists -> (N, D)
    try:
        X = np.stack(emb_list).astype("float32")
    except ValueError as e:
        raise ValueError(
            "Embeddings in 'embedding' column are not all the same length. "
            "Check output of 73_2_alignment_large.py."
        ) from e

    N, D = X.shape
    print(f"[FUSION] Using N={N}, embedding_dim={D}")

    # Shuffle once and split into train/val
    idx = np.arange(N)
    np.random.shuffle(idx)

    n_val = int(N * val_fraction)
    val_idx = idx[:n_val]
    train_idx = idx[n_val:]

    X_train, y_train = X[train_idx], ages[train_idx]
    X_val, y_val = X[val_idx], ages[val_idx]

    print(f"[FUSION] Train size = {X_train.shape[0]}, Val size = {X_val.shape[0]}")

    return (X_train, y_train), (X_val, y_val), D


def make_dataloaders(
    X_train, y_train, X_val, y_val, batch_size: int, device: torch.device
):
    """
    Wrap numpy arrays in TensorDatasets and return (train_loader, val_loader).

    Tensors are kept on CPU here; each training iteration moves them to
    *device* via ``.to(device)``.
    """
    # Move to device *inside* training loop via .to(device), so here we just keep CPU tensors
    train_ds = TensorDataset(
        torch.from_numpy(X_train),
        torch.from_numpy(y_train),
    )
    val_ds = TensorDataset(
        torch.from_numpy(X_val),
        torch.from_numpy(y_val),
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=False)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, drop_last=False)

    return train_loader, val_loader


def train_fusion(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    epochs: int,
    lr: float,
    weight_decay: float,
):
    """
    Run the training loop for *model* and return a curves dict.

    Args:
        model:        FusionRegressor instance (will be moved to *device*).
        train_loader: DataLoader for training split.
        val_loader:   DataLoader for validation split.
        device:       Torch device.
        epochs:       Number of training epochs.
        lr:           Adam learning rate.
        weight_decay: L2 regularisation coefficient for Adam.

    Returns:
        curves: dict with keys 'epoch', 'train_mse', 'val_mse', 'val_mae',
                each a list of length *epochs*.
    """
    model.to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    curves = {
        "epoch": [],
        "train_mse": [],
        "val_mse": [],
        "val_mae": [],
    }

    for epoch in range(1, epochs + 1):
        # --------------------- Train --------------------- #
        model.train()
        train_losses = []

        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad()
            preds = model(xb)
            loss = criterion(preds, yb)
            loss.backward()
            optimizer.step()

            train_losses.append(loss.detach().item())

        train_mse = float(np.mean(train_losses)) if train_losses else float("nan")

        # --------------------- Validation ---------------- #
        model.eval()
        val_losses = []
        val_mae_list = []

        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                yb = yb.to(device)

                preds = model(xb)
                mse = criterion(preds, yb).item()
                mae = torch.mean(torch.abs(preds - yb)).item()

                val_losses.append(mse)
                val_mae_list.append(mae)

        val_mse = float(np.mean(val_losses)) if val_losses else float("nan")
        val_mae = float(np.mean(val_mae_list)) if val_mae_list else float("nan")

        curves["epoch"].append(epoch)
        curves["train_mse"].append(train_mse)
        curves["val_mse"].append(val_mse)
        curves["val_mae"].append(val_mae)

        print(
            f"[FUSION] Epoch {epoch:02d} | "
            f"train MSE={train_mse:.4f} | val MSE={val_mse:.4f}, val MAE={val_mae:.4f}"
        )

    return curves


# ------------------------- Main ------------------------- #

def main():
    """
    Entry point: seed RNG, prepare data, build FusionRegressor, train, and
    save model weights plus training curves.
    """
    args = build_parser().parse_args()
    set_seed(args.seed)

    root = Path(__file__).resolve().parents[1]
    proc = root / "data" / "processed"
    aligned_path = proc / "aligned_all.parquet"

    out_root = root / "experiments" / f"run_{args.run_id}"
    models_dir = out_root / "models"
    out_root.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"[FUSION] Using device: {device}")

    # Prepare data
    (X_train, y_train), (X_val, y_val), in_dim = prepare_data(
        aligned_path, val_fraction=args.val_fraction
    )

    train_loader, val_loader = make_dataloaders(
        X_train, y_train, X_val, y_val, args.batch_size, device
    )

    # Build model
    model = FusionRegressor(in_dim=in_dim, hidden_dim=256, hidden_dim2=128, dropout=0.1)

    print("[FUSION] Starting training...")
    curves = train_fusion(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    # Save model + curves
    model_path = models_dir / "fusion_regressor_v2.pt"
    torch.save(model.state_dict(), model_path)
    print(f"[FUSION] Saved fusion regressor -> {model_path}")

    curves_path = out_root / "fusion_curves_v2.pkl"
    with open(curves_path, "wb") as f:
        pickle.dump(curves, f)
    print(f"[FUSION] Saved curves -> {curves_path}")


if __name__ == "__main__":
    main()
