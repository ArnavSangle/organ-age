"""
71_2_train_modality_encoders.py

Unsupervised autoencoders for each modality (RNA / X-ray / MRI).
- Loads unified parquet files
- Trains a small encoder+decoder per modality with MSE reconstruction loss
- Applies robust per-sample normalization to avoid NaNs / scale issues
- Saves encoder weights to models/run_<run_id>/encoder_*.pt
"""

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


# ----------------------------
# Utilities
# ----------------------------

def set_seed(seed: int = 42):
    """Set random seeds for NumPy and PyTorch (both CPU and all GPUs)."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dir(path: Path):
    """Create *path* and all intermediate parents if they do not already exist."""
    path.mkdir(parents=True, exist_ok=True)


def normalize_batch(x: torch.Tensor) -> torch.Tensor:
    """
    Make a batch numerically safe and roughly standardized per-sample.
    x: (batch, feat_dim)
    """
    # Replace NaN / Inf
    x = torch.nan_to_num(x, nan=0.0, posinf=1e3, neginf=-1e3)

    # Per-sample mean/std
    mean = x.mean(dim=1, keepdim=True)
    std = x.std(dim=1, keepdim=True)

    x = (x - mean) / (std + 1e-6)

    # Clamp extremes just in case
    x = torch.clamp(x, -10.0, 10.0)
    return x


# ----------------------------
# Dataset
# ----------------------------

class FeatureDataset(Dataset):
    """
    Minimal PyTorch Dataset that wraps a 2-D float32 feature matrix.

    Each item is a single row tensor of shape (D,), returned as a CPU tensor
    ready for batching by DataLoader.
    """

    def __init__(self, features: np.ndarray):
        """
        Args:
            features: (N, D) float32 numpy array of pre-computed features.
        """
        assert features.ndim == 2
        self.x = features.astype("float32")

    def __len__(self):
        """Return the number of samples."""
        return self.x.shape[0]

    def __getitem__(self, idx):
        """Return the feature vector at *idx* as a float32 tensor."""
        return torch.from_numpy(self.x[idx])


# ----------------------------
# Model
# ----------------------------

class AutoEncoder(nn.Module):
    """
    Symmetric autoencoder for unsupervised feature compression.

    Encoder: input_dim -> hidden_dim -> latent_dim
    Decoder: latent_dim -> hidden_dim -> input_dim

    Trained with MSE reconstruction loss on per-sample normalised inputs.
    Only the encoder weights are persisted after training (the decoder is
    discarded) so the encoder can be reused for downstream fusion/alignment.
    """

    def __init__(self, input_dim: int, latent_dim: int = 256, hidden_dim: int = 512):
        """
        Args:
            input_dim:  Dimensionality of the raw input features.
            latent_dim: Size of the encoder bottleneck (default 256).
            hidden_dim: Width of the single hidden layer in each sub-network
                        (default 512).
        """
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, input_dim),
        )

    def forward(self, x):
        """Encode *x* to a latent vector, then decode back to input space."""
        z = self.encoder(x)
        recon = self.decoder(z)
        return recon


# ----------------------------
# Training logic
# ----------------------------

def load_features(parquet_path: Path) -> np.ndarray:
    """
    Read *parquet_path* and return the 'features' column as a dense (N, D)
    float32 numpy array.
    """
    df = pd.read_parquet(parquet_path)
    # "features" column is list-like; convert to dense array
    feats = np.asarray(df["features"].tolist(), dtype="float32")
    return feats


def train_autoencoder(
    name: str,
    parquet_path: Path,
    out_dir: Path,
    device: torch.device,
    epochs: int,
    batch_size: int,
    lr: float,
    latent_dim: int,
    hidden_dim: int,
):
    """
    Train a single-modality autoencoder and save only the encoder weights.

    Args:
        name:        Human-readable modality label (e.g. 'rna', 'xray', 'mri').
        parquet_path: Path to the unified parquet file for this modality.
        out_dir:     Directory where the encoder checkpoint is written.
        device:      Torch device to use for training.
        epochs:      Number of training epochs.
        batch_size:  Mini-batch size.
        lr:          Adam learning rate.
        latent_dim:  Dimensionality of the encoder bottleneck.
        hidden_dim:  Width of the encoder/decoder hidden layers.

    The encoder state dict is saved to ``out_dir/encoder_<name>.pt``.
    Batches with non-finite loss are skipped rather than crashing.
    """
    print(f"[{name}] Loading {parquet_path}")
    feats = load_features(parquet_path)
    n, d = feats.shape
    print(f"[{name}] N={n}, D={d}")

    dataset = FeatureDataset(feats)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    model = AutoEncoder(input_dim=d, latent_dim=latent_dim, hidden_dim=hidden_dim).to(
        device
    )
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    mse_loss = nn.MSELoss()

    for epoch in range(1, epochs + 1):
        model.train()
        losses = []

        for xb in loader:
            xb = xb.to(device, non_blocking=True)

            # normalize per-sample for stability
            xb_norm = normalize_batch(xb)

            opt.zero_grad()
            recon = model(xb_norm)
            loss = mse_loss(recon, xb_norm)

            if not torch.isfinite(loss):
                print(f"[{name}] Non-finite loss at epoch {epoch}; skipping batch")
                continue

            loss.backward()
            opt.step()
            losses.append(loss.item())

        if len(losses) == 0:
            print(f"[{name}] No valid batches in epoch {epoch}, stopping early")
            break

        epoch_mse = float(np.mean(losses))
        print(f"[{name}] Epoch {epoch:02d} | MSE={epoch_mse:.4f}")

    # Save encoder only (for downstream fusion / alignment)
    enc_path = out_dir / f"encoder_{name}.pt"
    torch.save(model.encoder.state_dict(), enc_path)
    print(f"[{name}] Saved encoder -> {enc_path}")


# ----------------------------
# Main
# ----------------------------

def main():
    """
    Parse CLI arguments, set the random seed, then sequentially train
    autoencoders for the RNA (GTEx), X-ray (CheXpert), and MRI (IXI)
    modalities.  Encoder checkpoints are written to
    ``models/run_<run_id>/encoder_<modality>.pt``.
    """
    parser = argparse.ArgumentParser(
        description="Train modality-specific autoencoders (v2, safer normalization)."
    )
    parser.add_argument("--run_id", type=str, default="v2")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--latent_dim", type=int, default=256)
    parser.add_argument("--hidden_dim", type=int, default=512)
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="cuda or cpu",
    )

    args = parser.parse_args()
    set_seed(42)

    device = torch.device(args.device)
    print(f"Using device: {device}")

    project_root = Path(__file__).resolve().parents[2]
    data_processed = project_root / "data" / "processed"
    models_dir = project_root / "models" / f"run_{args.run_id}"
    ensure_dir(models_dir)

    # RNA
    train_autoencoder(
        name="rna",
        parquet_path=data_processed / "unified_gtex.parquet",
        out_dir=models_dir,
        device=device,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        latent_dim=args.latent_dim,
        hidden_dim=args.hidden_dim,
    )

    # X-ray
    train_autoencoder(
        name="xray",
        parquet_path=data_processed / "unified_chexpert.parquet",
        out_dir=models_dir,
        device=device,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        latent_dim=args.latent_dim,
        hidden_dim=args.hidden_dim,
    )

    # MRI
    train_autoencoder(
        name="mri",
        parquet_path=data_processed / "unified_ixi.parquet",
        out_dir=models_dir,
        device=device,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        latent_dim=args.latent_dim,
        hidden_dim=args.hidden_dim,
    )


if __name__ == "__main__":
    main()
