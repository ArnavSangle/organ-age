# scripts/72_1_train_multimodal_fusion_v2.py
"""
train_fusion_sweep_v2.py

Train a multimodal fusion MLP age regressor (v2 sweep variant) on the
combined dataset ``data/processed/unified_all.parquet``.

All three modalities are combined into one flat dataset where each sample's
feature vector is padded/truncated to the maximum observed feature length and
a 3-dim modality one-hot is appended.  This lets a single MLP learn from all
modalities simultaneously.

Key features:
  - Automatic detection of base_dim from the longest feature vector.
  - Per-modality MSE / MAE reported at every validation step.
  - Best-epoch model and metrics saved separately from the final-epoch model.
  - Stronger L2 regularisation applied to the first layer (fc1) to reduce
    over-fitting on high-dimensional RNA features.

Output:
    experiments/run_<run_id>/models/fusion_age_mlp_v2.pt
    experiments/run_<run_id>/models/fusion_age_mlp_v2_best.pt
    experiments/run_<run_id>/fusion_best_metrics_v2.json
    experiments/run_<run_id>/fusion_curves_v2.pkl
"""

import argparse
from pathlib import Path
import json
import pickle

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader


ROOT = Path(".")
PROCESSED = ROOT / "data" / "processed"


# ------------------------- Dataset -------------------------


class FusionDataset(Dataset):
    """
    Dataset over unified_all.parquet for fusion v2.

    Expects columns:
      - 'features' : variable-length float list/array
      - 'age'      : float
      - 'modality' : {'rna', 'xray', 'mri'}

    We:
      - pad/truncate to base_dim
      - append 3-dim modality one-hot at the end (total_dim = base_dim + 3)
    """

    MODALITY_MAP = {"rna": 0, "xray": 1, "mri": 2}

    def __init__(self, df: pd.DataFrame, base_dim: int):
        """
        Build a fusion dataset from *df*.

        Args:
            df:       Unified parquet DataFrame with 'features', 'age', and
                      'modality' columns.  Rows with NaN age or unknown
                      modality are dropped.
            base_dim: All feature vectors are zero-padded or truncated to this
                      length before the modality one-hot is appended.
        """
        df = df.copy()
        df = df[df["age"].notna()].reset_index(drop=True)

        # Keep only modalities we know
        df = df[df["modality"].isin(self.MODALITY_MAP.keys())].reset_index(drop=True)

        self.features = df["features"].tolist()
        self.ages = df["age"].astype("float32").to_numpy()
        self.modalities = df["modality"].map(self.MODALITY_MAP).astype("int64").to_numpy()

        self.base_dim = int(base_dim)
        self.total_dim = self.base_dim + 3  # +3 for modality one-hot

    def __len__(self):
        """Return the number of samples in the dataset."""
        return len(self.ages)

    def __getitem__(self, idx):
        """Return (x_full, age, mod_idx) tensors for sample at *idx*."""
        feat = self.features[idx]
        age = self.ages[idx]
        mod_idx = int(self.modalities[idx])

        # Coerce to float32 array and pad/truncate to base_dim
        feat_arr = np.asarray(feat, dtype="float32")
        x_base = np.zeros(self.base_dim, dtype="float32")
        L = min(len(feat_arr), self.base_dim)
        x_base[:L] = feat_arr[:L]

        # Append modality one-hot
        x_full = np.zeros(self.total_dim, dtype="float32")
        x_full[: self.base_dim] = x_base
        x_full[self.base_dim + mod_idx] = 1.0  # one-hot

        # Use tensor() (copies) to avoid non-writable warnings
        x = torch.tensor(x_full, dtype=torch.float32)
        y = torch.tensor(age, dtype=torch.float32)
        m = torch.tensor(mod_idx, dtype=torch.long)

        return x, y, m


# ------------------------- Model -------------------------


class FusionMLP(nn.Module):
    """
    Fusion MLP over concatenated features + modality one-hot.

    Architecture:
      input_dim -> 1024 -> 512 -> 1

    With GELU, LayerNorm, and Dropout.
    """

    def __init__(self, input_dim: int, dropout: float = 0.1):
        """
        Args:
            input_dim: Total input dimensionality (base_dim + 3 for one-hot).
            dropout:   Dropout probability applied after each hidden layer.
        """
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 1024)
        self.ln1 = nn.LayerNorm(1024)
        self.fc2 = nn.Linear(1024, 512)
        self.ln2 = nn.LayerNorm(512)
        self.fc3 = nn.Linear(512, 1)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """Return predicted scalar age for each sample in the batch."""
        x = self.fc1(x)
        x = self.act(x)
        x = self.ln1(x)
        x = self.dropout(x)

        x = self.fc2(x)
        x = self.act(x)
        x = self.ln2(x)
        x = self.dropout(x)

        x = self.fc3(x)
        return x.squeeze(-1)


# ------------------------- Training -------------------------


def train_fusion(
    df: pd.DataFrame,
    device: torch.device,
    batch_size: int,
    n_epochs: int,
    run_dir: Path,
    lr: float = 1e-3,
    dropout: float = 0.1,
):
    """
    Train the FusionMLP on the combined unified dataset.

    Args:
        df:         Combined DataFrame from unified_all.parquet.
        device:     Torch device for training.
        batch_size: Mini-batch size.
        n_epochs:   Number of training epochs.
        run_dir:    Experiment directory; models and metrics are written here.
        lr:         AdamW learning rate.
        dropout:    Dropout probability applied after each hidden layer.

    Saves the final model, the best-epoch model, best metrics JSON, and a
    pickle of all training curves to *run_dir*.
    """
    run_dir.mkdir(parents=True, exist_ok=True)
    models_dir = run_dir / "models"
    models_dir.mkdir(exist_ok=True)

    # Detect base_dim from features lengths
    lengths = df["features"].map(len)
    base_dim = int(lengths.max())
    print(f"[fusion] Detected base_dim (max feature length) = {base_dim}")

    ds_full = FusionDataset(df, base_dim=base_dim)
    N = len(ds_full)
    input_dim = ds_full.total_dim

    print(f"[fusion] N = {N}, input_dim = {input_dim}")

    # Train/val split
    rng = np.random.default_rng(seed=123)
    idx_all = np.arange(N)
    rng.shuffle(idx_all)
    n_val = int(0.15 * N)
    val_idx = idx_all[:n_val]
    train_idx = idx_all[n_val:]

    def subset_dataset(ds, indices):
        """Return a new FusionDataset containing only rows at *indices*."""
        new = FusionDataset.__new__(FusionDataset)
        # slice underlying arrays
        new.features = [ds.features[i] for i in indices]
        new.ages = ds.ages[indices]
        new.modalities = ds.modalities[indices]
        new.base_dim = ds.base_dim
        new.total_dim = ds.total_dim
        return new

    ds_train = subset_dataset(ds_full, train_idx)
    ds_val = subset_dataset(ds_full, val_idx)

    print(
        f"[fusion] Train size = {len(ds_train)}, "
        f"Val size = {len(ds_val)}"
    )

    train_loader = DataLoader(
        ds_train,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
    )
    val_loader = DataLoader(
        ds_val,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )

    model = FusionMLP(input_dim=input_dim, dropout=dropout).to(device)
    loss_fn = nn.MSELoss(reduction="none")

    # Param groups: stronger weight decay on first layer
    param_groups = [
        {
            "params": list(model.fc1.parameters()),
            "weight_decay": 1e-2,
        },
        {
            "params": (
                list(model.ln1.parameters())
                + list(model.fc2.parameters())
                + list(model.ln2.parameters())
                + list(model.fc3.parameters())
            ),
            "weight_decay": 1e-4,
        },
    ]
    optimizer = torch.optim.AdamW(param_groups, lr=lr)

    # Tracking
    curves = {
        "epoch": [],
        "train_mse": [],
        "val_mse": [],
        "val_mae": [],
        "val_mse_rna": [],
        "val_mae_rna": [],
        "val_mse_xray": [],
        "val_mae_xray": [],
        "val_mse_mri": [],
        "val_mae_mri": [],
    }

    best_val_mse = float("inf")
    best_epoch = -1
    best_state_dict = None
    best_metrics = {}

    modality_names = {0: "rna", 1: "xray", 2: "mri"}

    for epoch in range(1, n_epochs + 1):
        # ---------- Train ----------
        model.train()
        train_losses = []

        for xb, yb, mb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad()
            preds = model(xb)
            loss_vec = loss_fn(preds, yb)
            loss = loss_vec.mean()
            loss.backward()
            optimizer.step()

            train_losses.append(loss_vec.detach().cpu().numpy())

        train_mse = float(np.concatenate(train_losses).mean())

        # ---------- Validation ----------
        model.eval()
        val_losses_all = []
        abs_errs_all = []

        # per-modality accumulators
        mod_losses = {0: [], 1: [], 2: []}
        mod_abs_errs = {0: [], 1: [], 2: []}

        with torch.no_grad():
            for xb, yb, mb in val_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                mb = mb.to(device)

                preds = model(xb)
                loss_vec = (preds - yb) ** 2
                abs_err_vec = torch.abs(preds - yb)

                loss_np = loss_vec.detach().cpu().numpy()
                abs_np = abs_err_vec.detach().cpu().numpy()

                val_losses_all.append(loss_np)
                abs_errs_all.append(abs_np)

                # per-modality slices
                for m_id in (0, 1, 2):
                    mask = (mb == m_id)
                    if mask.any():
                        mask_np = mask.cpu().numpy().astype(bool)
                        mod_losses[m_id].append(loss_np[mask_np])
                        mod_abs_errs[m_id].append(abs_np[mask_np])

                # flatten all val losses / abs errors
        val_mse = float(np.concatenate(val_losses_all).mean())
        val_mae = float(np.concatenate(abs_errs_all).mean())


        # Per-modality metrics (if present)
        mod_mse = {}
        mod_mae = {}
        for m_id in (0, 1, 2):
            if len(mod_losses[m_id]) > 0:
                losses_concat = np.concatenate(mod_losses[m_id])
                abs_concat = np.concatenate(mod_abs_errs[m_id])
                mod_mse[m_id] = float(losses_concat.mean())
                mod_mae[m_id] = float(abs_concat.mean())
            else:
                mod_mse[m_id] = float("nan")
                mod_mae[m_id] = float("nan")

        # Log & store curves
        curves["epoch"].append(epoch)
        curves["train_mse"].append(train_mse)
        curves["val_mse"].append(val_mse)
        curves["val_mae"].append(val_mae)
        curves["val_mse_rna"].append(mod_mse[0])
        curves["val_mae_rna"].append(mod_mae[0])
        curves["val_mse_xray"].append(mod_mse[1])
        curves["val_mae_xray"].append(mod_mae[1])
        curves["val_mse_mri"].append(mod_mse[2])
        curves["val_mae_mri"].append(mod_mae[2])

        print(
            f"[fusion] Epoch {epoch:02d} | "
            f"train MSE={train_mse:.3f} | "
            f"val MSE={val_mse:.3f}, val MAE={val_mae:.3f} | "
            f"RNA MSE={mod_mse[0]:.3f}, MAE={mod_mae[0]:.3f} | "
            f"X-ray MSE={mod_mse[1]:.3f}, MAE={mod_mae[1]:.3f} | "
            f"MRI MSE={mod_mse[2]:.3f}, MAE={mod_mae[2]:.3f}"
        )

        # ---------- Best-epoch tracking ----------
        if val_mse < best_val_mse:
            best_val_mse = val_mse
            best_epoch = epoch
            best_state_dict = model.state_dict()
            best_metrics = {
                "epoch": epoch,
                "val_mse": val_mse,
                "val_mae": val_mae,
                "val_mse_rna": mod_mse[0],
                "val_mae_rna": mod_mae[0],
                "val_mse_xray": mod_mse[1],
                "val_mae_xray": mod_mae[1],
                "val_mse_mri": mod_mse[2],
                "val_mae_mri": mod_mae[2],
            }

    # ---------- Save final + best ----------
    final_path = models_dir / "fusion_age_mlp_v2.pt"
    torch.save(
        {
            "state_dict": model.state_dict(),
            "input_dim": input_dim,
            "base_dim": base_dim,
            "total_dim": ds_full.total_dim,
            "dropout": dropout,
            "lr": lr,
            "best_epoch": best_epoch,
            "best_val_mse": best_val_mse,
        },
        final_path,
    )
    print(f"[fusion] Saved final fusion model -> {final_path}")

    if best_state_dict is not None:
        best_path = models_dir / "fusion_age_mlp_v2_best.pt"
        torch.save(
            {
                "state_dict": best_state_dict,
                "input_dim": input_dim,
                "base_dim": base_dim,
                "total_dim": ds_full.total_dim,
                "dropout": dropout,
                "lr": lr,
                "best_epoch": best_epoch,
                "best_val_mse": best_val_mse,
            },
            best_path,
        )
        print(
            f"[fusion] Saved BEST fusion model (epoch {best_epoch}) -> {best_path}"
        )

        # Also dump JSON with best metrics
        best_json = run_dir / "fusion_best_metrics_v2.json"
        with best_json.open("w") as f:
            json.dump(best_metrics, f, indent=2)
        print(f"[fusion] Saved best metrics -> {best_json}")

    # Save curves for plotting
    curves_path = run_dir / "fusion_curves_v2.pkl"
    with curves_path.open("wb") as f:
        pickle.dump(curves, f)
    print(f"[fusion] Saved fusion curves -> {curves_path}")


# ------------------------- main -------------------------


def main():
    """
    Parse CLI arguments, load unified_all.parquet, and launch fusion training.
    The run directory is determined by ``--run_id``.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="cuda or cpu",
    )
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--run_id", type=str, default="pca001")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--dropout", type=float, default=0.1)
    args = parser.parse_args()

    device = torch.device(args.device)
    print(f"Using device: {device}")
    print(
        f"[fusion] Config: batch_size={args.batch_size}, "
        f"epochs={args.epochs}, lr={args.lr}, dropout={args.dropout}"
    )

    unified_path = PROCESSED / "unified_all.parquet"
    if not unified_path.exists():
        raise FileNotFoundError(f"Unified dataset not found: {unified_path}")

    print(f"[fusion] Loading unified_all from {unified_path}")
    df = pd.read_parquet(unified_path)

    run_dir = ROOT / "experiments" / f"run_{args.run_id}"
    print(f"[RUN] Saving fusion v2 outputs in: {run_dir}")

    train_fusion(
        df=df,
        device=device,
        batch_size=args.batch_size,
        n_epochs=args.epochs,
        run_dir=run_dir,
        lr=args.lr,
        dropout=args.dropout,
    )


if __name__ == "__main__":
    main()
