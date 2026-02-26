# scripts/71_1_train_unified_agentic_pca.py
"""
train_unimodal_pca_head_v1.py

Train per-modality MLP age regressors using PCA-compressed features and an
agentic (curriculum) sampling strategy.

Unlike ``train_unimodal_embedding_head_v1.py``, this script expects the GTEx
features to already be PCA-compressed (unified_gtex_pca.parquet produced by
encode_gtex_pca_v1.py) rather than full-dimensional gene expression vectors.

Agentic curriculum:
  - Each epoch, per-sample MSE losses are tracked.
  - A WeightedRandomSampler is rebuilt each epoch so that harder samples are
    drawn more frequently.
  - The blend between uniform and hard-sample weighting is linearly ramped
    from ``agentic_factor_start`` to ``agentic_factor_end`` after
    ``warmup_epochs`` initial uniform-sampling epochs.

Output per modality:  experiments/run_<run_id>/models/<name>_age_mlp.pt
Summary curves:        experiments/run_<run_id>/unimodal_curves_pca.pkl
"""

import argparse
from pathlib import Path
import pickle

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler


ROOT = Path(".")
PROCESSED = ROOT / "data" / "processed"


# ------------------------- Dataset -------------------------


class AgeDataset(Dataset):
    """
    Generic dataset for one modality (RNA / X-ray / MRI) using a unified_*.parquet file.

    Expects columns:
      - 'features': array-like (same dim for all rows in this file)
      - 'age': float (NaNs are filtered out)
    """

    def __init__(self, df: pd.DataFrame):
        """
        Initialise from a DataFrame with 'features' and 'age' columns.
        Rows with NaN age are filtered out.  ``input_dim`` is inferred from
        the first surviving row.
        """
        df = df.copy()
        df = df[df["age"].notna()].reset_index(drop=True)

        self.features = df["features"].tolist()
        self.ages = df["age"].astype("float32").to_numpy()

        first = self._coerce_array(self.features[0])
        self.input_dim = first.shape[0]

    @staticmethod
    def _coerce_array(x):
        """Convert *x* (list, array, etc.) to a writable float32 numpy array."""
        if isinstance(x, np.ndarray):
            return x.astype("float32", copy=True)
        return np.asarray(x, dtype="float32").copy()

    def __len__(self):
        """Return the number of samples in the dataset."""
        return len(self.ages)

    def __getitem__(self, idx):
        """Return (feature tensor, age tensor, local index) for sample *idx*."""
        x = self._coerce_array(self.features[idx])
        y = self.ages[idx]
        # idx is local index in this dataset (0..len-1)
        return torch.from_numpy(x), torch.tensor(y, dtype=torch.float32), idx


# ------------------------- Model -------------------------


class AgeMLP(nn.Module):
    """
    Simple MLP age regressor:

      input_dim -> 512 -> 256 -> 1
    """

    def __init__(self, input_dim: int):
        """
        Args:
            input_dim: Dimensionality of the input feature vector.
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.GELU(),
            nn.LayerNorm(512),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.LayerNorm(256),
            nn.Linear(256, 1),
        )

    def forward(self, x):
        """Return predicted scalar age for each sample in the batch."""
        return self.net(x).squeeze(-1)


# ------------------------- Training utils -------------------------


def train_one_modality(
    name: str,
    df: pd.DataFrame,
    device: torch.device,
    batch_size: int,
    n_epochs: int,
    lr: float,
    weight_decay: float,
    agentic_factor_start: float,
    agentic_factor_end: float,
    warmup_epochs: int,
    run_dir: Path,
    val_fraction: float = 0.15,
):
    """
    Train a model for a single modality (GTEx / CheXpert / IXI) with an
    agentic curriculum:

      - Track per-sample loss each epoch.
      - Increase sampling probability of hard samples in the next epoch.

    Agentic factor is linearly ramped from agentic_factor_start to
    agentic_factor_end after warmup_epochs.
    """
    print(f"\n===== Training modality: {name} =====")
    ds_full = AgeDataset(df)
    N = len(ds_full)
    input_dim = ds_full.input_dim
    print(f"[{name}] Trainable samples = {N}, input_dim = {input_dim}")

    # Train/val split
    rng = np.random.default_rng(seed=42)
    idx_all = np.arange(N)
    rng.shuffle(idx_all)
    n_val = int(val_fraction * N)
    val_idx = idx_all[:n_val]
    train_idx = idx_all[n_val:]

    def subset_dataset(ds, indices):
        """Return a new AgeDataset containing only the rows selected by *indices*."""
        new = AgeDataset.__new__(AgeDataset)
        new.features = [ds.features[i] for i in indices]
        new.ages = ds.ages[indices]
        new.input_dim = ds.input_dim
        return new

    ds_train = subset_dataset(ds_full, train_idx)
    ds_val = subset_dataset(ds_full, val_idx)

    print(f"[{name}] Train size = {len(ds_train)}, Val size = {len(ds_val)}")

    # Agentic per-sample loss tracker (one scalar per train example)
    train_losses_per_sample = np.ones(len(ds_train), dtype=np.float32)

    model = AgeMLP(input_dim).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay,
    )
    loss_fn = nn.MSELoss(reduction="none")

    curves = {
        "train_mse": [],
        "val_mse": [],
        "val_mae": [],
    }

    for epoch in range(1, n_epochs + 1):
        model.train()

        # ----- agentic factor schedule -----
        if epoch <= warmup_epochs:
            af = agentic_factor_start
        else:
            # Linear ramp after warmup
            t = (epoch - warmup_epochs) / max(1, (n_epochs - warmup_epochs))
            af = agentic_factor_start + t * (agentic_factor_end - agentic_factor_start)
        af = float(np.clip(af, 0.0, 1.0))

        # ----- build sampler -----
        losses = train_losses_per_sample.copy()
        if np.all(np.isfinite(losses)) and losses.max() > 0:
            losses = losses / (losses.max() + 1e-8)
        else:
            losses = np.ones_like(losses)

        uniform = np.ones_like(losses) / len(losses)
        hard = losses / (losses.sum() + 1e-8)

        weights = (1.0 - af) * uniform + af * hard
        weights = torch.tensor(weights, dtype=torch.float32)

        sampler = WeightedRandomSampler(
            weights=weights,
            num_samples=len(weights),
            replacement=True,
        )

        train_loader = DataLoader(
            ds_train,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=0,
        )

        epoch_losses_vec = []
        per_sample_loss = np.zeros(len(ds_train), dtype=np.float32)
        seen_counts = np.zeros(len(ds_train), dtype=np.float32)

        # ----- training loop -----
        for xb, yb, idxb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad()
            preds = model(xb)
            loss_vec = loss_fn(preds, yb)  # [batch]
            loss = loss_vec.mean()
            loss.backward()
            optimizer.step()

            loss_np = loss_vec.detach().cpu().numpy()
            idx_np = idxb.detach().cpu().numpy()

            epoch_losses_vec.append(loss_np)

            per_sample_loss[idx_np] += loss_np
            seen_counts[idx_np] += 1.0

        # Average loss per sample
        avg_loss = np.zeros_like(per_sample_loss)
        mask = seen_counts > 0
        avg_loss[mask] = per_sample_loss[mask] / seen_counts[mask]
        # For never-sampled (very unlikely), fill with global mean
        if (~mask).any():
            fill_val = avg_loss[mask].mean() if mask.any() else 1.0
            avg_loss[~mask] = fill_val

        train_losses_per_sample = avg_loss
        train_loss_mean = float(np.mean(np.concatenate(epoch_losses_vec)))

        # ----- validation -----
        model.eval()
        val_losses = []
        abs_errs = []

        with torch.no_grad():
            val_loader = DataLoader(
                ds_val,
                batch_size=batch_size,
                shuffle=False,
                num_workers=0,
            )
            for xb, yb, _ in val_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                preds = model(xb)
                l = loss_fn(preds, yb)
                val_losses.append(l.detach().cpu().numpy())
                abs_errs.append(torch.abs(preds - yb).detach().cpu().numpy())

        val_loss_mean = float(np.mean(np.concatenate(val_losses)))
        val_mae = float(np.mean(np.concatenate(abs_errs)))

        curves["train_mse"].append(train_loss_mean)
        curves["val_mse"].append(val_loss_mean)
        curves["val_mae"].append(val_mae)

        print(
            f"[{name}] Epoch {epoch:02d} | "
            f"af={af:.2f} | "
            f"train MSE={train_loss_mean:.3f} | "
            f"val MSE={val_loss_mean:.3f}, val MAE={val_mae:.3f}"
        )

    # Save model + curves
    models_dir = run_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    out_path = models_dir / f"{name}_age_mlp.pt"
    torch.save({"state_dict": model.state_dict(), "input_dim": input_dim}, out_path)
    print(f"[{name}] Saved model -> {out_path}")

    return curves


# ------------------------- main -------------------------


def main():
    """
    Parse CLI arguments and sequentially train age regressors for
    PCA-compressed GTEx RNA, CheXpert X-ray, and IXI MRI modalities.
    Training curves for all three modalities are pickled together in
    ``experiments/run_<run_id>/unimodal_curves_pca.pkl``.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="cuda or cpu",
    )
    parser.add_argument("--batch_size", type=int, default=256)

    parser.add_argument("--epochs_rna", type=int, default=30)
    parser.add_argument("--epochs_xray", type=int, default=20)
    parser.add_argument("--epochs_mri", type=int, default=60)

    parser.add_argument("--lr_rna", type=float, default=1e-3)
    parser.add_argument("--lr_xray", type=float, default=1e-3)
    parser.add_argument("--lr_mri", type=float, default=5e-4)

    parser.add_argument("--wd_rna", type=float, default=1e-4)
    parser.add_argument("--wd_xray", type=float, default=1e-3)
    parser.add_argument("--wd_mri", type=float, default=1e-4)

    parser.add_argument("--agentic_start", type=float, default=0.0)
    parser.add_argument("--agentic_end", type=float, default=0.7)
    parser.add_argument("--warmup_epochs", type=int, default=5)

    parser.add_argument(
        "--run_id",
        type=str,
        default="pca001",
        help="run identifier (for experiments/run_<id>)",
    )

    args = parser.parse_args()

    device = torch.device(args.device)
    print(f"Using device: {device}")

    run_dir = ROOT / "experiments" / f"run_{args.run_id}"
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"[RUN] Saving models/curves in: {run_dir}")

    # ----- GTEx (RNA, PCA version) -----
    gtex_file = PROCESSED / "unified_gtex_pca.parquet"
    if gtex_file.exists():
        df_gtex = pd.read_parquet(gtex_file)
        curves_rna = train_one_modality(
            "gtex_rna_pca",
            df_gtex,
            device=device,
            batch_size=args.batch_size,
            n_epochs=args.epochs_rna,
            lr=args.lr_rna,
            weight_decay=args.wd_rna,
            agentic_factor_start=args.agentic_start,
            agentic_factor_end=args.agentic_end,
            warmup_epochs=args.warmup_epochs,
            run_dir=run_dir,
        )
    else:
        print("[WARN] unified_gtex_pca.parquet not found, skipping GTEx.")
        curves_rna = None

    # ----- CheXpert (X-ray) -----
    cxp_file = PROCESSED / "unified_chexpert.parquet"
    if cxp_file.exists():
        df_cxp = pd.read_parquet(cxp_file)
        curves_xray = train_one_modality(
            "chexpert_xray",
            df_cxp,
            device=device,
            batch_size=args.batch_size,
            n_epochs=args.epochs_xray,
            lr=args.lr_xray,
            weight_decay=args.wd_xray,
            agentic_factor_start=args.agentic_start,
            agentic_factor_end=args.agentic_end,
            warmup_epochs=args.warmup_epochs,
            run_dir=run_dir,
        )
    else:
        print("[WARN] unified_chexpert.parquet not found, skipping CheXpert.")
        curves_xray = None

    # ----- IXI (MRI) -----
    ixi_file = PROCESSED / "unified_ixi.parquet"
    if ixi_file.exists():
        df_ixi = pd.read_parquet(ixi_file)
        curves_mri = train_one_modality(
            "ixi_mri",
            df_ixi,
            device=device,
            batch_size=args.batch_size,
            n_epochs=args.epochs_mri,
            lr=args.lr_mri,
            weight_decay=args.wd_mri,
            agentic_factor_start=args.agentic_start,
            agentic_factor_end=args.agentic_end,
            warmup_epochs=args.warmup_epochs,
            run_dir=run_dir,
        )
    else:
        print("[WARN] unified_ixi.parquet not found, skipping IXI.")
        curves_mri = None

    # ----- Save curves summary -----
    curves = {
        "gtex_rna_pca": curves_rna,
        "chexpert_xray": curves_xray,
        "ixi_mri": curves_mri,
    }
    curves_path = run_dir / "unimodal_curves_pca.pkl"
    with curves_path.open("wb") as f:
        pickle.dump(curves, f)
    print(f"[RUN] Saved unimodal curves -> {curves_path}")


if __name__ == "__main__":
    main()
