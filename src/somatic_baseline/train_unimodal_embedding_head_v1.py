# scripts/71_train_unified_agentic.py
"""
train_unimodal_embedding_head_v1.py

Train per-modality MLP age regressors from raw (non-PCA) modality features
using an agentic (hard-sample curriculum) sampling strategy.

Datasets used:
  - GTEx RNA    (data/processed/unified_gtex.parquet)
  - CheXpert    (data/processed/unified_chexpert.parquet)
  - IXI MRI     (data/processed/unified_ixi.parquet)

Agentic curriculum:
  - Per-sample MSE losses are tracked each epoch.
  - A WeightedRandomSampler is rebuilt so harder samples are drawn more often.
  - ``agentic_factor=0.7`` is used (70 % weight towards hard samples, 30 %
    uniform) throughout training.

Output per modality:  experiments/run_<run_id>/models/<name>_age_mlp.pt
Loss plot per modality: experiments/run_<run_id>/figures/<name>_loss.png
Summary curves:         experiments/run_<run_id>/unimodal_curves.pkl
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import matplotlib.pyplot as plt
import pickle

ROOT = Path(".")


# ------------------------- Dataset -------------------------


class AgeDataset(Dataset):
    """
    Dataset for one modality from unified_*.parquet

    Expects columns:
      - 'features' : list/array-like of shape [D]
      - 'age'      : float (no NaNs; filtering done outside)
    """

    def __init__(self, df: pd.DataFrame):
        """
        Args:
            df: DataFrame with 'features' and 'age' columns.  Rows where
                'age' is NaN are filtered out.

        Raises:
            ValueError: if 'age' column is missing or no samples remain
                        after filtering.
        """
        df = df.copy()
        if "age" not in df.columns:
            raise ValueError("DataFrame must contain an 'age' column")

        # keep only rows where age is not NaN
        df = df[df["age"].notna()].reset_index(drop=True)

        self.ages = df["age"].astype("float32").to_numpy()

        feats = df["features"].tolist()
        self.features = [self._to_array(x) for x in feats]

        if len(self.features) == 0:
            raise ValueError("No samples after filtering for non-NaN age")

        self.input_dim = self.features[0].shape[0]

    @staticmethod
    def _to_array(x):
        """Convert *x* to a writable float32 numpy array (copies if read-only)."""
        # robust conversion to float32 and ensure writable
        arr = np.asarray(x, dtype="float32")
        if not arr.flags.writeable:
            arr = arr.copy()
        return arr

    def __len__(self):
        """Return the number of samples in the dataset."""
        return len(self.ages)

    def __getitem__(self, idx):
        """Return (feature tensor, age tensor, local index) for sample *idx*."""
        x = self.features[idx]
        y = self.ages[idx]
        # idx is LOCAL index in this dataset (0..len-1)
        return (
            torch.from_numpy(x),
            torch.tensor(y, dtype=torch.float32),
            int(idx),
        )


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


# ------------------------- Training -------------------------


def train_one_modality(
    name: str,
    df: pd.DataFrame,
    device: torch.device,
    batch_size: int,
    n_epochs: int,
    agentic_factor: float,
    run_dir: Path,
    val_fraction: float = 0.15,
    lr: float = 1e-3,
):
    """
    Train a model for a single modality (GTEx / CheXpert / IXI) with an
    "agentic" curriculum:

    - Track per-sample loss each epoch.
    - Increase sampling probability of hard samples in the next epoch.

    agentic_factor controls how strongly we focus on hard examples:
      0.0 -> uniform sampling
      0.7 -> 70% of probability mass follows normalized loss weights.
    """

    print(f"\n===== Training modality: {name} =====")

    # 1) Clean dataframe and build full dataset to inspect sizes
    df_clean = df[df["age"].notna()].reset_index(drop=True)
    if len(df_clean) == 0:
        print(f"[{name}] No samples with age; skipping.")
        return {
            "train_mse": [],
            "val_mse": [],
            "val_mae": [],
        }

    # Random split indices
    rng = np.random.default_rng(seed=42)
    idx_all = np.arange(len(df_clean))
    rng.shuffle(idx_all)
    n_val = max(1, int(val_fraction * len(df_clean)))
    val_idx = idx_all[:n_val]
    train_idx = idx_all[n_val:]

    df_train = df_clean.iloc[train_idx].reset_index(drop=True)
    df_val = df_clean.iloc[val_idx].reset_index(drop=True)

    ds_train = AgeDataset(df_train)
    ds_val = AgeDataset(df_val)

    N_train = len(ds_train)
    N_val = len(ds_val)
    input_dim = ds_train.input_dim

    print(
        f"[{name}] Train size = {N_train}, "
        f"Val size = {N_val}, input_dim = {input_dim}"
    )

    # per-sample losses for agentic curriculum
    train_losses_per_sample = np.ones(N_train, dtype=np.float32)

    model = AgeMLP(input_dim).to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss(reduction="none")  # per-sample loss

    curves = {
        "train_mse": [],
        "val_mse": [],
        "val_mae": [],
    }

    for epoch in range(1, n_epochs + 1):
        model.train()

        # ----- build sampler with agentic weights -----
        losses = train_losses_per_sample.copy()
        # normalize to [0,1]
        if np.all(np.isfinite(losses)) and losses.max() > 0:
            losses = losses / (losses.max() + 1e-8)
        else:
            losses = np.ones_like(losses)

        uniform = np.ones_like(losses) / len(losses)
        hard = losses / (losses.sum() + 1e-8)
        weights = (1.0 - agentic_factor) * uniform + agentic_factor * hard
        weights = torch.tensor(weights, dtype=torch.float32)

        sampler = WeightedRandomSampler(
            weights=weights, num_samples=len(weights), replacement=True
        )

        train_loader = DataLoader(
            ds_train,
            batch_size=batch_size,
            sampler=sampler,
            num_workers=0,
        )

        # ----- training loop -----
        epoch_batch_losses = []
        per_sample_loss = np.zeros(N_train, dtype=np.float32)
        visits = np.zeros(N_train, dtype=np.float32)

        for xb, yb, idxb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            optim.zero_grad()
            preds = model(xb)
            loss_vec = loss_fn(preds, yb)  # [batch]
            loss = loss_vec.mean()
            loss.backward()
            optim.step()

            loss_np = loss_vec.detach().cpu().numpy()
            idx_np = idxb.numpy()

            per_sample_loss[idx_np] += loss_np
            visits[idx_np] += 1.0
            epoch_batch_losses.append(loss_np)

        # average loss per sample across draws
        avg_loss = per_sample_loss / np.maximum(1.0, visits)
        train_losses_per_sample = avg_loss

        train_mse = float(np.mean(np.concatenate(epoch_batch_losses)))

        # ----- validation -----
        model.eval()
        val_losses = []
        abs_errs = []
        with torch.no_grad():
            val_loader = DataLoader(
                ds_val, batch_size=batch_size, shuffle=False, num_workers=0
            )
            for xb, yb, _ in val_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                preds = model(xb)
                lv = loss_fn(preds, yb)
                val_losses.append(lv.detach().cpu().numpy())
                abs_errs.append(torch.abs(preds - yb).detach().cpu().numpy())

        val_loss_mean = float(np.mean(np.concatenate(val_losses)))
        val_mae = float(np.mean(np.concatenate(abs_errs)))

        curves["train_mse"].append(train_mse)
        curves["val_mse"].append(val_loss_mean)
        curves["val_mae"].append(val_mae)

        print(
            f"[{name}] Epoch {epoch:02d} | "
            f"train MSE={train_mse:.3f} | "
            f"val MSE={val_loss_mean:.3f}, val MAE={val_mae:.3f}"
        )

    # save model for this run/modality
    model_dir = run_dir / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    out_path = model_dir / f"{name}_age_mlp.pt"
    torch.save({"state_dict": model.state_dict(), "input_dim": input_dim}, out_path)
    print(f"[{name}] Saved model -> {out_path}")

    # quick per-run loss plot
    fig_dir = run_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    epochs = np.arange(1, len(curves["train_mse"]) + 1)

    plt.figure()
    plt.plot(epochs, curves["train_mse"], label="train MSE")
    plt.plot(epochs, curves["val_mse"], label="val MSE")
    plt.xlabel("Epoch")
    plt.ylabel("MSE")
    plt.title(f"{name} loss curves")
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_dir / f"{name}_loss.png")
    plt.close()

    return curves


# ------------------------- main -------------------------


def main():
    """
    Parse CLI arguments and sequentially train age regressors for the GTEx
    RNA, CheXpert X-ray, and IXI MRI modalities using raw (non-PCA) features.
    All training curves are pickled to
    ``experiments/run_<run_id>/unimodal_curves.pkl``.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="cuda or cpu",
    )
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--epochs_rna", type=int, default=20)
    parser.add_argument("--epochs_xray", type=int, default=12)
    parser.add_argument("--epochs_mri", type=int, default=25)
    parser.add_argument(
        "--run_id",
        type=str,
        default="000",
        help="3-digit run id used to bucket curves/models",
    )
    args = parser.parse_args()

    device = torch.device(args.device)
    print(f"Using device: {device}")

    # per-run directory
    run_dir = ROOT / "experiments" / f"run_{args.run_id}"
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"[RUN] Saving models/curves in: {run_dir}")

    processed = ROOT / "data" / "processed"

    all_curves = {}

    # ----- GTEx (RNA) -----
    gtex_file = processed / "unified_gtex.parquet"
    if gtex_file.exists():
        df_gtex = pd.read_parquet(gtex_file)
        curves_rna = train_one_modality(
            "gtex_rna",
            df_gtex,
            device=device,
            batch_size=args.batch_size,
            n_epochs=args.epochs_rna,
            agentic_factor=0.7,
            run_dir=run_dir,
        )
        all_curves["gtex_rna"] = curves_rna
    else:
        print("[WARN] unified_gtex.parquet not found, skipping GTEx.")

    # ----- CheXpert (X-ray) -----
    cxp_file = processed / "unified_chexpert.parquet"
    if cxp_file.exists():
        df_cxp = pd.read_parquet(cxp_file)
        curves_xray = train_one_modality(
            "chexpert_xray",
            df_cxp,
            device=device,
            batch_size=args.batch_size,
            n_epochs=args.epochs_xray,
            agentic_factor=0.7,
            run_dir=run_dir,
        )
        all_curves["chexpert_xray"] = curves_xray
    else:
        print("[WARN] unified_chexpert.parquet not found, skipping CheXpert.")

    # ----- IXI (MRI) -----
    ixi_file = processed / "unified_ixi.parquet"
    if ixi_file.exists():
        df_ixi = pd.read_parquet(ixi_file)
        curves_mri = train_one_modality(
            "ixi_mri",
            df_ixi,
            device=device,
            batch_size=args.batch_size,
            n_epochs=args.epochs_mri,
            agentic_factor=0.7,
            run_dir=run_dir,
        )
        all_curves["ixi_mri"] = curves_mri
    else:
        print("[WARN] unified_ixi.parquet not found, skipping IXI.")

    # save all curves for this run
    curves_path = run_dir / "unimodal_curves.pkl"
    with open(curves_path, "wb") as f:
        pickle.dump(all_curves, f)
    print(f"[RUN {args.run_id}] Saved unimodal curves -> {curves_path}")


if __name__ == "__main__":
    main()
