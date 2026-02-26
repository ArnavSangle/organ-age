# scripts/72_train_multimodal_fusion.py
"""
train_fusion_single_v2.py

Train a multimodal fusion age regressor with modality-specific encoders and a
shared regression head.

Architecture (FusionModel):
  - Three independent encoders (RNA / X-ray / MRI), each mapping their
    modality's raw feature dimension to a 256-d hidden representation:
      input -> 512 -> GELU -> LayerNorm -> 256 -> GELU -> LayerNorm
  - A single shared head applied to each encoder's output:
      256 -> 128 -> GELU -> LayerNorm -> 1

A custom collate function (fusion_collate) groups each mini-batch by modality
so that all forward passes within a batch share the same encoder, avoiding
issues with heterogeneous feature dimensionalities.

Output:
    experiments/run_<run_id>/models/fusion_age_mlp.pt
    experiments/run_<run_id>/figures/fusion_loss.png
    experiments/run_<run_id>/fusion_curves.pkl
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import pickle

ROOT = Path(".")


# ------------------------- Dataset & Collate -------------------------


class FusionDataset(Dataset):
    """
    Mixed dataset over all three modalities.

    We flatten:
      - GTEx RNA    -> modality_id = 0
      - CheXpert    -> modality_id = 1
      - IXI MRI     -> modality_id = 2

    Each item:
      (features [D_mod], age, modality_id)
    """

    def __init__(self, df_rna: pd.DataFrame, df_xray: pd.DataFrame, df_mri: pd.DataFrame):
        """
        Concatenate three per-modality DataFrames into a single flat dataset.

        Each DataFrame must have 'features' (list/array) and 'age' (float)
        columns.  Rows with NaN age are dropped.  Modality IDs are assigned
        as 0 (RNA), 1 (X-ray), 2 (MRI).
        """
        self.features = []
        self.ages = []
        self.mod_ids = []

        def _add_block(df, mod_id):
            """Append all rows of *df* to the dataset with the given *mod_id*."""
            if df is None or len(df) == 0:
                return
            df = df[df["age"].notna()].reset_index(drop=True)
            for _, row in df.iterrows():
                feats = np.asarray(row["features"], dtype="float32")
                if not feats.flags.writeable:
                    feats = feats.copy()
                self.features.append(feats)
                self.ages.append(np.float32(row["age"]))
                self.mod_ids.append(mod_id)

        # 0 = RNA, 1 = Xray, 2 = MRI
        _add_block(df_rna, 0)
        _add_block(df_xray, 1)
        _add_block(df_mri, 2)

        if len(self.features) == 0:
            raise ValueError("FusionDataset: no samples found (all modalities empty?)")

    def __len__(self):
        """Return the total number of samples across all modalities."""
        return len(self.ages)

    def __getitem__(self, idx):
        """Return (feature tensor, age tensor, modality_id int) for sample *idx*."""
        x = torch.from_numpy(self.features[idx])
        y = torch.tensor(self.ages[idx], dtype=torch.float32)
        m = int(self.mod_ids[idx])
        return x, y, m


def fusion_collate(batch):
    """
    Custom collate that groups samples by modality.

    Returns a dict:
      {
        0: (x_rna or None,  y_rna or None),
        1: (x_xray or None, y_xray or None),
        2: (x_mri or None,  y_mri or None),
      }
    """
    out = {}

    for mod_id in (0, 1, 2):
        xs = [b[0] for b in batch if b[2] == mod_id]
        ys = [b[1] for b in batch if b[2] == mod_id]
        if xs:
            xs = torch.stack(xs, dim=0)
            ys = torch.stack(ys, dim=0)
            out[mod_id] = (xs, ys)
        else:
            out[mod_id] = (None, None)

    return out


# ------------------------- Model -------------------------


class FusionModel(nn.Module):
    """
    Multimodal age regressor with modality-specific encoders and a shared head.

      RNA:  input_dim_rna  -> enc_rna -> 256
      XRay: input_dim_xray -> enc_xray -> 256
      MRI:  input_dim_mri  -> enc_mri -> 256

      All share a head:
          256 -> 128 -> 1
    """

    def __init__(self, dim_rna: int, dim_xray: int, dim_mri: int, hidden_dim: int = 256):
        """
        Args:
            dim_rna:    Input dimensionality for the RNA encoder, or None/<=0
                        to skip building an RNA encoder.
            dim_xray:   Input dimensionality for the X-ray encoder, or None/<=0.
            dim_mri:    Input dimensionality for the MRI encoder, or None/<=0.
            hidden_dim: Shared hidden dimensionality output by each encoder
                        and consumed by the shared head (default 256).
        """
        super().__init__()

        # Some modalities might be missing; guard dims <= 0
        self.has_rna = dim_rna is not None and dim_rna > 0
        self.has_xray = dim_xray is not None and dim_xray > 0
        self.has_mri = dim_mri is not None and dim_mri > 0

        if self.has_rna:
            self.enc_rna = nn.Sequential(
                nn.Linear(dim_rna, 512),
                nn.GELU(),
                nn.LayerNorm(512),
                nn.Linear(512, hidden_dim),
                nn.GELU(),
                nn.LayerNorm(hidden_dim),
            )
        else:
            self.enc_rna = None

        if self.has_xray:
            self.enc_xray = nn.Sequential(
                nn.Linear(dim_xray, 512),
                nn.GELU(),
                nn.LayerNorm(512),
                nn.Linear(512, hidden_dim),
                nn.GELU(),
                nn.LayerNorm(hidden_dim),
            )
        else:
            self.enc_xray = None

        if self.has_mri:
            self.enc_mri = nn.Sequential(
                nn.Linear(dim_mri, 512),
                nn.GELU(),
                nn.LayerNorm(512),
                nn.Linear(512, hidden_dim),
                nn.GELU(),
                nn.LayerNorm(hidden_dim),
            )
        else:
            self.enc_mri = None

        self.head = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.GELU(),
            nn.LayerNorm(128),
            nn.Linear(128, 1),
        )

    def forward_modality(self, x: torch.Tensor, mod_id: int) -> torch.Tensor:
        """
        Forward a batch for a single modality id: 0=RNA,1=Xray,2=MRI
        Returns shape [B].
        """
        if mod_id == 0:
            if not self.has_rna:
                raise RuntimeError("FusionModel: RNA encoder requested but dim_rna <= 0")
            z = self.enc_rna(x)
        elif mod_id == 1:
            if not self.has_xray:
                raise RuntimeError("FusionModel: X-ray encoder requested but dim_xray <= 0")
            z = self.enc_xray(x)
        elif mod_id == 2:
            if not self.has_mri:
                raise RuntimeError("FusionModel: MRI encoder requested but dim_mri <= 0")
            z = self.enc_mri(x)
        else:
            raise ValueError(f"Unknown modality id {mod_id}")

        out = self.head(z).squeeze(-1)
        return out
        

# ------------------------- Helpers -------------------------


def split_train_val(df: pd.DataFrame, val_fraction: float = 0.15, seed: int = 42):
    """
    Randomly split *df* (after dropping NaN-age rows) into train and val.

    Returns:
        df_train: (1 - val_fraction) fraction of the data.
        df_val:   val_fraction of the data (at least 1 row).
    """
    df = df[df["age"].notna()].reset_index(drop=True)
    if len(df) == 0:
        return df, df
    rng = np.random.default_rng(seed)
    idx = np.arange(len(df))
    rng.shuffle(idx)
    n_val = max(1, int(val_fraction * len(df)))
    val_idx = idx[:n_val]
    train_idx = idx[n_val:]
    df_train = df.iloc[train_idx].reset_index(drop=True)
    df_val = df.iloc[val_idx].reset_index(drop=True)
    return df_train, df_val


def infer_dim(df: pd.DataFrame):
    """
    Infer the feature vector dimensionality from the first row of *df*.
    Returns ``None`` when *df* is None or empty.
    """
    if df is None or len(df) == 0:
        return None
    arr = np.asarray(df.iloc[0]["features"], dtype="float32")
    return int(arr.shape[0])


# ------------------------- Training -------------------------


def train_fusion(
    device: torch.device,
    batch_size: int,
    n_epochs: int,
    run_dir: Path,
):
    """
    Load per-modality parquet files, build the FusionModel with
    modality-specific encoders, train with the custom fusion_collate loader,
    and save the model, loss curves, and a loss plot.

    Args:
        device:     Torch device for training.
        batch_size: Mini-batch size (samples from all modalities combined).
        n_epochs:   Number of training epochs.
        run_dir:    Experiment root directory for saving outputs.
    """
    processed = ROOT / "data" / "processed"

    # Load per-modality unified files
    gtex_path = processed / "unified_gtex.parquet"
    cxp_path = processed / "unified_chexpert.parquet"
    ixi_path = processed / "unified_ixi.parquet"

    df_rna = pd.read_parquet(gtex_path) if gtex_path.exists() else None
    df_xray = pd.read_parquet(cxp_path) if cxp_path.exists() else None
    df_mri = pd.read_parquet(ixi_path) if ixi_path.exists() else None

    if df_rna is None and df_xray is None and df_mri is None:
        raise RuntimeError("No unified_* parquet files found for fusion training.")

    # Train/val split per modality
    df_rna_tr, df_rna_val = split_train_val(df_rna) if df_rna is not None else (None, None)
    df_xray_tr, df_xray_val = split_train_val(df_xray) if df_xray is not None else (None, None)
    df_mri_tr, df_mri_val = split_train_val(df_mri) if df_mri is not None else (None, None)

    # Build datasets
    train_ds = FusionDataset(df_rna_tr, df_xray_tr, df_mri_tr)
    val_ds = FusionDataset(df_rna_val, df_xray_val, df_mri_val)

    # Input dims (from train splits)
    dim_rna = infer_dim(df_rna_tr) if df_rna_tr is not None else None
    dim_xray = infer_dim(df_xray_tr) if df_xray_tr is not None else None
    dim_mri = infer_dim(df_mri_tr) if df_mri_tr is not None else None

    print(f"[Fusion] Modality dims: RNA={dim_rna}, X-ray={dim_xray}, MRI={dim_mri}")

    model = FusionModel(dim_rna, dim_xray, dim_mri, hidden_dim=256).to(device)
    loss_fn = nn.MSELoss(reduction="none")
    optim = torch.optim.AdamW(model.parameters(), lr=1e-3)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=fusion_collate,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=fusion_collate,
    )

    curves = {
        "train_mse": [],
        "val_mse": [],
        "val_mae": [],
    }

    for epoch in range(1, n_epochs + 1):
        # --------- TRAIN ---------
        model.train()
        epoch_loss_sum = 0.0
        epoch_count = 0

        for batch in train_loader:
            optim.zero_grad()

            total_loss = 0.0
            total_count = 0

            for mod_id in (0, 1, 2):
                xb, yb = batch[mod_id]
                if xb is None:
                    continue
                xb = xb.to(device)
                yb = yb.to(device)
                preds = model.forward_modality(xb, mod_id)
                loss_vec = loss_fn(preds, yb)
                total_loss += loss_vec.sum()
                total_count += yb.shape[0]

            if total_count == 0:
                continue

            loss = total_loss / total_count
            loss.backward()
            optim.step()

            epoch_loss_sum += total_loss.item()
            epoch_count += total_count

        train_mse = epoch_loss_sum / max(1, epoch_count)

        # --------- VAL ---------
        model.eval()
        val_sqerrs = []
        val_abserrs = []

        with torch.no_grad():
            for batch in val_loader:
                for mod_id in (0, 1, 2):
                    xb, yb = batch[mod_id]
                    if xb is None:
                        continue
                    xb = xb.to(device)
                    yb = yb.to(device)
                    preds = model.forward_modality(xb, mod_id)
                    sq = (preds - yb) ** 2
                    ae = torch.abs(preds - yb)
                    val_sqerrs.append(sq.detach().cpu().numpy())
                    val_abserrs.append(ae.detach().cpu().numpy())

        if val_sqerrs:
            val_mse = float(np.mean(np.concatenate(val_sqerrs)))
            val_mae = float(np.mean(np.concatenate(val_abserrs)))
        else:
            val_mse = float("nan")
            val_mae = float("nan")

        curves["train_mse"].append(train_mse)
        curves["val_mse"].append(val_mse)
        curves["val_mae"].append(val_mae)

        print(
            f"[Fusion] Epoch {epoch:02d} | "
            f"train MSE={train_mse:.3f} | "
            f"val MSE={val_mse:.3f}, val MAE={val_mae:.3f}"
        )

    # save model
    model_dir = run_dir / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / "fusion_age_mlp.pt"
    torch.save(
        {
            "state_dict": model.state_dict(),
            "dim_rna": dim_rna,
            "dim_xray": dim_xray,
            "dim_mri": dim_mri,
        },
        model_path,
    )
    print(f"[Fusion] Saved model -> {model_path}")

    # save curves
    curves_path = run_dir / "fusion_curves.pkl"
    with open(curves_path, "wb") as f:
        pickle.dump(curves, f)
    print(f"[Fusion] Saved curves -> {curves_path}")

    # quick loss plot
    fig_dir = run_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    epochs = np.arange(1, len(curves["train_mse"]) + 1)

    plt.figure()
    plt.plot(epochs, curves["train_mse"], label="train MSE")
    plt.plot(epochs, curves["val_mse"], label="val MSE")
    plt.xlabel("Epoch")
    plt.ylabel("MSE")
    plt.title("Fusion model loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_dir / "fusion_loss.png")
    plt.close()

    return curves
        

# ------------------------- main -------------------------


def main():
    """
    Parse CLI arguments and invoke train_fusion().
    The run directory (``experiments/run_<run_id>``) is created if absent.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="cuda or cpu",
    )
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument(
        "--run_id",
        type=str,
        default="000",
        help="3-digit run id (must match unimodal run folder)",
    )
    args = parser.parse_args()

    device = torch.device(args.device)
    print(f"Using device: {device}")

    run_dir = ROOT / "experiments" / f"run_{args.run_id}"
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"[Fusion] Using run dir: {run_dir}")

    train_fusion(
        device=device,
        batch_size=args.batch_size,
        n_epochs=args.epochs,
        run_dir=run_dir,
    )


if __name__ == "__main__":
    main()
