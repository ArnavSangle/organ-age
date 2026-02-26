# scripts/73_train_contrastive_alignment.py
"""
Train simple contrastive / alignment projectors that map each modality
(RNA / X-ray / MRI) into a shared embedding space.

This version is **defensive**:
- It never mixes feature vectors of different dimensionality in one batch.
- It works even though GTEx (RNA) has 59k-dim vectors while imaging has 1024-dim.
- It saves per-run training curves under: experiments/run_<ID>/contrastive_curves.pkl
- It saves projector weights under: models/run_<ID>/contrastive_projectors.pt

Called from 90_run_experiments.py like:

  python -u scripts/73_train_contrastive_alignment.py \
      --device cuda \
      --batch_size 256 \
      --epochs 8 \
      --run_id 001
"""
import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

ROOT = Path(".")


# ------------------------------ Dataset ------------------------------


class SimpleAgeDataset(Dataset):
    """
    One-modality dataset.

    Expects a DataFrame with columns:
      - 'features': list/array of floats (same length for all rows)
      - 'age': float (no NaNs; we filter them out).
    """

    def __init__(self, df: pd.DataFrame, max_samples: int | None = None):
        """
        Args:
            df:          DataFrame with 'features' and 'age' columns.
                         Rows with NaN age are dropped automatically.
            max_samples: If given, randomly sub-sample to at most this many
                         rows (using random_state=42 for reproducibility).
        """
        df = df.copy()
        df = df[df["age"].notna()].reset_index(drop=True)
        if max_samples is not None and len(df) > max_samples:
            df = df.sample(n=max_samples, random_state=42).reset_index(drop=True)

        self.ages = df["age"].astype("float32").to_numpy()
        self.features = df["features"].tolist()

        # Cache dimensionality
        first = self._coerce_array(self.features[0])
        self.input_dim = int(first.shape[0])

    @staticmethod
    def _coerce_array(x):
        """Return *x* as a writable float32 numpy array."""
        if isinstance(x, np.ndarray):
            arr = x
        else:
            arr = np.asarray(x, dtype="float32")
        # Make a writable copy to avoid PyTorch warning about non-writable arrays
        if not arr.flags.writeable:
            arr = arr.copy()
        return arr.astype("float32")

    def __len__(self):
        """Return the number of samples."""
        return len(self.ages)

    def __getitem__(self, idx: int):
        """Return (feature tensor, age tensor) for sample at *idx*."""
        x = self._coerce_array(self.features[idx])
        y = self.ages[idx]
        # Important: copy to make sure each tensor is independent and writable
        return torch.from_numpy(x.copy()), torch.tensor(y, dtype=torch.float32)


# ------------------------------ Models ------------------------------


class Projector(nn.Module):
    """
    Simple MLP projector: input_dim -> 512 -> proj_dim
    """

    def __init__(self, input_dim: int, proj_dim: int = 128):
        """
        Args:
            input_dim: Dimensionality of the raw modality features.
            proj_dim:  Dimensionality of the shared projection space
                       (default 128).
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.GELU(),
            nn.LayerNorm(512),
            nn.Linear(512, proj_dim),
        )

    def forward(self, x):
        """Project input *x* (B, input_dim) -> (B, proj_dim)."""
        return self.net(x)


# --------------------------- Train function --------------------------


def train_alignment(
    device: torch.device,
    batch_size: int,
    n_epochs: int,
    run_id: str,
):
    """
    Align RNA / X-ray / MRI distributions in a shared space by matching their
    batch-level means:

      L = ||μ_rna − μ_xray||^2 + ||μ_rna − μ_mri||^2 + ||μ_xray − μ_mri||^2

    where μ_* are mean embeddings of each batch for each modality.
    """

    processed = ROOT / "data" / "processed"

    # ----- Load unified per-modality tables -----
    gtex_path = processed / "unified_gtex.parquet"
    cxp_path = processed / "unified_chexpert.parquet"
    ixi_path = processed / "unified_ixi.parquet"

    if not (gtex_path.exists() and cxp_path.exists() and ixi_path.exists()):
        print("[contrastive] One or more unified_* files missing, skipping.")
        return

    df_rna = pd.read_parquet(gtex_path)
    df_xray = pd.read_parquet(cxp_path)
    df_mri = pd.read_parquet(ixi_path)

    print(
        f"[contrastive] Loaded unified tables:"
        f" RNA={df_rna.shape}, X-ray={df_xray.shape}, MRI={df_mri.shape}"
    )

    # ----- Build datasets & loaders (separate per modality) -----
    ds_rna = SimpleAgeDataset(df_rna)
    ds_xray = SimpleAgeDataset(df_xray)
    ds_mri = SimpleAgeDataset(df_mri)

    print(
        f"[contrastive] Dataset sizes:"
        f" RNA={len(ds_rna)}, X-ray={len(ds_xray)}, MRI={len(ds_mri)}"
    )

    loader_rna = DataLoader(
        ds_rna, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True
    )
    loader_xray = DataLoader(
        ds_xray, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True
    )
    loader_mri = DataLoader(
        ds_mri, batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True
    )

    # ----- Projectors -----
    proj_dim = 128
    proj_rna = Projector(ds_rna.input_dim, proj_dim).to(device)
    proj_xray = Projector(ds_xray.input_dim, proj_dim).to(device)
    proj_mri = Projector(ds_mri.input_dim, proj_dim).to(device)

    params = list(proj_rna.parameters()) + list(proj_xray.parameters()) + list(
        proj_mri.parameters()
    )
    optim = torch.optim.AdamW(params, lr=1e-3, weight_decay=1e-4)

    # Where to save things for this run
    run_dir = ROOT / "experiments" / f"run_{run_id}"
    run_dir.mkdir(parents=True, exist_ok=True)
    model_dir = ROOT / "models" / f"run_{run_id}"
    model_dir.mkdir(parents=True, exist_ok=True)

    train_curve = []

    for epoch in range(1, n_epochs + 1):
        proj_rna.train()
        proj_xray.train()
        proj_mri.train()

        # Re-create iterators each epoch
        it_rna = iter(loader_rna)
        it_xray = iter(loader_xray)
        it_mri = iter(loader_mri)

        # Number of steps = min length among loaders
        n_steps = min(len(loader_rna), len(loader_xray), len(loader_mri))
        epoch_losses = []

        for _ in range(n_steps):
            try:
                xb_rna, _ = next(it_rna)
                xb_xray, _ = next(it_xray)
                xb_mri, _ = next(it_mri)
            except StopIteration:
                break

            xb_rna = xb_rna.to(device)
            xb_xray = xb_xray.to(device)
            xb_mri = xb_mri.to(device)

            optim.zero_grad()

            z_rna = proj_rna(xb_rna)
            z_xray = proj_xray(xb_xray)
            z_mri = proj_mri(xb_mri)

            mu_rna = z_rna.mean(dim=0)
            mu_xray = z_xray.mean(dim=0)
            mu_mri = z_mri.mean(dim=0)

            loss = (
                (mu_rna - mu_xray).pow(2).mean()
                + (mu_rna - mu_mri).pow(2).mean()
                + (mu_xray - mu_mri).pow(2).mean()
            )

            loss.backward()
            optim.step()

            epoch_losses.append(float(loss.detach().cpu().item()))

        if epoch_losses:
            epoch_loss = float(np.mean(epoch_losses))
        else:
            epoch_loss = float("nan")

        train_curve.append(epoch_loss)
        print(f"[contrastive] Epoch {epoch:02d} | align loss={epoch_loss:.4f}")

    # ----- Save artifacts -----
    # Training curve
    curves_path = run_dir / "contrastive_curves.pkl"
    with curves_path.open("wb") as f:
        pickle.dump(
            {
                "epochs": list(range(1, n_epochs + 1)),
                "align_loss": train_curve,
            },
            f,
        )
    print(f"[contrastive] Saved curves -> {curves_path}")

    # Projector weights
    ckpt = {
        "proj_dim": proj_dim,
        "rna_input_dim": ds_rna.input_dim,
        "xray_input_dim": ds_xray.input_dim,
        "mri_input_dim": ds_mri.input_dim,
        "proj_rna": proj_rna.state_dict(),
        "proj_xray": proj_xray.state_dict(),
        "proj_mri": proj_mri.state_dict(),
    }
    ckpt_path = model_dir / "contrastive_projectors.pt"
    torch.save(ckpt, ckpt_path)
    print(f"[contrastive] Saved projectors -> {ckpt_path}")


# ------------------------------- main -------------------------------


def main():
    """
    Parse CLI arguments and call train_alignment().
    Projector weights are saved to ``models/run_<run_id>/contrastive_projectors.pt``
    and training curves to ``experiments/run_<run_id>/contrastive_curves.pkl``.
    """
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="cuda or cpu",
    )
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--run_id", type=str, default="000")
    args = ap.parse_args()

    device = torch.device(args.device)
    print(f"Using device: {device}")

    train_alignment(
        device=device,
        batch_size=args.batch_size,
        n_epochs=args.epochs,
        run_id=args.run_id,
    )


if __name__ == "__main__":
    main()
