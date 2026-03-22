"""
v4_5_compute_latent_ig.py
==========================
Compute *latent-space* Integrated Gradient (IG) importance vectors for each
organ using the v4.5 ``CrossFusionV45`` model.

For every organ that has RNA rows in the aligned embedding table the script:
  1. Extracts the ``z_rna_*`` embedding columns as the input feature space.
  2. Computes the mean absolute gradient of the predicted age (mu) with
     respect to the input embedding, averaged over up to ``--max_per_organ``
     samples.  This yields a ``(D,)`` importance vector in latent space.
  3. Saves the vector as ``analysis/v4_5_ig_latent/latent_ig_<organ>.npy``.

These per-organ ``.npy`` vectors are consumed by ``v4_5_ig_to_genes.py``
and ``v4_5_map_latent_to_genes.py`` to lift importance back to gene space
via the RNA projector weight matrix.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from v4_5_crossfusion_model import CrossFusionV45


def get_project_root() -> Path:
    """
    Return the project root directory (two levels above this file).

    Assumes the file lives at ``<root>/src/vitalis_organage/``.

    Returns
    -------
    Path
        Absolute path to the repository root.
    """
    # src/vitalis_organage/ -> root is two levels up
    return Path(__file__).resolve().parents[2]


def build_organ_mapping(df: pd.DataFrame):
    """
    Build a sorted organ-name to integer-ID mapping from the dataset.

    The mapping must be identical to the one used during model training so
    that the organ embedding table is indexed correctly.

    Parameters
    ----------
    df : pd.DataFrame
        Full aligned embedding table containing an ``organ`` column.

    Returns
    -------
    tuple[list[str], dict[str, int]]
        ``(organs, organ_to_id)`` where ``organs`` is the sorted list of
        unique organ names and ``organ_to_id`` maps each name to its integer
        index.
    """
    organs = sorted(df["organ"].dropna().unique())
    organ_to_id = {org: i for i, org in enumerate(organs)}
    print("[DATA] Organs mapping:", organ_to_id)
    return organs, organ_to_id


def load_model(device: torch.device, n_organs: int) -> torch.nn.Module:
    """
    Rebuild CrossFusionV45 exactly as in training and load the v4.5 weights.
    """
    root = get_project_root()
    ckpt_path = root / "models" / "v4_5" / "fusion_cross_v4_5.pt"

    print(f"[MODEL] Initializing CrossFusionV45 with n_organs={n_organs}")
    model = CrossFusionV45(
        emb_dim=256,
        organ_dim=64,
        n_organs=n_organs,
        d_model=256,
    ).to(device)

    print(f"[MODEL] Loading state_dict from {ckpt_path}")
    state_dict = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def compute_latent_ig_for_organ(
    model: torch.nn.Module,
    device: torch.device,
    organ_name: str,
    X_np: np.ndarray,
    organ_id: int,
    batch_size: int = 256,
) -> np.ndarray:
    """
    X_np: (N, D) numpy array of RNA embeddings for this organ.
    Returns: (D,) IG-like vector = mean |grad mu / grad X| over samples.
    """
    N, D = X_np.shape
    print(f"[IG] Organ={organ_name} | N={N}, D={D}, organ_id={organ_id}")

    X_tensor = torch.from_numpy(X_np).float()
    organ_tensor = torch.full((N,), organ_id, dtype=torch.long)

    ds = TensorDataset(X_tensor, organ_tensor)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False)

    grad_accum = torch.zeros(D, device=device)

    for xb, ob in tqdm(loader, desc=f"[IG] {organ_name}", leave=False):
        xb = xb.to(device)
        ob = ob.to(device)

        xb.requires_grad_(True)

        # Forward pass: model(x, organ_id) -> mu, sigma
        mu, sigma = model(xb, ob)  # shapes (B,), (B,)
        if mu.ndim > 1:
            mu = mu.squeeze(-1)

        # Use mean predicted age as scalar target
        loss = mu.mean()

        model.zero_grad()
        if xb.grad is not None:
            xb.grad.zero_()

        loss.backward()

        grads = xb.grad  # (B, D)
        grad_accum += grads.abs().sum(dim=0)

    grad_mean = (grad_accum / N).detach().cpu().numpy()
    return grad_mean


def main():
    """
    Command-line entry point for latent IG computation.

    Loads the aligned embedding table, initialises the model, and for each
    organ with RNA data runs ``compute_latent_ig_for_organ`` and saves the
    resulting importance vector as a ``.npy`` file.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data",
        type=str,
        default="data/processed/aligned/v3_aligned_contrastive.parquet",
        help="Aligned contrastive parquet used for v4.5 training.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=256,
        help="Batch size for IG computation.",
    )
    parser.add_argument(
        "--max_per_organ",
        type=int,
        default=5000,
        help="Max RNA samples per organ used for IG.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="cuda or cpu",
    )
    args = parser.parse_args()

    device = torch.device(
        "cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu"
    )
    if args.device == "cuda" and device.type != "cuda":
        print("[WARN] CUDA requested but not available; using CPU.")

    root = get_project_root()
    data_path = root / args.data
    print(f"[ROOT] Project root: {root}")
    print(f"[DATA] Loading {data_path}")

    df = pd.read_parquet(data_path)

    # Recreate organ mapping exactly like training
    organs, organ_to_id = build_organ_mapping(df)

    # Filter to RNA rows only
    df_rna = df[df["modality"] == "rna"].copy()
    if df_rna.empty:
        raise RuntimeError("[DATA] No RNA rows found in dataset.")

    # Feature columns
    feat_cols = [c for c in df_rna.columns if c.startswith("z_rna_")]
    if not feat_cols:
        raise RuntimeError("[DATA] No 'z_rna_*' columns found in RNA subset.")

    print(f"[DATA] Using {len(feat_cols)} RNA feature columns: {feat_cols[:5]}...")

    # Initialize model
    model = load_model(device, n_organs=len(organ_to_id))

    # Output dir for latent IG vectors
    out_dir = root / "analysis" / "v4_5_ig_latent"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[OUT] Saving latent IG vectors to {out_dir}")

    # Loop over organs
    for organ in organs:
        # Only consider organs that appear in RNA rows
        df_o = df_rna[df_rna["organ"] == organ].copy()
        if df_o.empty:
            print(f"[DATA] No RNA rows for organ '{organ}', skipping.")
            continue

        # Drop any rows with non-finite features or age
        X = df_o[feat_cols].to_numpy(dtype="float32")
        age = df_o["age"].to_numpy(dtype="float32")

        mask_X = np.isfinite(X).all(axis=1)
        mask_y = np.isfinite(age)
        mask = mask_X & mask_y

        if mask.sum() == 0:
            print(f"[DATA] All rows invalid for organ '{organ}', skipping.")
            continue

        X = X[mask]

        # Optional subsampling for speed
        if len(X) > args.max_per_organ:
            idx = np.random.choice(len(X), size=args.max_per_organ, replace=False)
            X = X[idx]

        organ_id = organ_to_id[organ]

        ig_vec = compute_latent_ig_for_organ(
            model=model,
            device=device,
            organ_name=organ,
            X_np=X,
            organ_id=organ_id,
            batch_size=args.batch_size,
        )

        out_path = out_dir / f"latent_ig_{organ}.npy"
        np.save(out_path, ig_vec)
        print(f"[OUT] Saved latent IG for organ={organ} -> {out_path}")


if __name__ == "__main__":
    main()
