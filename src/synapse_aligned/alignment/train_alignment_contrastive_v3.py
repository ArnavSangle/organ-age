"""
train_alignment_contrastive_v3.py - Contrastive multimodal alignment (v3).

Trains three modality-specific Projector networks (RNA, X-ray, MRI) using an
InfoNCE (contrastive) objective so that matched samples from different
modalities are pulled close together in a shared 256-dimensional embedding
space.

Workflow
--------
1. Load the pre-computed PCA embeddings from ``v3_aligned_base.parquet``.
2. Build per-modality DataLoaders and instantiate one Projector per modality.
3. Optimise all three projectors jointly using the sum of pairwise InfoNCE
   losses over mini-batches (RNA-Xray, RNA-MRI, Xray-MRI).
4. Save the training loss curve, the projector weights, and a new parquet file
   (``v3_aligned_contrastive.parquet``) that appends the aligned ``z_*``
   embeddings to the original rows.

Outputs (written under ``v3_output/``)
---------------------------------------
- ``data/contrastive_loss_v3.npy``      - per-epoch mean InfoNCE loss.
- ``models/alignment_contrastive_v3.pt`` - state-dicts for all three projectors.
- ``data/v3_aligned_contrastive.parquet`` - original table + aligned z_* columns.
"""

import pandas as pd
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from itertools import cycle
from pathlib import Path

OUT = Path("v3_output")
DATA = OUT / "data"
MODELS = OUT / "models"


def load_mod(df: pd.DataFrame, modality_name: str, emb_prefix: str):
    """
    Filter df by modality and load the embedding block for that modality.

    Returns:
      - df_mod: filtered dataframe (only this modality)
      - X: tensor of shape (N_mod, D)
      - age: tensor of shape (N_mod,)
    """
    df_mod = df[df["modality"] == modality_name].copy()
    if df_mod.empty:
        raise RuntimeError(f"[V3 CONTRASTIVE] No rows found for modality '{modality_name}'")

    cols = [c for c in df_mod.columns if c.startswith(emb_prefix)]
    if not cols:
        raise RuntimeError(
            f"[V3 CONTRASTIVE] No columns with prefix '{emb_prefix}' for modality '{modality_name}'"
        )

    X = df_mod[cols].values.astype("float32")
    age = df_mod["age"].values.astype("float32")

    # Tiny safety net; PCA output should already be finite
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    print(
        f"[V3 CONTRASTIVE] {modality_name}: "
        f"X shape = {X.shape}, emb_prefix = '{emb_prefix}', rows = {len(df_mod)}"
    )

    return df_mod, torch.tensor(X), torch.tensor(age)


class Projector(nn.Module):
    """
    Modality-specific linear projector with LayerNorm for contrastive alignment.

    Maps a raw embedding from one modality into a shared, normalised space so
    that InfoNCE can be applied across modalities.

    Architecture: Linear -> LayerNorm -> ReLU -> Linear -> LayerNorm
    """

    def __init__(self, D_in, D=256):
        """
        Initialise the projector.

        Parameters
        ----------
        D_in : int
            Dimensionality of the input embedding (e.g. 64 for PCA-reduced data).
        D : int, optional
            Dimensionality of the output shared space. Defaults to 256.
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(D_in, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, D),
            nn.LayerNorm(D),
        )

    def forward(self, x):
        """
        Project input embeddings into the shared space.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape ``(N, D_in)``.

        Returns
        -------
        torch.Tensor
            Projected tensor of shape ``(N, D)``.
        """
        return self.net(x)


def info_nce(z1, z2, temp=0.1):
    """
    Numerically-stable InfoNCE for one-to-one pairs (z1[i] ↔ z2[i]).
    """
    eps = 1e-8
    z1 = z1 / (z1.norm(dim=1, keepdim=True) + eps)
    z2 = z2 / (z2.norm(dim=1, keepdim=True) + eps)

    logits = z1 @ z2.T / temp  # (N, N) if batch sizes match
    logits = torch.clamp(logits, -30.0, 30.0)

    labels = torch.arange(z1.size(0), device=z1.device)
    loss = nn.CrossEntropyLoss()(logits, labels)
    return loss


def main():
    """
    Entry point for contrastive alignment training.

    Loads per-modality embeddings, trains three Projectors jointly using
    pairwise InfoNCE losses, and writes the aligned embeddings and model
    weights to disk.
    """
    print("[V3 CONTRASTIVE] Loading base embeddings...")
    df = pd.read_parquet(DATA / "v3_aligned_base.parquet")
    print("[V3 CONTRASTIVE] df shape:", df.shape)
    print("[V3 CONTRASTIVE] modalities:\n", df["modality"].value_counts())

    # Load per-modality blocks
    df_rna, Xr, Ar = load_mod(df, "rna",  "emb_rna_")
    df_xray, Xx, Ax = load_mod(df, "xray", "emb_xray_")
    df_mri, Xm, Am = load_mod(df, "mri",  "emb_mri_")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[V3 CONTRASTIVE] Using device: {device}")

    B = 512  # batch size
    dr = DataLoader(TensorDataset(Xr, Ar), batch_size=B, shuffle=True, drop_last=True)
    dx = DataLoader(TensorDataset(Xx, Ax), batch_size=B, shuffle=True, drop_last=True)
    dm = DataLoader(TensorDataset(Xm, Am), batch_size=B, shuffle=True, drop_last=True)

    Pr = Projector(64).to(device)
    Px = Projector(64).to(device)
    Pm = Projector(64).to(device)

    opt = torch.optim.AdamW(
        list(Pr.parameters()) + list(Px.parameters()) + list(Pm.parameters()),
        lr=3e-5,
        weight_decay=1e-4,
    )

    print("[V3 CONTRASTIVE] Training (500 epochs)...")
    max_epochs = 500
    epoch_losses = []

    for epoch in range(max_epochs):
        losses = []

        for (rb, _), (xb, _), (mb, _) in zip(cycle(dr), cycle(dx), dm):
            # Make sure batch sizes match across modalities
            n = min(rb.size(0), xb.size(0), mb.size(0))
            rb = rb[:n].to(device)
            xb = xb[:n].to(device)
            mb = mb[:n].to(device)

            zr, zx, zm = Pr(rb), Px(xb), Pm(mb)

            loss = (
                info_nce(zr, zx) +
                info_nce(zr, zm) +
                info_nce(zx, zm)
            )

            if not torch.isfinite(loss):
                print("[V3 CONTRASTIVE] Non-finite loss detected, stopping training.")
                print("  loss:", loss)
                break

            opt.zero_grad()
            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                list(Pr.parameters()) + list(Px.parameters()) + list(Pm.parameters()),
                max_norm=1.0,
            )

            opt.step()
            losses.append(loss.item())

        if not losses:
            print(f"[V3 CONTRASTIVE] No valid steps in epoch {epoch+1}, stopping early.")
        mean_loss = float(np.mean(losses))
        epoch_losses.append(mean_loss)
        print(f"[epoch {epoch+1:03d}] loss={mean_loss:.4f}")

    # Save loss curve for plotting later
    DATA.mkdir(parents=True, exist_ok=True)
    loss_path = DATA / "contrastive_loss_v3.npy"
    np.save(loss_path, np.array(epoch_losses, dtype=np.float32))
    print("[V3 CONTRASTIVE] Saved loss curve ->", loss_path)

    MODELS.mkdir(exist_ok=True, parents=True)
    model_path = MODELS / "alignment_contrastive_v3.pt"
    torch.save(
        {"rna": Pr.state_dict(), "xray": Px.state_dict(), "mri": Pm.state_dict()},
        model_path,
    )
    print("[V3 CONTRASTIVE] Saved projector weights ->", model_path)

    print("[V3 CONTRASTIVE] Saving aligned embeddings...")
    with torch.no_grad():
        Zr = Pr(Xr.to(device)).cpu().numpy()
        Zx = Px(Xx.to(device)).cpu().numpy()
        Zm = Pm(Xm.to(device)).cpu().numpy()

    # Build aligned embedding blocks in one shot to avoid fragmentation
    rna_cols = {f"z_rna_{i}": Zr[:, i] for i in range(Zr.shape[1])}
    xray_cols = {f"z_xray_{i}": Zx[:, i] for i in range(Zx.shape[1])}
    mri_cols = {f"z_mri_{i}": Zm[:, i] for i in range(Zm.shape[1])}

    df_rna_out = pd.concat(
        [df_rna.reset_index(drop=True), pd.DataFrame(rna_cols)], axis=1
    )
    df_xray_out = pd.concat(
        [df_xray.reset_index(drop=True), pd.DataFrame(xray_cols)], axis=1
    )
    df_mri_out = pd.concat(
        [df_mri.reset_index(drop=True), pd.DataFrame(mri_cols)], axis=1
    )

    # Concatenate back to a single table
    df_out = pd.concat([df_rna_out, df_xray_out, df_mri_out], axis=0)
    df_out = df_out.sort_index()  # optional: preserve original row order

    out_path = DATA / "v3_aligned_contrastive.parquet"
    df_out.to_parquet(out_path)
    print("[V3 CONTRASTIVE] Saved aligned table ->", out_path)
    print("[V3 CONTRASTIVE] Done.")


if __name__ == "__main__":
    main()
