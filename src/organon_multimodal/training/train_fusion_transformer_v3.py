"""
train_fusion_transformer_v3.py

Train a multimodal fusion Transformer regressor (v3) that predicts
chronological age from per-modality PCA embeddings (RNA / X-ray / MRI).

Architecture:
  - A shared Projector MLP that maps 64-d modality embeddings into a
    common 64-d aligned space (optionally initialised from a pre-trained
    alignment checkpoint and frozen during fusion training).
  - A learnable modality embedding (nn.Embedding) added to the projected
    vector to inject modality identity.
  - A single-token Transformer encoder that contextualises the combined
    representation.
  - A two-layer regression head that predicts scalar age.

Input:  data/processed/aligned/v3_aligned_base.parquet
Output: models/organon_multimodal/fusion_v3/models/fusion_transformer_<run_id>.pt
        models/organon_multimodal/fusion_v3/curves/fusion_curves_<run_id>.npy
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

DATA = Path("data/processed/aligned")
MODELS = Path("models/organon_multimodal")
MODELS.mkdir(parents=True, exist_ok=True)

OUT = MODELS / "fusion_v3"
OUT.mkdir(parents=True, exist_ok=True)




# ---------------------------------------------------------------------
# Small projector (same structure as in v3 alignment)
# ---------------------------------------------------------------------
class Projector(nn.Module):
    """
    Two-layer MLP that projects a per-modality embedding into a shared space.

    Architecture: in_dim -> hidden -> out_dim (with ReLU activation).
    """

    def __init__(self, in_dim=64, hidden=128, out_dim=64):
        """
        Args:
            in_dim:  Input feature dimensionality (default 64).
            hidden:  Width of the single hidden layer (default 128).
            out_dim: Output dimensionality of the shared space (default 64).
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x):
        """Project input *x* (B, in_dim) -> (B, out_dim)."""
        return self.net(x)


# ---------------------------------------------------------------------
# Fusion Transformer model
# ---------------------------------------------------------------------
class FusionRegressor(nn.Module):
    """
    Multimodal age regressor that fuses per-modality embeddings via a
    Transformer encoder.

    Pipeline (per sample):
      1. Project the raw modality embedding with Projector.
      2. Add a learned modality embedding to inject modality identity.
      3. Linearly project to d_model and pass through a Transformer encoder
         (single token – sequence length = 1).
      4. Regress the [CLS]-like token representation to a scalar age.

    The Projector can optionally be pre-loaded from a contrastive alignment
    checkpoint and frozen so the shared embedding space is preserved.
    """

    def __init__(
        self,
        in_dim=64,
        d_model=128,
        n_heads=4,
        n_layers=2,
        num_modalities=3,
    ):
        """
        Args:
            in_dim:          Dimensionality of per-modality input embeddings.
            d_model:         Internal Transformer model dimension.
            n_heads:         Number of attention heads.
            n_layers:        Number of Transformer encoder layers.
            num_modalities:  Number of distinct modalities (for Embedding table).
        """
        super().__init__()
        self.projector = Projector(in_dim=in_dim, hidden=128, out_dim=64)

        # modality embedding to inject modality info into the aligned space
        self.mod_embed = nn.Embedding(num_modalities, 64)

        self.input_proj = nn.Linear(64, d_model)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=256,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)

        self.head = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

    def forward(self, x, mod_id):
        """
        x: (B, D_in)  – base embeddings (per-modality 64-dim)
        mod_id: (B,)  – int modality IDs {0,1,2}
        """
        # Align into shared space
        z = self.projector(x)  # (B, 64)

        # Add modality embedding
        me = self.mod_embed(mod_id)  # (B, 64)
        z = z + me

        # Single-token transformer (sequence length = 1)
        token = self.input_proj(z).unsqueeze(1)  # (B, 1, d_model)
        h = self.encoder(token)[:, 0, :]        # (B, d_model)

        out = self.head(h).squeeze(1)           # (B,)
        return out


# ---------------------------------------------------------------------
# Data utilities
# ---------------------------------------------------------------------
def build_multimodal_dataset(df: pd.DataFrame):
    """
    Build X, y, mod_id from v3_aligned_base.parquet.
    Uses per-modality 64-d embeddings and maps modalities to {0,1,2}.
    """
    # Normalize modality labels
    mod_map = {
        "rna": "rna",
        "gtex_rna": "rna",
        "xray": "xray",
        "chexpert_xray": "xray",
        "mri": "mri",
        "ixi_mri": "mri",
    }
    df = df.copy()
    df["modality_std"] = df["modality"].map(mod_map)

    prefix_map = {
        "rna": "emb_rna_",
        "xray": "emb_xray_",
        "mri": "emb_mri_",
    }
    mod_id_map = {"rna": 0, "xray": 1, "mri": 2}

    X_list = []
    y_list = []
    modid_list = []

    for mod_std in ["rna", "xray", "mri"]:
        dfm = df[df["modality_std"] == mod_std]
        if dfm.empty:
            print(f"[V3 FUSION] WARNING: no rows for modality '{mod_std}'")
            continue

        prefix = prefix_map[mod_std]
        cols = [c for c in dfm.columns if c.startswith(prefix)]
        if not cols:
            print(f"[V3 FUSION] WARNING: no columns found with prefix '{prefix}'")
            continue

        X = dfm[cols].to_numpy(dtype="float32")
        y = dfm["age"].to_numpy(dtype="float32")

        # Clean NaNs/infs
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)

        X_list.append(X)
        y_list.append(y)
        modid_list.append(
            np.full(len(y), mod_id_map[mod_std], dtype="int64")
        )

        print(
            f"[V3 FUSION] {mod_std}: X shape={X.shape}, y shape={y.shape}, "
            f"mod_id={mod_id_map[mod_std]}"
        )

    if not X_list:
        raise RuntimeError("[V3 FUSION] No modality data found in table.")

    X_all = np.concatenate(X_list, axis=0)
    y_all = np.concatenate(y_list, axis=0)
    mod_all = np.concatenate(modid_list, axis=0)

    # Shuffle consistently
    idx = np.arange(len(y_all))
    np.random.shuffle(idx)
    X_all = X_all[idx]
    y_all = y_all[idx]
    mod_all = mod_all[idx]

    print(
        f"[V3 FUSION] Final dataset: X={X_all.shape}, y={y_all.shape}, mod={mod_all.shape}"
    )
    return X_all, y_all, mod_all


# ---------------------------------------------------------------------
# Main training
# ---------------------------------------------------------------------
def main():
    """
    Parse CLI arguments, load embeddings, build the Transformer fusion model
    (optionally loading and freezing a pre-trained Projector), run the
    training loop, and save the best checkpoint and loss curves.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_id", type=str, default="v3")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("[V3 FUSION] CUDA not available, falling back to CPU.")
        device = "cpu"

    OUT.mkdir(exist_ok=True)
    (OUT / "models").mkdir(parents=True, exist_ok=True)
    (OUT / "curves").mkdir(parents=True, exist_ok=True)

    print("[V3 FUSION] Loading base embeddings...")
    df = pd.read_parquet(DATA / "v3_aligned_base.parquet")
    print(f"[V3 FUSION] Loaded: shape={df.shape}")

    # Build dataset
    X_all, y_all, mod_all = build_multimodal_dataset(df)
    feat_dim = X_all.shape[1]
    print(f"[V3 FUSION] Feature dimension: {feat_dim}")

    X_t = torch.from_numpy(X_all).float()
    y_t = torch.from_numpy(y_all).float()
    mod_t = torch.from_numpy(mod_all).long()

    # Train/val split
    N = len(y_all)
    n_train = int(0.85 * N)
    train_ds = TensorDataset(X_t[:n_train], y_t[:n_train], mod_t[:n_train])
    val_ds = TensorDataset(X_t[n_train:], y_t[n_train:], mod_t[n_train:])

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, drop_last=False
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False, drop_last=False
    )

    # Model
    model = FusionRegressor(in_dim=feat_dim, d_model=128, n_heads=4, n_layers=2).to(
        device
    )

    # Try to load pre-trained projector from alignment_v3.pt
    align_path = OUT / "models" / "alignment_v3.pt"
    if align_path.exists():
        print(f"[V3 FUSION] Loading alignment weights from {align_path}")
        state = torch.load(align_path, map_location=device)
        # state has keys "net.0.weight", etc. – matches Projector().net.*
        model.projector.load_state_dict(state, strict=False)
        # 🔒 Freeze projector so we don't blow up the aligned space
        for p in model.projector.parameters():
            p.requires_grad = False
        print("[V3 FUSION] Projector frozen (no gradients).")
    else:
        print("[V3 FUSION] WARNING: alignment_v3.pt not found; training projector from scratch.")

    # Only train unfrozen parameters (Transformer + head [+ projector if not frozen])
    trainable_params = [p for p in model.parameters() if p.requires_grad]

    # 🔧 Safer learning rate
    optimizer = torch.optim.AdamW(trainable_params, lr=3e-5, weight_decay=1e-4)
    mse_loss = nn.MSELoss()

    train_curve = []
    val_mse_curve = []
    val_mae_curve = []

    print("[V3 FUSION] Starting training...")
    best_val_mse = float("inf")
    best_state = None

    for epoch in range(1, args.epochs + 1):
        model.train()
        batch_losses = []

        for xb, yb, mb in train_loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            mb = mb.to(device, non_blocking=True)

            optimizer.zero_grad()
            pred = model(xb, mb)

            # Clean any NaN in prediction/target
            pred = torch.nan_to_num(pred, nan=0.0, posinf=0.0, neginf=0.0)
            yb = torch.nan_to_num(yb, nan=0.0, posinf=0.0, neginf=0.0)

            loss = mse_loss(pred, yb)
            if torch.isnan(loss) or torch.isinf(loss):
                print("[V3 FUSION] Warning: NaN/inf loss encountered; skipping batch.")
                continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=5.0)
            optimizer.step()

            batch_losses.append(loss.item())

        train_mse = float(np.mean(batch_losses)) if batch_losses else float("nan")
        train_curve.append(train_mse)

        # Validation
        model.eval()
        val_losses = []
        val_abs = []
        with torch.no_grad():
            for xb, yb, mb in val_loader:
                xb = xb.to(device, non_blocking=True)
                yb = yb.to(device, non_blocking=True)
                mb = mb.to(device, non_blocking=True)

                pred = model(xb, mb)
                pred = torch.nan_to_num(pred, nan=0.0, posinf=0.0, neginf=0.0)
                yb = torch.nan_to_num(yb, nan=0.0, posinf=0.0, neginf=0.0)

                loss = mse_loss(pred, yb)
                val_losses.append(loss.item())
                val_abs.append(torch.mean(torch.abs(pred - yb)).item())

        val_mse = float(np.mean(val_losses))
        val_mae = float(np.mean(val_abs))
        val_mse_curve.append(val_mse)
        val_mae_curve.append(val_mae)

        print(
            f"[V3 FUSION] Epoch {epoch:02d} | "
            f"train MSE={train_mse:.4f} | val MSE={val_mse:.4f}, val MAE={val_mae:.4f}"
        )

        # Track best model
        if val_mse < best_val_mse:
            best_val_mse = val_mse
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}

        # 🔒 Safety: if it ever explodes above 1000, bail out early

    # Save best model (not last)
    model_path = OUT / "models" / f"fusion_transformer_{args.run_id}.pt"
    if best_state is not None:
        torch.save(best_state, model_path)
        print("[V3 FUSION] Saved BEST model ->", model_path)
    else:
        torch.save(model.state_dict(), model_path)
        print("[V3 FUSION] Saved LAST model ->", model_path)

    curves = {
        "train_mse": train_curve,
        "val_mse": val_mse_curve,
        "val_mae": val_mae_curve,
        "best_val_mse": best_val_mse,
    }
    curves_path = OUT / "curves" / f"fusion_curves_{args.run_id}.npy"
    np.save(curves_path, curves, allow_pickle=True)
    print("[V3 FUSION] Saved curves ->", curves_path)


if __name__ == "__main__":
    main()
