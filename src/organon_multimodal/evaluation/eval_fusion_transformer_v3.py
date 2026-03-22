"""
eval_fusion_transformer_v3.py

Evaluate a trained v3 FusionRegressor on the full aligned dataset and report
overall and per-modality MSE / MAE metrics.

Usage:
    python eval_fusion_transformer_v3.py --run_id <id> [--device cuda|cpu]

Loads:
    data/processed/aligned/v3_aligned_base.parquet
    models/organon_multimodal/fusion_v3/models/fusion_transformer_<run_id>.pt

Prints overall and per-modality (RNA / X-ray / MRI) MSE and MAE to stdout.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

# ---------------- Paths ----------------
DATA = Path("data/processed/aligned")

MODELS_ROOT = Path("models/organon_multimodal")
FUSION_DIR = MODELS_ROOT / "fusion_v3"
MODEL_DIR = FUSION_DIR / "models"  # matches training output: .../fusion_v3/models/...
MODEL_DIR.mkdir(parents=True, exist_ok=True)


# ---- same Projector / FusionRegressor as in training ----
class Projector(nn.Module):
    """
    Two-layer MLP projector (mirrors train_fusion_transformer_v3.Projector).

    Maps a per-modality embedding from *in_dim* to *out_dim* via one hidden
    layer with ReLU activation.
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


class FusionRegressor(nn.Module):
    """
    Reconstruction of the training-time FusionRegressor for inference only.

    Identical architecture to the one in train_fusion_transformer_v3.py;
    kept here so the evaluation script has no import dependency on the
    training script.
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
        Args:
            x:      (B, in_dim) float tensor of modality embeddings.
            mod_id: (B,) long tensor of modality indices {0, 1, 2}.

        Returns:
            Predicted age scalar per sample, shape (B,).
        """
        z = self.projector(x)                     # (B, 64)
        z = z + self.mod_embed(mod_id)           # modality info
        token = self.input_proj(z).unsqueeze(1)  # (B,1,d_model)
        h = self.encoder(token)[:, 0, :]         # (B,d_model)
        out = self.head(h).squeeze(1)            # (B,)
        return out


def build_dataset(df: pd.DataFrame):
    """
    Construct evaluation arrays from the v3 aligned base parquet.

    Returns:
        X_all      : (N, D) float32 feature matrix (one modality block per row).
        y_all      : (N,) float32 age array.
        modid_all  : (N,) int64 modality index (0=RNA, 1=X-ray, 2=MRI).
        modname_all: (N,) object array of standardised modality name strings.

    NaN/inf values in features and ages are replaced with 0. Order is
    preserved (no shuffle) to allow correct per-modality indexing.
    """
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

    prefix_map = {"rna": "emb_rna_", "xray": "emb_xray_", "mri": "emb_mri_"}
    mod_id_map = {"rna": 0, "xray": 1, "mri": 2}

    X_list, y_list, modid_list, modname_list = [], [], [], []

    for mod_std in ["rna", "xray", "mri"]:
        dfm = df[df["modality_std"] == mod_std]
        if dfm.empty:
            continue
        cols = [c for c in dfm.columns if c.startswith(prefix_map[mod_std])]
        if not cols:
            continue

        X = dfm[cols].to_numpy(dtype="float32")
        y = dfm["age"].to_numpy(dtype="float32")
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)

        X_list.append(X)
        y_list.append(y)
        mid = np.full(len(y), mod_id_map[mod_std], dtype="int64")
        X_mod_names = np.full(len(y), mod_std, dtype=object)

        modid_list.append(mid)
        modname_list.append(X_mod_names)

    X_all = np.concatenate(X_list, axis=0)
    y_all = np.concatenate(y_list, axis=0)
    modid_all = np.concatenate(modid_list, axis=0)
    modname_all = np.concatenate(modname_list, axis=0)

    # No shuffle here – we want to keep mapping for per-modality analysis
    return X_all, y_all, modid_all, modname_all


def main(args):
    """
    Load the trained FusionRegressor, run inference on the full aligned
    dataset, and print overall and per-modality MSE / MAE metrics.
    """
    # ---------------- Device ----------------
    if args.device == "cuda" and not torch.cuda.is_available():
        print("[V3 EVAL] cuda requested but not available, falling back to cpu")
        device = "cpu"
    else:
        device = args.device

    print("[V3 EVAL] Using device:", device)

    # ---------------- Data ----------------
    df = pd.read_parquet(DATA / "v3_aligned_base.parquet")
    print("[V3 EVAL] Loaded base table:", df.shape)

    X_all, y_all, modid_all, modname_all = build_dataset(df)
    print("[V3 EVAL] Dataset shapes:",
          "X", X_all.shape, "y", y_all.shape, "modid", modid_all.shape)

    X_t = torch.from_numpy(X_all).float().to(device)
    y_t = torch.from_numpy(y_all).float().to(device)
    modid_t = torch.from_numpy(modid_all).long().to(device)

    ds = TensorDataset(X_t, y_t, modid_t)
    loader = DataLoader(ds, batch_size=1024, shuffle=False)

    # ---------------- Model ----------------
    model = FusionRegressor(in_dim=X_all.shape[1]).to(device)

    model_path = MODEL_DIR / f"fusion_transformer_{args.run_id}.pt"
    print(f"[V3 EVAL] Loading model from: {model_path}")

    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)
    model.eval()

    mse_loss = nn.MSELoss(reduction="sum")

    # overall metrics
    total_sq_err = 0.0
    total_abs_err = 0.0
    total_n = 0

    # per-modality buffers
    per_mod_sq = {"rna": 0.0, "xray": 0.0, "mri": 0.0}
    per_mod_abs = {"rna": 0.0, "xray": 0.0, "mri": 0.0}
    per_mod_n = {"rna": 0, "xray": 0, "mri": 0}

    with torch.no_grad():
        offset = 0
        for xb, yb, mb in loader:
            bsz = xb.size(0)
            preds = model(xb, mb)

            preds = torch.nan_to_num(preds, nan=0.0, posinf=0.0, neginf=0.0)
            yb = torch.nan_to_num(yb, nan=0.0, posinf=0.0, neginf=0.0)

            sq_err = (preds - yb) ** 2
            abs_err = torch.abs(preds - yb)

            total_sq_err += sq_err.sum().item()
            total_abs_err += abs_err.sum().item()
            total_n += bsz

            # per modality using original numpy modname_all
            names_batch = modname_all[offset:offset + bsz]
            offset += bsz
            for i, name in enumerate(names_batch):
                name = str(name)
                if name not in per_mod_sq:
                    continue
                per_mod_sq[name] += sq_err[i].item()
                per_mod_abs[name] += abs_err[i].item()
                per_mod_n[name] += 1

    overall_mse = total_sq_err / total_n
    overall_mae = total_abs_err / total_n

    print("\n=== Overall Fusion Metrics (v3) ===")
    print(f"MSE = {overall_mse:.3f}")
    print(f"MAE = {overall_mae:.3f}")

    print("\n=== Per-modality Metrics (v3) ===")
    for name in ["rna", "xray", "mri"]:
        n = per_mod_n[name]
        if n == 0:
            print(f"{name}: no samples")
            continue
        mse = per_mod_sq[name] / n
        mae = per_mod_abs[name] / n
        print(f"{name:4s} | N={n:7d} | MSE={mse:8.3f} | MAE={mae:6.3f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_id", type=str, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()
    main(args)
