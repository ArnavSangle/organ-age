"""
eval_compare_versions.py - Side-by-side evaluation of v3 baseline vs v3.5 cross-fusion.

Loads both the v3 ``FusionRegressor`` (modality-token transformer with learned
modality embeddings) and the v3.5 ``CrossFusion`` (contrastive-aligned,
single-token transformer) and evaluates each on its corresponding aligned
dataset.

Metrics (MSE and MAE) are reported for the full dataset and broken out by
modality (RNA, X-ray, MRI), then written to both a CSV file and a Markdown
table for easy comparison.

CLI arguments
-------------
--device      : PyTorch device string (default "cuda").
--out_csv     : Path for the output CSV (default
                ``data/analysis/summaries/v3_vs_v3_5_metrics.csv``).
--out_md      : Path for the Markdown table output (default
                ``data/analysis/summaries/v3_vs_v3_5_metrics.md``).
--v3_data     : Path to v3 base-aligned parquet (default
                ``data/processed/aligned/v3_aligned_base.parquet``).
--v35_data    : Path to v3.5 contrastive-aligned parquet (default
                ``data/processed/aligned/v3_aligned_contrastive.parquet``).
--v3_model    : Path to v3 model checkpoint .pt file.
--v35_model   : Path to v3.5 model checkpoint .pt file.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from sklearn.metrics import mean_squared_error, mean_absolute_error


# ---------------------------------------------------------------
#   v3 BASELINE FUSION MODEL (FusionRegressor)
#   matches train_fusion_transformer_v3.py
# ---------------------------------------------------------------

class Projector(nn.Module):
    """
    Two-layer MLP projector used inside the v3 ``FusionRegressor``.

    Maps a raw embedding to a 64-d representation before modality-token fusion.
    This definition mirrors the one used during v3 training so that the
    checkpoint can be loaded with ``strict=True``.
    """

    def __init__(self, in_dim=64, hidden=128, out_dim=64):
        """
        Parameters
        ----------
        in_dim : int
            Input dimensionality (default 64).
        hidden : int
            Hidden layer width (default 128).
        out_dim : int
            Output dimensionality (default 64).
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x):
        """
        Parameters
        ----------
        x : torch.Tensor of shape (B, in_dim)

        Returns
        -------
        torch.Tensor of shape (B, out_dim)
        """
        return self.net(x)


class FusionRegressor(nn.Module):
    """
    v3 baseline fusion model: modality-token transformer with age regression head.

    Architecture
    ------------
    1. Raw embedding -> ``Projector`` (64 -> 64).
    2. Add a learnable per-modality embedding (``nn.Embedding``).
    3. Project to ``d_model`` and pass through a ``TransformerEncoder``.
    4. Take the single CLS-like token output and feed through an MLP head to
       produce a scalar age prediction.

    This class is defined here to match the architecture saved in the v3
    checkpoint so weights can be loaded with ``strict=True``.
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
        Parameters
        ----------
        in_dim : int
            Dimension of the input embeddings (default 64).
        d_model : int
            Internal transformer model dimension (default 128).
        n_heads : int
            Number of attention heads (default 4).
        n_layers : int
            Number of transformer encoder layers (default 2).
        num_modalities : int
            Number of distinct modality IDs used in the modality embedding
            lookup (default 3: RNA=0, Xray=1, MRI=2).
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
        Predict biological age from a raw embedding and a modality ID.

        Parameters
        ----------
        x : torch.Tensor of shape (B, in_dim)
            Raw PCA embedding for each sample.
        mod_id : torch.Tensor of shape (B,), dtype long
            Integer modality index: 0=RNA, 1=Xray, 2=MRI.

        Returns
        -------
        torch.Tensor of shape (B,)
            Scalar age predictions.
        """
        z = self.projector(x)          # (B, 64)
        me = self.mod_embed(mod_id)    # (B, 64)
        z = z + me
        token = self.input_proj(z).unsqueeze(1)  # (B,1,d_model)
        h = self.encoder(token)[:, 0, :]         # (B,d_model)
        out = self.head(h).squeeze(1)            # (B,)
        return out


# ---------------------------------------------------------------
#   v3.5 CROSS-FUSION MODEL (CrossFusion)
#   matches train_fusion_transformer_cross_v3.py
# ---------------------------------------------------------------

class CrossFusion(nn.Module):
    """
    v3.5 cross-fusion transformer: 4-layer encoder with GELU activation.

    Accepts a single contrastive-aligned embedding vector per sample (no
    separate modality ID is needed because the alignment already harmonises
    modalities).  The vector is treated as a length-1 sequence, passed through
    a ``TransformerEncoder``, and decoded by an MLP head to a scalar age.

    This class is defined here (mirroring ``train_fusion_transformer_cross_v3``)
    so that the v3.5 checkpoint can be loaded with ``strict=True`` without
    importing the training module.
    """

    def __init__(self, D: int):
        """
        Parameters
        ----------
        D : int
            Dimension of the aligned embedding (expected to be 256 for v3.5
            contrastive outputs).
        """
        super().__init__()
        layer = nn.TransformerEncoderLayer(
            d_model=D,
            nhead=8,
            dim_feedforward=512,
            activation="gelu",
            batch_first=True,
        )
        self.attn = nn.TransformerEncoder(layer, num_layers=4)
        self.head = nn.Sequential(
            nn.Linear(D, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor of shape (B, D)

        Returns
        -------
        torch.Tensor of shape (B,) - scalar age predictions.
        """
        x = x.unsqueeze(1)          # (B,1,D)
        h = self.attn(x).squeeze(1) # (B,D)
        return self.head(h).squeeze(1)


# ---------------------------------------------------------------
#   DATA HELPERS – mirror your training logic
# ---------------------------------------------------------------

def build_v3_eval_arrays(df: pd.DataFrame, allowed_modalities=None):
    """
    Rebuild X, y, mod_id exactly like build_multimodal_dataset(),
    without shuffling, and optionally filtering by modality.
    """
    mod_map = {
        "rna": "rna",
        "gtex_rna": "rna",
        "xray": "xray",
        "chexpert_xray": "xray",
        "mri": "mri",
        "ixi_mri": "mri",
    }
    prefix_map = {
        "rna": "emb_rna_",
        "xray": "emb_xray_",
        "mri": "emb_mri_",
    }
    mod_id_map = {"rna": 0, "xray": 1, "mri": 2}

    df = df.copy()
    df["modality_std"] = df["modality"].map(mod_map)

    if allowed_modalities is None:
        allowed_modalities = ["rna", "xray", "mri"]

    X_list = []
    y_list = []
    modid_list = []

    for mod_std in allowed_modalities:
        dfm = df[df["modality_std"] == mod_std]
        if dfm.empty:
            continue

        prefix = prefix_map[mod_std]
        cols = [c for c in dfm.columns if c.startswith(prefix)]
        if not cols:
            continue

        X = dfm[cols].to_numpy(dtype="float32")
        y = dfm["age"].to_numpy(dtype="float32")

        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
        y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)

        X_list.append(X)
        y_list.append(y)
        modid_list.append(np.full(len(y), mod_id_map[mod_std], dtype="int64"))

    if not X_list:
        return None, None, None

    X_all = np.concatenate(X_list, axis=0)
    y_all = np.concatenate(y_list, axis=0)
    mod_all = np.concatenate(modid_list, axis=0)
    return X_all, y_all, mod_all


def build_v35_eval_arrays(df: pd.DataFrame, allowed_modalities=None):
    """
    Rebuild X, y exactly like build_dataset() in cross-fusion training,
    but optionally filtered by modality.
    """
    if allowed_modalities is None:
        allowed_modalities = ["rna", "xray", "mri"]

    cols_r = [c for c in df.columns if c.startswith("z_rna_")]
    cols_x = [c for c in df.columns if c.startswith("z_xray_")]
    cols_m = [c for c in df.columns if c.startswith("z_mri_")]

    X_list = []
    y_list = []

    def _filter_block(df_mod, cols):
        X = df_mod[cols].to_numpy(dtype="float32")
        y = df_mod["age"].to_numpy(dtype="float32")
        mask_X = np.isfinite(X).all(axis=1)
        mask_y = np.isfinite(y)
        mask = mask_X & mask_y
        return X[mask], y[mask]

    if "rna" in allowed_modalities and cols_r:
        df_r = df[df["modality"] == "rna"].copy()
        Xr, yr = _filter_block(df_r, cols_r)
        X_list.append(Xr)
        y_list.append(yr)

    if "xray" in allowed_modalities and cols_x:
        df_x = df[df["modality"] == "xray"].copy()
        Xx, yx = _filter_block(df_x, cols_x)
        X_list.append(Xx)
        y_list.append(yx)

    if "mri" in allowed_modalities and cols_m:
        df_m = df[df["modality"] == "mri"].copy()
        Xm, ym = _filter_block(df_m, cols_m)
        X_list.append(Xm)
        y_list.append(ym)

    if not X_list:
        return None, None

    X_all = np.concatenate(X_list, axis=0)
    y_all = np.concatenate(y_list, axis=0)
    return X_all, y_all


def compute_metrics(pred, y):
    """
    Compute regression metrics between predictions and ground-truth ages.

    Parameters
    ----------
    pred : array-like of shape (N,)
        Model age predictions.
    y : array-like of shape (N,)
        Ground-truth chronological ages.

    Returns
    -------
    dict with keys:
        "N"   - number of samples (int)
        "MSE" - mean squared error (float)
        "MAE" - mean absolute error (float)
    """
    return {
        "N": int(len(y)),
        "MSE": float(mean_squared_error(y, pred)),
        "MAE": float(mean_absolute_error(y, pred)),
    }


# ---------------------------------------------------------------
#   BATCHED EVAL HELPERS (fixes your CUDA error)
# ---------------------------------------------------------------

def eval_v3_batched(model, df, device, allowed_modalities=None, batch_size=2048):
    """
    Run batched inference with the v3 ``FusionRegressor`` and return metrics.

    Parameters
    ----------
    model : FusionRegressor
        Loaded v3 model in eval mode.
    df : pd.DataFrame
        v3 base-aligned dataframe (contains ``emb_*`` columns).
    device : str
        PyTorch device string ("cuda" or "cpu").
    allowed_modalities : list[str] or None
        Modalities to include (e.g. ``["rna"]``).  ``None`` uses all three.
    batch_size : int
        Number of samples per forward pass (default 2048).

    Returns
    -------
    dict or None
        ``compute_metrics`` output dict, or ``None`` if no matching data is
        found.
    """
    X, y, mod = build_v3_eval_arrays(df, allowed_modalities)
    if X is None:
        return None

    preds = []
    n = len(X)
    for start in range(0, n, batch_size):
        end = start + batch_size
        xb = torch.tensor(X[start:end], dtype=torch.float32, device=device)
        mb = torch.tensor(mod[start:end], dtype=torch.long, device=device)
        with torch.no_grad():
            pb = model(xb, mb).cpu().numpy()
        preds.append(pb)

    pred = np.concatenate(preds, axis=0)
    return compute_metrics(pred, y)


def eval_v35_batched(model, df, device, allowed_modalities=None, batch_size=2048):
    """
    Run batched inference with the v3.5 ``CrossFusion`` model and return metrics.

    Parameters
    ----------
    model : CrossFusion
        Loaded v3.5 model in eval mode.
    df : pd.DataFrame
        v3.5 contrastive-aligned dataframe (contains ``z_*`` columns).
    device : str
        PyTorch device string ("cuda" or "cpu").
    allowed_modalities : list[str] or None
        Modalities to include (e.g. ``["xray"]``).  ``None`` uses all three.
    batch_size : int
        Number of samples per forward pass (default 2048).

    Returns
    -------
    dict or None
        ``compute_metrics`` output dict, or ``None`` if no matching data is
        found.
    """
    X, y = build_v35_eval_arrays(df, allowed_modalities)
    if X is None:
        return None

    preds = []
    n = len(X)
    for start in range(0, n, batch_size):
        end = start + batch_size
        xb = torch.tensor(X[start:end], dtype=torch.float32, device=device)
        with torch.no_grad():
            pb = model(xb).cpu().numpy()
        preds.append(pb)

    pred = np.concatenate(preds, axis=0)
    return compute_metrics(pred, y)


# ---------------------------------------------------------------
#   MAIN
# ---------------------------------------------------------------

def main():
    """
    Parse CLI arguments, load both model versions, evaluate each on its
    corresponding dataset, and write the comparison table to CSV and Markdown.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out_csv", default="data/analysis/summaries/v3_vs_v3_5_metrics.csv")
    parser.add_argument("--out_md",  default="data/analysis/summaries/v3_vs_v3_5_metrics.md")

    parser.add_argument("--v3_data",   default="data/processed/aligned/v3_aligned_base.parquet")
    parser.add_argument("--v35_data",  default="data/processed/aligned/v3_aligned_contrastive.parquet")

    parser.add_argument("--v3_model",  default="models/organon_multimodal/fusion/fusion_transformer_v3.pt")
    parser.add_argument("--v35_model", default="models/synapse_aligned/fusion_cross_v3_5/models/fusion_cross_v3_cross.pt")

    args = parser.parse_args()
    device = args.device

    # ---- Load data ----
    df_v3  = pd.read_parquet(args.v3_data)
    df_v35 = pd.read_parquet(args.v35_data)

    # ---- Rebuild + load v3 baseline model (in_dim = 64) ----
    print("[COMPARE] Rebuilding v3 FusionRegressor (in_dim=64)...")
    model_v3 = FusionRegressor(in_dim=64).to(device)
    state_v3 = torch.load(args.v3_model, map_location=device)
    model_v3.load_state_dict(state_v3, strict=True)
    model_v3.eval()

    # ---- Rebuild + load v3.5 CrossFusion (D = z_rna_ dim) ----
    cols_r = [c for c in df_v35.columns if c.startswith("z_rna_")]
    if not cols_r:
        raise RuntimeError("Could not find any 'z_rna_' columns in v3_aligned_contrastive.")
    D_v35 = len(cols_r)
    print(f"[COMPARE] Rebuilding v3.5 CrossFusion with D={D_v35}...")
    model_v35 = CrossFusion(D=D_v35).to(device)
    state_v35 = torch.load(args.v35_model, map_location=device)
    model_v35.load_state_dict(state_v35, strict=True)
    model_v35.eval()

    rows = []

    # ===========================================================
    # v3 baseline – ALL + per-modality
    # ===========================================================
    print("[COMPARE] Evaluating v3 baseline...")

    res_all = eval_v3_batched(model_v3, df_v3, device, ["rna", "xray", "mri"])
    rows.append({"Model": "v3 baseline", "Modality": "ALL", **res_all})

    for mod in ["rna", "xray", "mri"]:
        res = eval_v3_batched(model_v3, df_v3, device, [mod])
        if res is not None:
            rows.append({"Model": "v3 baseline", "Modality": mod, **res})

    # ===========================================================
    # v3.5 cross-fusion – ALL + per-modality
    # ===========================================================
    print("[COMPARE] Evaluating v3.5 cross-fusion...")

    res_all_35 = eval_v35_batched(model_v35, df_v35, device, ["rna", "xray", "mri"])
    rows.append({"Model": "v3.5 cross-fusion", "Modality": "ALL", **res_all_35})

    for mod in ["rna", "xray", "mri"]:
        res = eval_v35_batched(model_v35, df_v35, device, [mod])
        if res is not None:
            rows.append({"Model": "v3.5 cross-fusion", "Modality": mod, **res})

    # ---- Save outputs ----
    df_out = pd.DataFrame(rows)
    out_csv = Path(args.out_csv)
    out_md  = Path(args.out_md)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(out_csv, index=False)

    md_lines = [
        "| Model | Modality | N | MSE | MAE |",
        "|---|---|---|---|---|",
    ]
    for _, r in df_out.iterrows():
        md_lines.append(
            f"| {r['Model']} | {r['Modality']} | {r['N']} | {r['MSE']:.3f} | {r['MAE']:.3f} |"
        )
    out_md.write_text("\n".join(md_lines), encoding="utf-8")

    print(f"[COMPARE] Saved CSV → {out_csv}")
    print(f"[COMPARE] Saved MD  → {out_md}")
    print("[COMPARE] Done.")


if __name__ == "__main__":
    main()
