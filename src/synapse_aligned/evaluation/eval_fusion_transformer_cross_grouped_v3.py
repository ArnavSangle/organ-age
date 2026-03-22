"""
eval_fusion_transformer_cross_grouped_v3.py - Grouped evaluation of the v3.5 CrossFusion model.

Extends the basic evaluation in ``eval_fusion_transformer_cross_v3.py`` with
optional sub-group breakdowns:

  * **Global** - overall MSE/MAE plus per-modality (RNA, X-ray, MRI).
  * **Organ** - metrics stratified by the ``organ`` column in the parquet table.
  * **Age bin** - metrics stratified by five age ranges: <20, 20-40, 40-60,
    60-80, 80+.

All results are collected into a single CSV at the path specified by
``--out_csv``.

CLI arguments
-------------
--device      : PyTorch device string (default "cuda").
--group_by    : Grouping strategy: "none", "organ", or "age_bin".
--out_csv     : Output CSV path (default
                ``data/analysis/summaries/v3_5_cross_grouped_metrics.csv``).
--data_path   : Path to the contrastive-aligned parquet (default
                ``data/processed/aligned/v3_aligned_contrastive.parquet``).
--model_path  : Path to the CrossFusion checkpoint .pt file.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from sklearn.metrics import mean_squared_error, mean_absolute_error


# ---------------------------------------------------------------
#   CrossFusion model (same as train_fusion_transformer_cross_v3.py)
# ---------------------------------------------------------------

class CrossFusion(nn.Module):
    """
    v3.5 cross-fusion transformer: 4-layer encoder with GELU + MLP regression head.

    Mirrors the definition in ``train_fusion_transformer_cross_v3.py`` so
    that checkpoints can be loaded directly without importing the training
    module.  Each sample's aligned ``D``-dimensional embedding is treated as a
    length-1 sequence, contextualised by the encoder, and decoded to a scalar
    age prediction.
    """

    def __init__(self, D: int):
        """
        Parameters
        ----------
        D : int
            Dimension of the contrastive-aligned input embeddings (typically
            256 for v3.5 outputs).
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
            Aligned embedding for each sample in the batch.

        Returns
        -------
        torch.Tensor of shape (B,)
            Scalar biological-age predictions.
        """
        x = x.unsqueeze(1)          # (B,1,D)
        h = self.attn(x).squeeze(1) # (B,D)
        return self.head(h).squeeze(1)


# ---------------------------------------------------------------
#   Data helpers – mirror your training build_dataset()
# ---------------------------------------------------------------

def _filter_block(df_mod: pd.DataFrame, cols: list[str]):
    """
    Extract embedding matrix and age labels from a modality-filtered dataframe,
    dropping rows where any embedding value or the age label is non-finite.

    Parameters
    ----------
    df_mod : pd.DataFrame
        Rows belonging to a single modality.
    cols : list[str]
        Names of the aligned embedding columns to extract (e.g. ``z_rna_*``).

    Returns
    -------
    X : np.ndarray of shape (N_clean, len(cols)), dtype float32
    y : np.ndarray of shape (N_clean,), dtype float32
    """
    X = df_mod[cols].to_numpy(dtype="float32")
    y = df_mod["age"].to_numpy(dtype="float32")

    mask_X = np.isfinite(X).all(axis=1)
    mask_y = np.isfinite(y)
    mask = mask_X & mask_y

    X = X[mask]
    y = y[mask]
    return X, y


def build_arrays_for_modalities(df: pd.DataFrame, allowed_modalities=None):
    """
    Rebuild X, y like in training, optionally filtered to a subset of modalities.
    Uses z_rna_*, z_xray_*, z_mri_*.
    """
    if allowed_modalities is None:
        allowed_modalities = ["rna", "xray", "mri"]

    cols_r = [c for c in df.columns if c.startswith("z_rna_")]
    cols_x = [c for c in df.columns if c.startswith("z_xray_")]
    cols_m = [c for c in df.columns if c.startswith("z_mri_")]

    X_list = []
    y_list = []

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
    Compute MSE and MAE between model predictions and ground-truth ages.

    Parameters
    ----------
    pred : array-like of shape (N,)
        Predicted ages.
    y : array-like of shape (N,)
        Ground-truth chronological ages.

    Returns
    -------
    dict with keys:
        "N"   - sample count (int)
        "MSE" - mean squared error (float)
        "MAE" - mean absolute error (float)
    """
    return {
        "N": int(len(y)),
        "MSE": float(mean_squared_error(y, pred)),
        "MAE": float(mean_absolute_error(y, pred)),
    }


def eval_batched(model, X: np.ndarray, y: np.ndarray, device: str, batch_size: int = 2048):
    """
    Run batched inference and return MSE / MAE metrics.

    Splits ``X`` into chunks of ``batch_size`` to avoid out-of-memory errors
    during GPU inference, then reassembles predictions before scoring.

    Parameters
    ----------
    model : CrossFusion
        Loaded model in eval mode.
    X : np.ndarray of shape (N, D)
        Aligned embedding matrix.
    y : np.ndarray of shape (N,)
        Ground-truth ages.
    device : str
        PyTorch device string ("cuda" or "cpu").
    batch_size : int
        Number of samples per inference batch (default 2048).

    Returns
    -------
    dict
        Output of ``compute_metrics``: keys "N", "MSE", "MAE".
    """
    n = len(X)
    preds = []
    for start in range(0, n, batch_size):
        end = start + batch_size
        xb = torch.tensor(X[start:end], dtype=torch.float32, device=device)
        with torch.no_grad():
            pb = model(xb).cpu().numpy()
        preds.append(pb)
    pred = np.concatenate(preds, axis=0)
    return compute_metrics(pred, y)


# ---------------------------------------------------------------
#   Group-by helpers
# ---------------------------------------------------------------

def add_age_bins(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds an age_bin column: [20,40), [40,60), [60,80), 80+.
    You can tweak these later if you want different bins.
    """
    bins = [0, 20, 40, 60, 80, 200]
    labels = ["<20", "20-40", "40-60", "60-80", "80+"]
    df = df.copy()
    df["age_bin"] = pd.cut(df["age"], bins=bins, labels=labels, right=False)
    return df


# ---------------------------------------------------------------
#   MAIN
# ---------------------------------------------------------------

def main():
    """
    Parse CLI arguments, load the CrossFusion model, run grouped evaluation,
    and save all metrics to a CSV file.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--group_by",
        type=str,
        default="none",
        choices=["none", "organ", "age_bin"],
        help="none: global + per-modality only; organ: group by organ; age_bin: group by age ranges.",
    )
    parser.add_argument(
        "--out_csv",
        default="data/analysis/summaries/v3_5_cross_grouped_metrics.csv",
    )

    parser.add_argument(
        "--data_path",
        default="data/processed/aligned/v3_aligned_contrastive.parquet",
    )
    parser.add_argument(
        "--model_path",
        default="models/synapse_aligned/fusion_cross_v3_5/models/fusion_cross_v3_cross.pt",
    )

    args = parser.parse_args()
    device = args.device

    df = pd.read_parquet(args.data_path)
    print("[GROUPED EVAL] df shape:", df.shape)
    print("[GROUPED EVAL] modalities:\n", df["modality"].value_counts())

    # Build model
    cols_r = [c for c in df.columns if c.startswith("z_rna_")]
    if not cols_r:
        raise RuntimeError("No z_rna_ columns found – check aligned contrastive table.")
    D = len(cols_r)
    print(f"[GROUPED EVAL] Feature dim D={D}")

    model = CrossFusion(D=D).to(device)
    state = torch.load(args.model_path, map_location=device)
    model.load_state_dict(state, strict=True)
    model.eval()

    rows = []

    # -----------------------------------------------------------
    # 1) Global metrics (ALL + per-modality)
    # -----------------------------------------------------------
    print("\n[GROUPED EVAL] Global metrics...")
    X_all, y_all = build_arrays_for_modalities(df, ["rna", "xray", "mri"])
    res_all = eval_batched(model, X_all, y_all, device)
    rows.append({"GroupType": "global", "Group": "ALL", **res_all})

    for mod in ["rna", "xray", "mri"]:
        X_m, y_m = build_arrays_for_modalities(df, [mod])
        if X_m is None:
            continue
        res = eval_batched(model, X_m, y_m, device)
        rows.append({"GroupType": "global_modality", "Group": mod, **res})

    # -----------------------------------------------------------
    # 2) Group by organ (if requested)
    # -----------------------------------------------------------
    if args.group_by == "organ":
        print("\n[GROUPED EVAL] Group-by ORGAN metrics...")
        organs = sorted(df["organ"].dropna().unique())
        for organ in organs:
            df_o = df[df["organ"] == organ]
            if df_o.empty:
                continue
            X_o, y_o = build_arrays_for_modalities(df_o, ["rna", "xray", "mri"])
            if X_o is None:
                continue
            res = eval_batched(model, X_o, y_o, device)
            print(f"  organ={organ:20s} | N={res['N']:6d} | MSE={res['MSE']:.3f} | MAE={res['MAE']:.3f}")
            rows.append({"GroupType": "organ", "Group": organ, **res})

    # -----------------------------------------------------------
    # 3) Group by age_bin (if requested)
    # -----------------------------------------------------------
    if args.group_by == "age_bin":
        print("\n[GROUPED EVAL] Group-by AGE_BIN metrics...")
        df_bin = add_age_bins(df)
        bins = df_bin["age_bin"].dropna().unique()
        # keep fixed label order if possible
        label_order = ["<20", "20-40", "40-60", "60-80", "80+"]
        for label in label_order:
            if label not in bins:
                continue
            df_b = df_bin[df_bin["age_bin"] == label]
            if df_b.empty:
                continue
            X_b, y_b = build_arrays_for_modalities(df_b, ["rna", "xray", "mri"])
            if X_b is None:
                continue
            res = eval_batched(model, X_b, y_b, device)
            print(f"  age_bin={label:5s} | N={res['N']:6d} | MSE={res['MSE']:.3f} | MAE={res['MAE']:.3f}")
            rows.append({"GroupType": "age_bin", "Group": label, **res})

    # -----------------------------------------------------------
    # Save to CSV
    # -----------------------------------------------------------
    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(f"\n[GROUPED EVAL] Saved metrics → {out_csv}")


if __name__ == "__main__":
    main()
