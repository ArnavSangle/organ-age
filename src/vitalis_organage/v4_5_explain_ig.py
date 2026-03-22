"""
v4_5_explain_ig.py
==================
Compute Integrated Gradient (IG) attributions for the v4.5 CrossFusion model
in *latent* embedding space (i.e. attributing the ``z_rna_*`` / ``z_xray_*``
/ ``z_mri_*`` dimensions rather than raw gene expression values).

Workflow
--------
1. Load the aligned embedding parquet.
2. Select a *background* set of rows for the IG baseline (mean embedding).
3. Select a *target* set of rows to explain.
4. For each target sample compute the path integral of gradients of ``mu``
   with respect to the input embedding from the baseline to the sample
   (Sundararajan et al., 2017).
5. Average ``|attribution|`` over all target samples to obtain a
   ``(D,)`` importance vector per latent dimension.
6. Save results as both ``.csv`` and ``.parquet`` under ``data/analysis/``.

The saved CSV columns are ``feature`` (e.g. ``z_rna_224``) and
``mean_abs_ig``.  These are used downstream by ``v4_5_plot_ig_panels.py``
and ``v4_5_map_latent_to_genes.py``.
"""
import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import torch
from torch import nn

# --- make sure 'src' is on sys.path so `vitalis_organage` imports work ---
THIS_FILE = Path(__file__).resolve()
SRC_ROOT = THIS_FILE.parents[1]  # .../organ-age/src
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from vitalis_organage.v4_5_crossfusion_model import CrossFusionV45


# -------------------------------------------------------------------
# Utilities
# -------------------------------------------------------------------

ORGAN_TO_ID = {
    "adipose": 0,
    "brain": 1,
    "brain_cortex": 2,
    "colon": 3,
    "heart": 4,
    "kidney": 5,
    "liver": 6,
    "lung": 7,
    "skeletal_muscle": 8,
    "skin": 9,
    "whole_blood": 10,
}

MOD_PREFIX = {
    "rna": "z_rna_",
    "xray": "z_xray_",
    "mri": "z_mri_",
}


def pick_rows(df, modality: str, organ: str, n_bg: int, n_target: int):
    """
    Select background and target row subsets for a given modality and organ.

    The full subset is shuffled with a fixed random seed before slicing so
    results are reproducible.  The same leading rows are used for both
    background (baseline) and target (to explain), up to their respective
    size limits.

    Parameters
    ----------
    df : pd.DataFrame
        Full aligned embedding table.
    modality : str
        One of ``'rna'``, ``'xray'``, ``'mri'``.
    organ : str
        Organ name as it appears in the ``organ`` column.
    n_bg : int
        Maximum number of rows to include in the background set.
    n_target : int
        Maximum number of rows to explain.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        ``(df_bg, df_target)`` both subsets of the modality+organ slice.

    Raises
    ------
    RuntimeError
        If no rows exist for the requested modality and organ combination.
    """
    df_sub = df[(df["modality"] == modality) & (df["organ"] == organ)].copy()
    if df_sub.empty:
        raise RuntimeError(f"No rows found for modality='{modality}' organ='{organ}'")

    # Shuffle once for reproducibility
    df_sub = df_sub.sample(frac=1.0, random_state=123).reset_index(drop=True)

    # Background rows for IG baseline distribution
    df_bg = df_sub.iloc[: min(n_bg, len(df_sub))]
    # Target rows to actually explain
    df_target = df_sub.iloc[: min(n_target, len(df_sub))]

    return df_bg, df_target


def extract_X(df_mod: pd.DataFrame, prefix: str):
    """
    Extract the embedding matrix and column names for a given prefix.

    Non-finite values (NaN, Inf, -Inf) are replaced with 0.0 so the tensor
    is safe to pass to the model.

    Parameters
    ----------
    df_mod : pd.DataFrame
        Subset of the aligned table for a single modality / organ.
    prefix : str
        Column prefix to filter on, e.g. ``'z_rna_'``.

    Returns
    -------
    tuple[np.ndarray, list[str]]
        ``(X, cols)`` where ``X`` has shape ``(N, D)`` and ``cols`` are the
        matched column names.

    Raises
    ------
    RuntimeError
        If no columns with the requested prefix exist.
    """
    cols = [c for c in df_mod.columns if c.startswith(prefix)]
    if not cols:
        raise RuntimeError(f"No columns with prefix '{prefix}' in dataframe.")
    X = df_mod[cols].to_numpy(dtype="float32")
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    return X, cols


# -------------------------------------------------------------------
# Integrated Gradients
# -------------------------------------------------------------------

def integrated_gradients(
    model: nn.Module,
    x: torch.Tensor,
    organ_id: torch.Tensor,
    baseline: torch.Tensor,
    steps: int = 64,
    device: str = "cuda",
) -> torch.Tensor:
    """
    Integrated Gradients for a single sample.

    x:        (D,) input embedding
    organ_id: scalar LongTensor with the organ index
    baseline: (D,) baseline embedding (e.g. zeros or mean)
    returns:  (D,) attribution
    """
    model.eval()

    # Move to device
    x = x.to(device)
    baseline = baseline.to(device)
    organ_id = organ_id.to(device)

    # Scale inputs from baseline -> x
    alphas = torch.linspace(0.0, 1.0, steps, device=device).view(-1, 1)  # (S, 1)
    x_diff = (x - baseline).unsqueeze(0)  # (1, D)
    scaled = baseline.unsqueeze(0) + alphas * x_diff  # (S, D)

    # Organ IDs for each scaled sample
    organ_ids = organ_id.repeat(steps)  # (S,)

    scaled.requires_grad_(True)
    preds, _ = model(scaled, organ_ids)  # CrossFusionV45 returns (mu, sigma)

    # Gradients of mu wrt inputs
    grads = torch.autograd.grad(
        outputs=preds,
        inputs=scaled,
        grad_outputs=torch.ones_like(preds),
        create_graph=False,
        retain_graph=False,
    )[0]  # (S, D)

    avg_grads = grads.mean(dim=0)        # (D,)
    attributions = x_diff.squeeze(0) * avg_grads  # (D,)
    return attributions.detach().cpu()


# -------------------------------------------------------------------
# Main explainability routine
# -------------------------------------------------------------------

def main():
    """
    Command-line entry point for the latent IG explainability script.

    Loads data, builds the background baseline, runs IG for all target
    samples, averages absolute attributions, and saves the per-feature
    importance table as CSV and parquet.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_path",
        type=str,
        default="data/processed/aligned/v3_aligned_contrastive.parquet",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="models/v4_5/fusion_cross_v4_5.pt",
    )
    parser.add_argument(
        "--modality",
        type=str,
        default="rna",
        choices=["rna", "xray", "mri"],
    )
    parser.add_argument(
        "--organ",
        type=str,
        default="liver",
        help="Organ name (must be one of ORGAN_TO_ID keys).",
    )
    parser.add_argument(
        "--n_background",
        type=int,
        default=256,
        help="Number of background samples for baseline.",
    )
    parser.add_argument(
        "--n_target",
        type=int,
        default=64,
        help="Number of target samples to explain.",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=64,
        help="Number of steps for Integrated Gradients.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="cuda or cpu",
    )
    parser.add_argument(
        "--out_csv",
        type=str,
        default="data/analysis/v4_5_ig_importance.csv",
        help="Output CSV for mean |attribution| per latent dimension.",
    )
    args = parser.parse_args()

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("[IG] CUDA not available, falling back to CPU.")
        device = "cpu"

    # 1) Load data
    data_path = Path(args.data_path)
    print(f"[IG] Loading data from {data_path} ...")
    df = pd.read_parquet(data_path)
    print("[IG] df shape:", df.shape)

    if args.organ not in ORGAN_TO_ID:
        raise RuntimeError(f"Unknown organ '{args.organ}'. Valid: {list(ORGAN_TO_ID.keys())}")

    organ_id = ORGAN_TO_ID[args.organ]
    prefix = MOD_PREFIX[args.modality]

    # 2) Select background + target rows
    df_bg, df_target = pick_rows(df, args.modality, args.organ, args.n_background, args.n_target)
    X_bg, cols = extract_X(df_bg, prefix)
    X_tgt, _ = extract_X(df_target, prefix)

    D = X_bg.shape[1]
    print(f"[IG] Modality={args.modality}, organ={args.organ}, D={D}")
    print(f"[IG] Background N={X_bg.shape[0]}, Target N={X_tgt.shape[0]}")

    # 3) Load model
    print("[IG] Initializing CrossFusionV45...")
    model_path = Path(args.model_path)
    model = CrossFusionV45(D, len(ORGAN_TO_ID)).to(device)

    # ✅ Drop organ-conditioning layers whose shapes changed since training
    state = torch.load(model_path, map_location=device)
    for bad_key in ["organ_embed.weight", "organ_proj.weight"]:
        if bad_key in state:
            print(f"[IG] Dropping {bad_key} from checkpoint; shape={state[bad_key].shape}")
            del state[bad_key]

    missing, unexpected = model.load_state_dict(state, strict=False)
    print(f"[IG] Loaded weights from {model_path}")
    if missing:
        print("[IG] WARNING: Missing keys:", missing)
    if unexpected:
        print("[IG] WARNING: Unexpected keys:", unexpected)

    model.eval()

    # 4) Baseline = mean embedding of background set
    baseline_np = X_bg.mean(axis=0, dtype="float32")
    baseline = torch.from_numpy(baseline_np)

    # 5) Compute IG for each target sample
    organ_tensor = torch.tensor([organ_id], dtype=torch.long, device=device)
    attributions = []

    print("[IG] Computing Integrated Gradients...")
    for i in range(X_tgt.shape[0]):
        x = torch.from_numpy(X_tgt[i])
        attr = integrated_gradients(
            model,
            x,
            organ_tensor,
            baseline,
            steps=args.steps,
            device=device,
        )
        attributions.append(attr.numpy())

    attributions = np.stack(attributions, axis=0)  # (N_target, D)
    mean_abs_attr = np.mean(np.abs(attributions), axis=0)  # (D,)

    # 6) Save CSV + parquet with feature importances in latent space
    out_df = pd.DataFrame(
        {
            "feature": cols,
            "mean_abs_ig": mean_abs_attr,
        }
    ).sort_values("mean_abs_ig", ascending=False)

    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_parquet(out_path.with_suffix(".parquet"))
    out_df.to_csv(out_path, index=False)

    print(f"[IG] Saved latent feature importances -> {out_path}")
    print(f"[IG] Top 10 features:\n{out_df.head(10)}")


if __name__ == "__main__":
    main()
