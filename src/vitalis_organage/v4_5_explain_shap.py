"""
v4_5_explain_shap.py
====================
SHAP-based explainability for the v4.5 CrossFusion model in latent
embedding space.

Uses ``shap.DeepExplainer`` to attribute the predicted organ age (mu output)
to the combined ``z_rna_*`` / ``z_xray_*`` / ``z_mri_*`` latent dimensions.
A random subset of 200 samples from the data is used as the background
distribution for the SHAP baseline; 1 000 samples are explained.

Two figures are produced and saved to ``figures/v4_5_explain/``:
  * ``shap_summary.png`` – beeswarm/summary plot of SHAP values.
  * ``shap_bar.png``     – bar chart of mean absolute SHAP values per feature.

Note: This script uses the CPU for SHAP computation to remain compatible
with ``shap.DeepExplainer`` constraints.
"""
import shap
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from v4_5_crossfusion_model import CrossFusionV45, gaussian_nll


def load_model(path, organ_dim=64, emb_dim=256, n_organs=11, device="cpu"):
    """
    Instantiate ``CrossFusionV45`` and load saved weights from a checkpoint.

    Parameters
    ----------
    path : str
        Path to the ``.pt`` checkpoint file.
    organ_dim : int
        Organ embedding dimensionality (must match the saved checkpoint).
    emb_dim : int
        Input embedding dimensionality (must match the saved checkpoint).
    n_organs : int
        Number of distinct organs (vocabulary size).
    device : str
        Target device string (``'cpu'`` or ``'cuda'``).

    Returns
    -------
    CrossFusionV45
        Model in ``eval`` mode with weights loaded from ``path``.
    """
    model = CrossFusionV45(
        emb_dim=emb_dim,
        organ_dim=organ_dim,
        n_organs=n_organs,
        d_model=256,
    )
    state = torch.load(path, map_location=device)
    model.load_state_dict(state, strict=True)
    model.to(device)
    model.eval()
    return model


def explain(model, X, organ_ids, outdir):
    """
    Run SHAP DeepExplainer on the model and save summary figures.

    Uses 200 randomly chosen samples from ``X`` as the background
    distribution.  SHAP values are computed for all rows of ``X``
    and visualised as a beeswarm summary plot and a bar plot.

    Parameters
    ----------
    model : CrossFusionV45
        Trained model in ``eval`` mode.
    X : np.ndarray
        Input embedding matrix, shape ``(N, D)``.
    organ_ids : np.ndarray
        Integer organ IDs aligned with rows of ``X``, shape ``(N,)``.
    outdir : str or Path
        Directory where the two PNG figures will be saved.
    """
    Path(outdir).mkdir(parents=True, exist_ok=True)

    # SHAP on latent embedding (256 dims)
    background = X[np.random.choice(len(X), 200, replace=False)]
    explainer = shap.DeepExplainer(
        lambda x: model(torch.tensor(x).float(), torch.tensor(organ_ids[:x.shape[0]])).detach()[0],
        torch.tensor(background).float(),
    )

    shap_vals = explainer.shap_values(torch.tensor(X).float())

    # --- Summary Plot
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_vals, X, show=False)
    plt.tight_layout()
    plt.savefig(Path(outdir) / "shap_summary.png", dpi=300)
    plt.close()

    # --- Bar plot
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_vals, X, plot_type="bar", show=False)
    plt.tight_layout()
    plt.savefig(Path(outdir) / "shap_bar.png", dpi=300)
    plt.close()

    print("[SHAP] Saved SHAP summary + bar plots.")


def main():
    """
    Entry point: load data and model, subsample for SHAP, and call ``explain``.
    """
    model_path = "models/v4_5/fusion_cross_v4_5.pt"
    data_path = "data/processed/aligned/v3_aligned_contrastive.parquet"

    outdir = "figures/v4_5_explain"

    df = pd.read_parquet(data_path)
    X = df[[c for c in df.columns if c.startswith("z_rna_") or c.startswith("z_xray_") or c.startswith("z_mri_")]].to_numpy()
    organs = df["organ"].astype("category").cat.codes.to_numpy()

    # Model
    model = load_model(model_path, device="cpu", n_organs=len(set(organs)))

    # Small subset for SHAP
    keep = np.random.choice(len(X), 1000, replace=False)
    explain(model, X[keep], organs[keep], outdir)


if __name__ == "__main__":
    main()
