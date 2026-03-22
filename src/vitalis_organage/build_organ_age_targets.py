"""
build_organ_age_targets.py
==========================
End-to-end pipeline that generates the organ-age normative table used by all
downstream analysis scripts.

Steps performed by ``main()``:
  1. Load the v3 aligned-contrastive embedding parquet.
  2. Build an organ-to-integer-ID mapping from the data (must match training).
  3. Instantiate and load the ``CrossFusionV45`` model checkpoint.
  4. Run batched inference to attach ``age_pred`` (mu) and ``age_sigma`` (sigma)
     for every RNA / X-ray / MRI row.
  5. Fit organ-specific linear normative curves on a designated healthy reference
     cohort (e.g. GTEx, IXI) and compute:
       - ``organ_age_delta``  : residual from the normative line (years)
       - ``organ_age_zscore`` : delta standardised by the reference spread
  6. Write the resulting table to ``data/analysis/organ_age_normative.parquet``.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

from v4_5_crossfusion_model import CrossFusionV45


# ---------------------------------------------------------------
# Helpers: attach v4.5 (mu, sigma) predictions to v3_aligned_contrastive
# ---------------------------------------------------------------

def _extract_block(df_mod: pd.DataFrame, cols_prefix: str, organ_to_id: dict):
    """
    From a modality-specific dataframe (df_mod), extract:
      - X: z_* embeddings as float32
      - idx: original row indices in the FULL df
      - organ_ids: integer organ IDs (0..n_organs-1)

    Cleans NaNs/inf in X and drops rows with missing organ.
    """
    cols = [c for c in df_mod.columns if c.startswith(cols_prefix)]
    if not cols:
        return None

    X = df_mod[cols].to_numpy(dtype="float32")
    organs_raw = df_mod["organ"].to_numpy()

    # Basic masks: finite embeddings + non-missing organ label
    mask_X = np.isfinite(X).all(axis=1)
    mask_o = pd.notna(organs_raw)

    mask = mask_X & mask_o
    if mask.sum() == 0:
        return None

    X = X[mask]
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    idx = df_mod.index.to_numpy()[mask]
    organs_clean = organs_raw[mask]

    organ_ids = np.array([organ_to_id[o] for o in organs_clean], dtype="int64")
    return X, idx, organ_ids


def attach_predictions(
    df: pd.DataFrame,
    model: nn.Module,
    organ_to_id: dict,
    device: str,
    batch_size: int = 2048,
) -> pd.DataFrame:
    """
    For each modality (rna/xray/mri), pull z_* embeddings, run the v4.5 model,
    and write predictions into:
        df['age_pred']  (mu)
        df['age_sigma'] (sigma)
    """
    df = df.copy()
    df["age_pred"] = np.nan
    df["age_sigma"] = np.nan

    cols_r = [c for c in df.columns if c.startswith("z_rna_")]
    cols_x = [c for c in df.columns if c.startswith("z_xray_")]
    cols_m = [c for c in df.columns if c.startswith("z_mri_")]

    # Infer D from any non-empty set of z_* columns
    D = None
    for cols in (cols_r, cols_x, cols_m):
        if cols:
            D = len(cols)
            break
    if D is None:
        raise RuntimeError("[ORGANAGE] No z_rna_, z_xray_, or z_mri_ columns found.")

    print(f"[ORGANAGE] Embedding dim D={D}")
    model.eval()

    def _predict_for_mod(modality: str, prefix: str):
        """
        Run batched inference for a single modality and write mu/sigma into df.

        Parameters
        ----------
        modality : str
            One of ``'rna'``, ``'xray'``, ``'mri'``.
        prefix : str
            Column prefix used to select the embedding columns, e.g.
            ``'z_rna_'``.
        """
        sub = df[df["modality"] == modality]
        if sub.empty:
            print(f"[ORGANAGE] No rows for modality '{modality}', skipping.")
            return

        out = _extract_block(sub, prefix, organ_to_id)
        if out is None:
            print(f"[ORGANAGE] No valid rows for modality '{modality}' / prefix '{prefix}', skipping.")
            return

        X, idx, organ_ids = out
        n = X.shape[0]
        mu_list = []
        sigma_list = []

        for start in range(0, n, batch_size):
            end = start + batch_size
            xb = torch.tensor(X[start:end], dtype=torch.float32, device=device)
            ob = torch.tensor(organ_ids[start:end], dtype=torch.long, device=device)

            with torch.no_grad():
                mu_b, sigma_b = model(xb, ob)
                mu_b = torch.nan_to_num(mu_b, nan=0.0, posinf=0.0, neginf=0.0)
                sigma_b = torch.nan_to_num(sigma_b, nan=10.0, posinf=10.0, neginf=1.0)

            mu_list.append(mu_b.cpu().numpy())
            sigma_list.append(sigma_b.cpu().numpy())

        mu_all = np.concatenate(mu_list, axis=0)
        sig_all = np.concatenate(sigma_list, axis=0)

        # Ensure 1D
        mu_all = np.asarray(mu_all).reshape(-1)
        sig_all = np.asarray(sig_all).reshape(-1)

        # Use Series with the same index to avoid len-mismatch weirdness
        mu_ser = pd.Series(mu_all, index=idx)
        sig_ser = pd.Series(sig_all, index=idx)

        df.loc[idx, "age_pred"] = mu_ser
        df.loc[idx, "age_sigma"] = sig_ser

        print(
            f"[ORGANAGE] modality={modality:5s} | N={n:6d} predictions attached "
            f"(μ, σ)."
        )


    if cols_r:
        _predict_for_mod("rna", "z_rna_")
    if cols_x:
        _predict_for_mod("xray", "z_xray_")
    if cols_m:
        _predict_for_mod("mri", "z_mri_")

    return df


# ---------------------------------------------------------------
# Normative curve fitting per organ (unchanged structure)
# ---------------------------------------------------------------

def fit_normative_curves(
    df_pred: pd.DataFrame,
    healthy_sources: list[str],
    min_ref_n: int = 50,
) -> pd.DataFrame:
    """
    For each organ, fit linear model:
        age_pred ~ age_chrono
    on a healthy reference subset (by source), then compute:
        organ_age_delta   = residual
        organ_age_zscore  = standardized residual.
    """
    df = df_pred.copy()

    # 🔧 Important: make index unique to avoid pandas reindex issues
    df = df.reset_index(drop=True)

    df["organ_age_delta"] = np.nan
    df["organ_age_zscore"] = np.nan


    organs = sorted(df["organ"].dropna().unique())
    print(f"[ORGANAGE] Found {len(organs)} organs:", organs)

    for organ in organs:
        df_o = df[df["organ"] == organ]

        # Reference subset: designated "healthy" sources only
        df_ref = df_o[
            df_o["source"].isin(healthy_sources)
        ].dropna(subset=["age_chrono", "age_pred"])
        n_ref = len(df_ref)
        if n_ref < min_ref_n:
            print(
                f"[ORGANAGE] organ={organ:20s} | REF N={n_ref:4d} < {min_ref_n}, "
                f"skipping normative fit."
            )
            continue

        x_ref = df_ref["age_chrono"].to_numpy(dtype="float32")
        y_ref = df_ref["age_pred"].to_numpy(dtype="float32")

        # Linear fit: age_pred ≈ a * age_chrono + b
        a, b = np.polyfit(x_ref, y_ref, 1)
        print(
            f"[ORGANAGE] organ={organ:20s} | REF N={n_ref:4d} | "
            f"fit: age_pred ≈ {a:.3f} * age + {b:.3f}"
        )

        # Apply to all samples for this organ
        df_o_all = df_o.dropna(subset=["age_chrono", "age_pred"]).copy()
        if df_o_all.empty:
            continue

        x_all = df_o_all["age_chrono"].to_numpy(dtype="float32")
        y_all = df_o_all["age_pred"].to_numpy(dtype="float32")

        expected = a * x_all + b
        residual = y_all - expected  # years older(+) / younger(-) than typical

        # Std from reference residuals
        ref_resid = y_ref - (a * x_ref + b)
        std_resid = ref_resid.std(ddof=1)
        if std_resid <= 1e-6 or not np.isfinite(std_resid):
            z_all = np.zeros_like(residual)
            print(
                f"[ORGANAGE] organ={organ:20s} | std_resid very small; "
                f"z-scores set to 0."
            )
        else:
            z_all = residual / std_resid

        # Ensure 1D arrays
        residual = np.asarray(residual).reshape(-1)
        z_all = np.asarray(z_all).reshape(-1)

        # Use Series with matching index to avoid len-mismatch
        delta_ser = pd.Series(residual, index=df_o_all.index)
        z_ser = pd.Series(z_all, index=df_o_all.index)

        df.loc[df_o_all.index, "organ_age_delta"] = delta_ser
        df.loc[df_o_all.index, "organ_age_zscore"] = z_ser

    return df


# ---------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------

def main():
    """
    Command-line entry point for the organ-age target builder.

    Parses CLI arguments, orchestrates data loading, model inference via
    ``attach_predictions``, normative curve fitting via
    ``fit_normative_curves``, and saves the final parquet table.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="cuda or cpu",
    )
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
        "--healthy_sources",
        type=str,
        default="gtex_v10,ixi",
        help="Comma-separated list of sources to treat as 'healthy' for normative curves.",
    )
    parser.add_argument(
        "--out_path",
        type=str,
        default="data/analysis/organ_age_normative.parquet",
    )
    parser.add_argument(
        "--min_ref_n",
        type=int,
        default=50,
        help="Minimum reference samples per organ required to fit a normative curve.",
    )
    args = parser.parse_args()

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("[ORGANAGE] CUDA not available; falling back to CPU.")
        device = "cpu"

    healthy_sources = [s.strip() for s in args.healthy_sources.split(",") if s.strip()]
    print("[ORGANAGE] Healthy reference sources:", healthy_sources)

    data_path = Path(args.data_path)
    model_path = Path(args.model_path)
    out_path = Path(args.out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # 1) Load aligned contrastive table
    print(f"[ORGANAGE] Loading data from {data_path} ...")
    df = pd.read_parquet(data_path)
    print("[ORGANAGE] df shape:", df.shape)

    # 2) Build organ mapping (must match training scheme)
    organs = sorted(df["organ"].dropna().unique())
    organ_to_id = {org: i for i, org in enumerate(organs)}
    print("[ORGANAGE] Organ mapping:", organ_to_id)
    n_organs = len(organ_to_id)

    # 3) Init v4.5 model and load weights
    cols_r = [c for c in df.columns if c.startswith("z_rna_")]
    if not cols_r:
        raise RuntimeError("No z_rna_ columns found; cannot infer D for CrossFusionV45.")
    D = len(cols_r)
    print(f"[ORGANAGE] Initializing CrossFusionV45 with D={D}, n_organs={n_organs} ...")

    model = CrossFusionV45(
        emb_dim=D,
        organ_dim=64,
        n_organs=n_organs,
        d_model=256,
    ).to(device)
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state, strict=True)
    print(f"[ORGANAGE] Loaded model weights from {model_path}")

    # 4) Attach predictions for all modalities (μ and σ)
    df_with_pred = attach_predictions(df, model, organ_to_id, device=device, batch_size=2048)

    # 5) Build compact organ-age table
    base_cols = [
        "subject_id",
        "organ",
        "modality",
        "source",
        "age",
        "age_pred",
        "age_sigma",
    ]
    missing = [c for c in base_cols if c not in df_with_pred.columns]
    if missing:
        raise RuntimeError(f"[ORGANAGE] Missing expected columns: {missing}")

    out_df = df_with_pred[base_cols].rename(columns={"age": "age_chrono"})

    # 6) Fit organ-specific normative curves + deltas + zscores
    out_df = fit_normative_curves(
        out_df,
        healthy_sources=healthy_sources,
        min_ref_n=args.min_ref_n,
    )

    # 7) Save result (same filename as v4 so downstream scripts still work)
    out_df.to_parquet(out_path)
    print(f"[ORGANAGE] Saved organ-age table -> {out_path}")
    print("[ORGANAGE] Done.")


if __name__ == "__main__":
    main()
