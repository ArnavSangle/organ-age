"""
make_all_figures.py

Command-line driver that generates the complete publication figure set for the
Organ-Age multimodal biological-age estimation paper.

The script accepts CSV prediction files for three model variants (unimodal,
baseline fusion v3, and aligned fusion v3.5) plus optional paths to embedding
arrays, training-curve logs, and an output directory.  It delegates all
rendering to the ``figlib`` helper library and writes PNG figures and a LaTeX
metrics table to the requested output directory.

Typical usage
-------------
    python make_all_figures.py \\
        --preds_unimodal results/unimodal_preds.csv \\
        --preds_v3       results/v3_preds.csv \\
        --preds_v35      results/v35_preds.csv \\
        --emb_unaligned  results/emb_unaligned.npy \\
        --emb_aligned    results/emb_aligned.npy \\
        --emb_meta       results/emb_meta.csv \\
        --curves_rna     results/rna_curves.csv \\
        --out_dir        figures/

Outputs (written to ``--out_dir``)
-----------------------------------
* conceptual_overview.png
* unimodal_pred_vs_true.png, v3_pred_vs_true.png, v35_pred_vs_true.png
* v35_age_accel_hist.png
* model_mae_comparison.png
* v35_metrics_table.tex  (use via \\input{figures/v35_metrics_table.tex})
* rna/xray/mri_training_curves.png  (when --curves_* args are supplied)
* umap_unaligned_by_modality.png, umap_unaligned_by_age.png  (optional)
* umap_aligned_by_modality.png,   umap_aligned_by_age.png    (optional)
"""

from __future__ import annotations

import os
import argparse
import numpy as np
import pandas as pd

from figlib import (
    Paths,
    _ensure_outdir,
    _read_preds,
    fig_conceptual_overview,
    fig_pred_vs_true,
    fig_age_accel,
    fig_model_comparison,
    fig_training_curves,
    fig_umap,
    write_metrics_table,
)

def main():
    """Parse CLI arguments, load prediction data, and render the full figure set.

    The function orchestrates the following steps:

    1. Parse command-line arguments and construct a ``Paths`` config object.
    2. Create the output directory if it does not already exist.
    3. Load prediction CSVs for each model variant via ``_read_preds``.
    4. Generate conceptual overview, scatter, histogram, and bar figures.
    5. Optionally render training-curve plots when log CSVs are supplied.
    6. Optionally project embeddings with UMAP and render coloured scatter
       plots when embedding arrays and metadata are supplied.

    All figures are saved as high-DPI PNGs; the metrics table is written
    as a LaTeX ``tabular`` fragment.
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds_unimodal", required=True, help="CSV with unimodal predictions")
    ap.add_argument("--preds_v3", required=True, help="CSV with baseline fusion v3 predictions")
    ap.add_argument("--preds_v35", required=True, help="CSV with aligned fusion v3.5 predictions")

    ap.add_argument("--emb_unaligned", default=None, help="NPY embeddings before alignment (optional)")
    ap.add_argument("--emb_aligned", default=None, help="NPY embeddings after alignment (optional)")
    ap.add_argument("--emb_meta", default=None, help="CSV metadata for embeddings (optional). Needs 'age' and 'modality' columns.")

    ap.add_argument("--curves_rna", default=None)
    ap.add_argument("--curves_xray", default=None)
    ap.add_argument("--curves_mri", default=None)

    ap.add_argument("--out_dir", default="figures")
    args = ap.parse_args()

    paths = Paths(
        preds_unimodal_csv=args.preds_unimodal,
        preds_v3_csv=args.preds_v3,
        preds_v35_csv=args.preds_v35,
        emb_unaligned_npy=args.emb_unaligned,
        emb_aligned_npy=args.emb_aligned,
        emb_meta_csv=args.emb_meta,
        curves_rna_csv=args.curves_rna,
        curves_xray_csv=args.curves_xray,
        curves_mri_csv=args.curves_mri,
        out_dir=args.out_dir,
    )

    _ensure_outdir(paths.out_dir)

    # -------------------------
    # Load predictions
    # -------------------------
    df_uni = _read_preds(paths.preds_unimodal_csv)
    df_v3 = _read_preds(paths.preds_v3_csv)
    df_v35 = _read_preds(paths.preds_v35_csv)

    # -------------------------
    # Conceptual + architecture figures (paper diagrams)
    # -------------------------
    fig_conceptual_overview(os.path.join(paths.out_dir, "conceptual_overview.png"))

    # -------------------------
    # Results figures
    # -------------------------
    fig_pred_vs_true(df_uni, os.path.join(paths.out_dir, "unimodal_pred_vs_true.png"),
                     title="Unimodal: predicted vs true age")
    fig_pred_vs_true(df_v3, os.path.join(paths.out_dir, "v3_pred_vs_true.png"),
                     title="Baseline fusion (v3): predicted vs true age")
    fig_pred_vs_true(df_v35, os.path.join(paths.out_dir, "v35_pred_vs_true.png"),
                     title="Aligned fusion (v3.5): predicted vs true age")

    fig_age_accel(df_v35, os.path.join(paths.out_dir, "v35_age_accel_hist.png"),
                  title="v3.5 age acceleration distribution")

    # Model comparison MAE
    def mae(df):
        """Return the mean absolute error between y_pred and y_true columns."""
        return float((df["y_pred"] - df["y_true"]).abs().mean())

    # Compute MAE for each model variant to drive the bar-chart comparison.
    mae_dict = {"Unimodal": mae(df_uni), "v3": mae(df_v3), "v3.5": mae(df_v35)}
    fig_model_comparison(mae_dict, os.path.join(paths.out_dir, "model_mae_comparison.png"))

    # Metrics table → LaTeX
    write_metrics_table(df_v35, os.path.join(paths.out_dir, "v35_metrics_table.tex"))

    # -------------------------
    # Training curves (optional)
    # -------------------------
    if paths.curves_rna_csv:
        fig_training_curves(paths.curves_rna_csv, os.path.join(paths.out_dir, "rna_training_curves.png"),
                            "RNA encoder training curves")
    if paths.curves_xray_csv:
        fig_training_curves(paths.curves_xray_csv, os.path.join(paths.out_dir, "xray_training_curves.png"),
                            "X-ray encoder training curves")
    if paths.curves_mri_csv:
        fig_training_curves(paths.curves_mri_csv, os.path.join(paths.out_dir, "mri_training_curves.png"),
                            "MRI encoder training curves")

    # -------------------------
    # UMAP (optional but recommended)
    # -------------------------
    if paths.emb_unaligned_npy and paths.emb_meta_csv:
        emb = np.load(paths.emb_unaligned_npy)
        meta = pd.read_csv(paths.emb_meta_csv)
        fig_umap(emb, meta, os.path.join(paths.out_dir, "umap_unaligned_by_modality.png"),
                 title="UMAP of unaligned embeddings (colored by modality)", color_by="modality")
        fig_umap(emb, meta, os.path.join(paths.out_dir, "umap_unaligned_by_age.png"),
                 title="UMAP of unaligned embeddings (colored by age)", color_by="age")

    if paths.emb_aligned_npy and paths.emb_meta_csv:
        emb = np.load(paths.emb_aligned_npy)
        meta = pd.read_csv(paths.emb_meta_csv)
        fig_umap(emb, meta, os.path.join(paths.out_dir, "umap_aligned_by_modality.png"),
                 title="UMAP of aligned embeddings (colored by modality)", color_by="modality")
        fig_umap(emb, meta, os.path.join(paths.out_dir, "umap_aligned_by_age.png"),
                 title="UMAP of aligned embeddings (colored by age)", color_by="age")

    print(f"[OK] Figures written to: {paths.out_dir}")
    print("[OK] Table written to: figures/v35_metrics_table.tex (input via \\input{figures/v35_metrics_table.tex})")


if __name__ == "__main__":
    main()
