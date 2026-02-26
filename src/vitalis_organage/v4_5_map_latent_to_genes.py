"""
v4_5_map_latent_to_genes.py
============================
Map latent-space IG importance back to individual genes for a single organ.

Given an IG CSV (columns: ``feature``, ``mean_abs_ig``) produced by
``v4_5_explain_ig.py`` and the RNA projector weight matrix ``W`` of shape
``(n_genes, D)``, this script:

  1. Selects the top ``--top_dims`` latent dimensions by mean absolute IG.
  2. For each selected latent dimension ``k``, retrieves the column ``W[:, k]``
     (the weight vector mapping genes to that latent dimension) and ranks genes
     by ``|W[i, k]|``.
  3. Emits a long-form table with columns:
       [organ, latent_dim, ig_importance, gene_rank, gene_name,
        gene_weight, gene_weight_abs]

This provides a *per-latent-dimension* window into which genes most strongly
drive the age prediction signal.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def load_ig_dims(ig_csv: Path, top_dims: int):
    """
    Load IG CSV and return a DataFrame with columns:
    ['dim', 'mean_abs_ig'], sorted by importance.
    """
    df = pd.read_csv(ig_csv)
    if "mean_abs_ig" not in df.columns or "feature" not in df.columns:
        raise RuntimeError(f"{ig_csv} missing required columns ['feature', 'mean_abs_ig'].")

    df = df.copy()
    df["dim"] = (
        df["feature"]
        .str.replace("z_rna_", "", regex=False)
        .astype(int)
    )
    df = df.sort_values("mean_abs_ig", ascending=False)
    df_top = df.head(top_dims)[["dim", "mean_abs_ig"]]
    return df_top


def load_weight_matrix(weight_path: Path, max_dim: int):
    """
    Load RNA projector weight matrix from .npy.
    Tries to infer orientation:
      - if shape = (n_genes, D) and D > max_dim -> treat as (n_genes, D)
      - if shape = (D, n_genes) and D > max_dim -> treat as (D, n_genes)
    Returns:
      W_genes_by_dim: np.ndarray shape (n_genes, D)
    """
    W = np.load(weight_path)
    if W.ndim != 2:
        raise RuntimeError(f"Expected 2D weight matrix, got shape {W.shape} from {weight_path}")

    n0, n1 = W.shape
    if n1 > max_dim and n0 <= n1:
        # assume (n_genes, D)
        W_gd = W
    elif n0 > max_dim and n0 >= n1:
        # assume (D, n_genes) -> transpose
        W_gd = W.T
    else:
        # fallback: if n1 >= max_dim, treat as (n_genes, D)
        if n1 >= max_dim:
            W_gd = W
        else:
            W_gd = W.T

    print(f"[MAP] Loaded weight matrix from {weight_path}, "
          f"interpreted as (n_genes, D) = {W_gd.shape}")
    return W_gd


def load_gene_names(genes_path: Path):
    """
    Load gene names from CSV/Parquet. Tries common column names.
    """
    if genes_path.suffix.lower() == ".parquet":
        df = pd.read_parquet(genes_path)
    else:
        df = pd.read_csv(genes_path)

    for col in ["gene", "gene_name", "symbol", "gene_symbol"]:
        if col in df.columns:
            names = df[col].astype(str).tolist()
            print(f"[MAP] Loaded {len(names)} gene names from column '{col}' in {genes_path}")
            return names

    raise RuntimeError(
        f"Could not find a gene-name column in {genes_path}. "
        f"Expected one of ['gene', 'gene_name', 'symbol', 'gene_symbol']."
    )


def build_latent_to_gene_table(
    ig_csv: Path,
    weight_path: Path,
    genes_path: Path,
    top_dims: int,
    top_genes: int,
    organ_label: str,
):
    """
    For a given organ's IG CSV:
      - pick top_dims latent dimensions by IG
      - for each dim, rank genes by |weight|
      - return a long-form DataFrame with columns:
        [organ, latent_dim, ig_importance, gene_name, gene_rank, gene_weight_abs]
    """
    ig_top = load_ig_dims(ig_csv, top_dims)
    max_dim = ig_top["dim"].max()

    W_gd = load_weight_matrix(weight_path, max_dim=max_dim)
    n_genes, D = W_gd.shape
    if max_dim >= D:
        raise RuntimeError(
            f"[MAP] Max IG dim index {max_dim} >= D={D} in weight matrix; "
            f"check that your RNA weights and IG dims use the same dimension size."
        )

    gene_names = load_gene_names(genes_path)
    if len(gene_names) != n_genes:
        print(
            f"[MAP] WARNING: gene name count ({len(gene_names)}) "
            f"!= n_genes in weight matrix ({n_genes}). "
            f"Truncating to min length."
        )
        n_min = min(len(gene_names), n_genes)
        gene_names = gene_names[:n_min]
        W_gd = W_gd[:n_min, :]

    records = []
    for _, row in ig_top.iterrows():
        dim = int(row["dim"])
        ig_importance = float(row["mean_abs_ig"])

        # weights for this latent dim: shape (n_genes,)
        w = W_gd[:, dim]
        abs_w = np.abs(w)

        # top-k gene indices
        idx_sorted = np.argsort(-abs_w)  # descending
        idx_top = idx_sorted[:top_genes]

        for rank, gi in enumerate(idx_top, start=1):
            records.append(
                {
                    "organ": organ_label,
                    "latent_dim": dim,
                    "ig_importance": ig_importance,
                    "gene_rank": rank,
                    "gene_name": gene_names[gi],
                    "gene_weight": float(w[gi]),
                    "gene_weight_abs": float(abs_w[gi]),
                }
            )

    out_df = pd.DataFrame.from_records(records)
    return out_df


def main():
    """
    Command-line entry point for the latent-to-gene mapping.

    Reads the IG CSV, weight matrix, and gene-name file, constructs the
    per-latent-dimension gene ranking table via
    ``build_latent_to_gene_table``, and writes the result to ``--out_csv``.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ig_csv",
        type=str,
        required=True,
        help="IG CSV for a single organ+modality (e.g. liver+RNA).",
    )
    parser.add_argument(
        "--rna_weight_npy",
        type=str,
        default="models/organon_unimodal/rna_projector_weights.npy",
        help="Path to RNA projection/encoder weight matrix (.npy).",
    )
    parser.add_argument(
        "--genes_path",
        type=str,
        default="data/processed/rna_gene_metadata.csv",
        help="CSV/Parquet with gene names aligned to the weight matrix.",
    )
    parser.add_argument(
        "--organ_label",
        type=str,
        default="liver",
        help="Label to put in the output table (e.g., 'liver', 'brain_cortex').",
    )
    parser.add_argument(
        "--top_dims",
        type=int,
        default=10,
        help="Number of top latent dims by IG to keep.",
    )
    parser.add_argument(
        "--top_genes",
        type=int,
        default=10,
        help="Number of top genes per latent dim.",
    )
    parser.add_argument(
        "--out_csv",
        type=str,
        default="data/analysis/v4_5_latent_to_genes.csv",
    )
    args = parser.parse_args()

    ig_csv = Path(args.ig_csv)
    weight_path = Path(args.rna_weight_npy)
    genes_path = Path(args.genes_path)
    out_path = Path(args.out_csv)

    out_df = build_latent_to_gene_table(
        ig_csv=ig_csv,
        weight_path=weight_path,
        genes_path=genes_path,
        top_dims=args.top_dims,
        top_genes=args.top_genes,
        organ_label=args.organ_label,
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False)
    print(f"[MAP] Saved latent→gene table -> {out_path}")
    print("[MAP] Example rows:")
    print(out_df.head(15))


if __name__ == "__main__":
    main()
