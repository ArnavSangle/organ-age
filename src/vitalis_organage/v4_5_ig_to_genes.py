"""
v4_5_ig_to_genes.py
====================
Project per-organ latent-space IG importance vectors back to gene space.

For each organ listed in ``ORGANS``:
  1. Load the latent IG vector (``D``-dimensional) from
     ``analysis/v4_5_ig_latent/latent_ig_<organ>.npy``.
  2. Load the RNA projector weight matrix ``W`` of shape ``(n_genes, D)``
     from ``models/v4_5/rna_projector_weights.npy``.
  3. Compute gene-level scores via the dot product:
         gene_scores = W @ ig_vec   -> shape (n_genes,)
  4. Sort genes by absolute score and write a ranked CSV to
     ``analysis/v4_5_gene_importance/gene_importance_<organ>.csv``.

Gene names are obtained by inspecting the column names of the unified GTEx
parquet (``data/processed/unified/unified_gtex.parquet``), supporting both
wide (one column per gene) and compressed (single vector column) formats.
"""
import numpy as np
import pandas as pd
from pathlib import Path


# -------- CONFIG: EDIT THESE TO MATCH YOUR IG FILES -------- #

# IG vectors should be 1D numpy arrays of length D (e.g. 256)
# Example filenames: latent_ig_liver.npy, latent_ig_heart.npy, etc.
ORGANS = ["liver", "heart", "kidney", "brain_cortex"]  # adjust as needed

IG_DIRNAME = "v4_5_ig_latent"          # folder under analysis/ where IG .npy live
IG_BASENAME_TEMPLATE = "latent_ig_{organ}.npy"

# ----------------------------------------------------------- #


def get_project_root() -> Path:
    """
    Return the repository root (two levels above this source file).

    Returns
    -------
    Path
        Absolute path to the repository root.
    """
    # organ-age/src/vitalis_organage/v4_5_ig_to_genes.py
    return Path(__file__).resolve().parents[2]


def load_gene_names(unified_gtex_path: Path, n_genes_expected: int) -> list[str]:
    """
    Derive the ordered list of gene names from the unified GTEx parquet.

    Supports two storage layouts:
      * **Wide format** – one column per gene; detected when the number of
        non-metadata columns equals ``n_genes_expected``.
      * **Compressed format** – a single ``features`` / ``expression`` /
        ``values`` column holding a vector; gene names are synthesised as
        ``gene_0``, ``gene_1``, … in that case.

    Parameters
    ----------
    unified_gtex_path : Path
        Path to the unified GTEx parquet whose column names encode gene IDs.
    n_genes_expected : int
        Expected number of genes (must match the first dimension of ``W``).

    Returns
    -------
    list[str]
        Ordered list of gene name strings, length ``n_genes_expected``.

    Raises
    ------
    RuntimeError
        If the column layout cannot be matched to either supported format.
    """
    print(f"[GENES] Loading columns from {unified_gtex_path}")
    df_head = pd.read_parquet(unified_gtex_path, engine="pyarrow").head(1)

    meta_cols = {
        "sample_id",
        "subject_id",
        "age",
        "sex",
        "organ",
        "modality",
        "source",
    }

    # Non-metadata columns
    gene_cols = [c for c in df_head.columns if c not in meta_cols]

    # Case 1: “wide” format – one column per gene
    if len(gene_cols) == n_genes_expected:
        print(f"[GENES] Detected {len(gene_cols)} separate gene columns.")
        return gene_cols

    # Case 2: “compressed” format – a single 'features' vector column
    if len(gene_cols) == 1 and gene_cols[0] in {"features", "expression", "values"}:
        col = gene_cols[0]
        print(f"[GENES] Detected single vector column '{col}'. "
              f"Using index-based gene names gene_0..gene_{n_genes_expected-1}.")
        return [f"gene_{i}" for i in range(n_genes_expected)]

    # Otherwise, we genuinely don't know what’s going on
    raise RuntimeError(
        f"[GENES] Expected either {n_genes_expected} separate gene columns "
        f"or a single 'features' column, but found columns: {gene_cols}"
    )


def main():
    """
    Entry point for the IG-to-gene projection pipeline.

    Loads IG vectors and the projector weight matrix, computes and ranks
    per-organ gene scores, and writes one CSV per organ to
    ``analysis/v4_5_gene_importance/``.
    """
    project_root = get_project_root()
    print(f"[ROOT] Project root: {project_root}")

    # Paths
    unified_gtex_path = project_root / "data" / "processed" / "unified" / "unified_gtex.parquet"
    weights_path = project_root / "models" / "v4_5" / "rna_projector_weights.npy"
    ig_dir = project_root / "analysis" / IG_DIRNAME
    out_dir = project_root / "analysis" / "v4_5_gene_importance"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load projector weights
    print(f"[WEIGHTS] Loading RNA projector weights from {weights_path}")
    W = np.load(weights_path)   # shape (n_genes, D)
    if W.ndim != 2:
        raise RuntimeError(f"[WEIGHTS] Expected 2D weights, got shape {W.shape}")
    n_genes, D = W.shape
    print(f"[WEIGHTS] Shape = (n_genes={n_genes}, D={D})")

    # Load gene names
    gene_names = load_gene_names(unified_gtex_path, n_genes_expected=n_genes)

    # Process each organ
    for organ in ORGANS:
        ig_path = ig_dir / IG_BASENAME_TEMPLATE.format(organ=organ)
        if not ig_path.exists():
            print(f"[IG] Skipping {organ}: {ig_path} not found.")
            continue

        print(f"[IG] Loading latent IG for organ='{organ}' from {ig_path}")
        ig_vec = np.load(ig_path)

        ig_vec = np.asarray(ig_vec).astype(float).reshape(-1)
        if ig_vec.shape[0] != D:
            raise RuntimeError(
                f"[IG] For organ={organ}, IG length={ig_vec.shape[0]} but "
                f"projector latent dim D={D}. They must match."
            )

        # gene_scores = W @ ig_vec   -> shape (n_genes,)
        gene_scores = W @ ig_vec

        abs_scores = np.abs(gene_scores)
        ranks = abs_scores.argsort()[::-1]  # descending

        sorted_gene_names = [gene_names[i] for i in ranks]
        sorted_scores = gene_scores[ranks]
        sorted_abs = abs_scores[ranks]

        df_out = pd.DataFrame(
            {
                "organ": organ,
                "gene": sorted_gene_names,
                "score_signed": sorted_scores,
                "score_abs": sorted_abs,
                "rank": np.arange(1, n_genes + 1),
            }
        )

        out_path = out_dir / f"gene_importance_{organ}.csv"
        df_out.to_csv(out_path, index=False)
        print(f"[OUT] Saved gene importance for organ='{organ}' -> {out_path}")

    print("[DONE] Gene-level importance CSVs written.")


if __name__ == "__main__":
    main()
