"""
v4_5_export_rna_weights.py
===========================
Export the RNA projector (encoder) weight matrix from a given checkpoint to a
``.npy`` file for use by the gene-importance attribution pipeline.

Given a checkpoint path this script:
  1. Loads the checkpoint, handling common wrapper keys
     (``state_dict``, ``model_state_dict``) and plain dicts.
  2. Finds the 2-D weight tensor whose one dimension equals ``--latent_dim``
     (default 256) and which has the largest total element count (a heuristic
     for identifying the gene-to-latent projection matrix).
  3. Orients the tensor to ``(n_genes, latent_dim)`` and saves it as a numpy
     array at ``--out_npy``.

The resulting file is consumed by ``v4_5_ig_to_genes.py`` and
``v4_5_map_latent_to_genes.py``.
"""
import argparse
from pathlib import Path

import numpy as np
import torch


def find_projector_weight(state_dict, latent_dim=256):
    """
    Heuristic:
      - look for 2D tensors where one dimension == latent_dim
      - pick the one with the largest total number of elements
      - return as (n_genes, latent_dim)
    """
    candidates = []

    for name, tensor in state_dict.items():
        if tensor.ndim != 2:
            continue
        h, w = tensor.shape
        if latent_dim in (h, w):
            candidates.append((name, tensor))

    if not candidates:
        raise RuntimeError(
            f"[EXPORT] No 2D weight with dim={latent_dim} found in state_dict."
        )

    # choose the largest by number of elements
    def numel(x):
        """Return the total element count of the tensor at index 1 of tuple x."""
        return x[1].numel()

    name, tensor = max(candidates, key=numel)
    h, w = tensor.shape

    print(f"[EXPORT] Selected weight '{name}' with shape {tuple(tensor.shape)}")

    # orient to (n_genes, latent_dim)
    if w == latent_dim:
        W = tensor.cpu().numpy()  # (n_genes, latent_dim)
    else:
        W = tensor.t().cpu().numpy()  # (latent_dim, n_genes) -> transpose

    print(f"[EXPORT] Interpreted as (n_genes, D) = {W.shape}")
    return W, name


def main():
    """
    Command-line entry point for the RNA weight exporter.

    Loads the checkpoint, extracts the projector weight via
    ``find_projector_weight``, and writes the oriented ``(n_genes, D)``
    matrix to ``--out_npy``.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to RNA encoder / projector checkpoint (.pt or .pth).",
    )
    parser.add_argument(
        "--latent_dim",
        type=int,
        default=256,
        help="Latent dim D that matches your z_rna_* embeddings.",
    )
    parser.add_argument(
        "--out_npy",
        type=str,
        default="models/v4_5/rna_projector_weights.npy",
        help="Where to save the (n_genes, D) matrix as .npy",
    )
    args = parser.parse_args()

    ckpt_path = Path(args.checkpoint)
    out_path = Path(args.out_npy)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[EXPORT] Loading checkpoint: {ckpt_path}")
    obj = torch.load(ckpt_path, map_location="cpu")

    # allow either full dict or {'model_state_dict': ...}
    if isinstance(obj, dict) and "state_dict" in obj:
        state_dict = obj["state_dict"]
    elif isinstance(obj, dict) and "model_state_dict" in obj:
        state_dict = obj["model_state_dict"]
    elif isinstance(obj, dict):
        # assume it's already a state_dict-like mapping
        state_dict = obj
    else:
        raise RuntimeError(
            f"[EXPORT] Unexpected checkpoint type: {type(obj)}; "
            f"expected a dict or state_dict."
        )

    W, name = find_projector_weight(state_dict, latent_dim=args.latent_dim)
    np.save(out_path, W)
    print(f"[EXPORT] Saved RNA projector weights from '{name}' -> {out_path}")


if __name__ == "__main__":
    main()
