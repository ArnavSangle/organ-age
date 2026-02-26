"""
v4_5_autofind_rna_weights.py
=============================
Automatically discover and export the RNA projector weight matrix from any
checkpoint file under the repository tree.

The script walks the entire project directory searching for ``.pt`` /
``.pth`` checkpoint files.  For each checkpoint it:
  1. Loads the file and extracts a state dict (handling ``state_dict`` /
     ``model_state_dict`` wrapper keys and plain dicts).
  2. Searches for 2-D tensors where one dimension equals ``LATENT_DIM``
     (256) and the other is at least 500 (a plausible gene count).
  3. Keeps the candidate with the largest gene-count dimension.

The selected weight matrix is oriented to ``(n_genes, D)`` and saved as
``models/v4_5/rna_projector_weights.npy``.

This is useful when the RNA encoder was saved as part of a larger checkpoint
and the exact weight-key name is unknown.
"""
import os
from pathlib import Path

import numpy as np
import torch


LATENT_DIM = 256  # matches your z_rna_* dim


def extract_state_dict(obj):
    """
    Try to get a state_dict from a loaded checkpoint.
    Handles common patterns like:
      - {'state_dict': ...}
      - {'model_state_dict': ...}
      - plain state_dict
    """
    if not isinstance(obj, dict):
        return None

    for key in ("state_dict", "model_state_dict"):
        if key in obj and isinstance(obj[key], dict):
            return obj[key]

    # fallback: assume the whole dict is a state_dict
    return obj


def find_candidate_weight(state_dict, latent_dim=LATENT_DIM, min_genes=500):
    """
    Look for 2D weights where one dimension == latent_dim.
    Return (tensor, name, n_genes) for the best candidate where the other
    dimension is large enough to plausibly be 'n_genes'.
    """
    best = None  # (tensor, name, n_genes)

    for name, tensor in state_dict.items():
        if not isinstance(tensor, torch.Tensor):
            continue
        if tensor.ndim != 2:
            continue

        h, w = tensor.shape
        if latent_dim not in (h, w):
            continue

        # interpret as (n_genes, D)
        if w == latent_dim:
            n_genes = h
        else:
            n_genes = w

        # we want something with lots of "genes"
        if n_genes < min_genes:
            continue

        if best is None or n_genes > best[2]:
            best = (tensor, name, n_genes)

    if best is None:
        return None, None, None

    return best


def main():
    """
    Walk the repository, find the best RNA projector weight candidate, orient
    it to ``(n_genes, D)``, and save it as a ``.npy`` file.

    Raises
    ------
    RuntimeError
        If no suitable candidate weight is found in any checkpoint under the
        repository root.
    """
    project_root = Path(__file__).resolve().parents[2]  # C:/.../organ-age
    out_path = project_root / "models" / "v4_5" / "rna_projector_weights.npy"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"[AUTO] Project root: {project_root}")
    print(f"[AUTO] Scanning for checkpoints under ENTIRE repo (this may take a bit)...")

    candidates = []  # list of (path, name, n_genes, tensor)

    # Scan everything under project_root (including archive_unused, scripts, etc.)
    for root, _, files in os.walk(project_root):
        for fname in files:
            if not fname.lower().endswith((".pt", ".pth")):
                continue

            ckpt_path = Path(root) / fname

            # Heuristic: skip obvious non-model stuff if needed
            # (right now we just keep it simple and try everything)
            print(f"[AUTO] Checking checkpoint: {ckpt_path}")

            try:
                obj = torch.load(ckpt_path, map_location="cpu")
            except Exception as e:
                print(f"[AUTO]  -> failed to load ({e}), skipping.")
                continue

            state_dict = extract_state_dict(obj)
            if state_dict is None:
                print("[AUTO]  -> no usable state_dict, skipping.")
                continue

            tensor, name, n_genes = find_candidate_weight(state_dict)
            if tensor is None:
                print("[AUTO]  -> no suitable (n_genes x D) weight found.")
                continue

            print(
                f"[AUTO]  -> candidate '{name}' with shape {tuple(tensor.shape)} "
                f"(n_genes ≈ {n_genes})"
            )
            candidates.append((ckpt_path, name, n_genes, tensor))

    if not candidates:
        raise RuntimeError(
            "[AUTO] No suitable RNA projector weights found in ANY checkpoint under the repo.\n"
            "Hint: the RNA encoder checkpoint might never have been saved, or it lives outside this repo."
        )

    # Pick the candidate with largest n_genes
    ckpt_path, name, n_genes, tensor = max(candidates, key=lambda x: x[2])
    h, w = tensor.shape
    print(
        f"[AUTO] Selected checkpoint:\n"
        f"       {ckpt_path}\n"
        f"       weight name='{name}', shape={tensor.shape}, n_genes={n_genes}"
    )

    # Orient to (n_genes, D)
    if w == LATENT_DIM:
        W = tensor.cpu().numpy()  # (n_genes, D)
    elif h == LATENT_DIM:
        W = tensor.t().cpu().numpy()  # (D, n_genes) -> (n_genes, D)
    else:
        raise RuntimeError(
            f"[AUTO] Selected tensor doesn't have dim={LATENT_DIM} in any axis."
        )

    print(f"[AUTO] Final matrix shape (n_genes, D) = {W.shape}")
    np.save(out_path, W)
    print(f"[AUTO] Saved RNA projector weights -> {out_path}")


if __name__ == "__main__":
    main()
