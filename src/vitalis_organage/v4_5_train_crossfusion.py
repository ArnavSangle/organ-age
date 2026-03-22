"""
v4_5_train_crossfusion.py
==========================
Training script for the ``CrossFusionV45`` organ-age regression model.

Pipeline overview
-----------------
1. Load the v3 aligned-contrastive embedding parquet
   (``data/processed/aligned/v3_aligned_contrastive.parquet``).
2. Build per-modality (RNA / X-ray / MRI) data blocks from the
   ``z_rna_*``, ``z_xray_*``, and ``z_mri_*`` embedding columns,
   filtering out rows with non-finite embeddings, missing ages, or
   missing organ labels.
3. Concatenate all modality blocks, shuffle, and split 85 / 15 into
   train and validation sets.
4. Train ``CrossFusionV45`` with:
     - ``AdamW`` optimiser (default lr=2e-4, weight_decay=1e-4)
     - Heteroscedastic Gaussian NLL loss (``gaussian_nll``)
     - Optional mixed-precision (``--amp``) via ``torch.cuda.amp``
     - Gradient clipping at max_norm=1.0
5. Save the best validation checkpoint to
   ``models/v4_5/fusion_cross_v4_5.pt``.

Typical usage::

    python v4_5_train_crossfusion.py --device cuda --epochs 40 --amp
"""
import argparse
import time
import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from pathlib import Path
from tqdm.auto import tqdm

from v4_5_crossfusion_model import CrossFusionV45, gaussian_nll


def build_dataset(df: pd.DataFrame):
    """
    Build clean dataset from v3_aligned_contrastive.parquet.

    Uses z_rna_*, z_xray_*, z_mri_* blocks and organ labels.
    Drops any rows with non-finite embeddings, age, or missing organ.
    """
    # Map organs to IDs
    organs = sorted(df["organ"].dropna().unique())  
    organ_to_id = {org: i for i, org in enumerate(organs)}
    print("[DATA] Organs:", organ_to_id)

    def _block(modality: str, prefix: str):
        """
        Extract a clean (X, y, organ_ids) block for one modality.

        Filters the global DataFrame to rows where ``modality`` matches,
        selects the embedding columns identified by ``prefix``, and drops
        rows with non-finite embeddings, non-finite ages, or missing organ
        labels.

        Parameters
        ----------
        modality : str
            One of ``'rna'``, ``'xray'``, ``'mri'``.
        prefix : str
            Column prefix for the embedding dimensions, e.g. ``'z_rna_'``.

        Returns
        -------
        tuple[np.ndarray, np.ndarray, np.ndarray] or (None, None, None)
            ``(X, y, organ_ids)`` arrays ready for training, or three
            ``None`` values if no valid data exists for the modality.
        """
        cols = [c for c in df.columns if c.startswith(prefix)]
        if not cols:
            print(f"[DATA] No columns for prefix '{prefix}', skipping modality '{modality}'")
            return None, None, None

        sub = df[df["modality"] == modality].copy()
        if sub.empty:
            print(f"[DATA] No rows for modality '{modality}'")
            return None, None, None

        X = sub[cols].to_numpy(dtype="float32")
        y = sub["age"].to_numpy(dtype="float32")
        organs_raw = sub["organ"].to_numpy()

        # Finite masks
        mask_X = np.isfinite(X).all(axis=1)
        mask_y = np.isfinite(y)
        # organs_raw is already a NumPy array; just use pd.notna directly
        mask_o = pd.notna(organs_raw)

        mask = mask_X & mask_y & mask_o
        if mask.sum() == 0:
            print(f"[DATA] All rows invalid for modality '{modality}', skipping.")
            return None, None, None

        X = X[mask]
        y = y[mask]
        organs_clean = organs_raw[mask]

        # Map organs to ids using vectorized mapping via pandas for speed
        organ_ids = pd.Series(organs_clean).map(organ_to_id).to_numpy(dtype="int64")

        print(
            f"[DATA] {modality:5s} | kept {X.shape[0]:6d} rows | "
            f"features={X.shape[1]}"
        )

        return X, y, organ_ids

    # Per-modality blocks
    Xr, yr, or_ids = _block("rna", "z_rna_")
    Xx, yx, ox_ids = _block("xray", "z_xray_")
    Xm, ym, om_ids = _block("mri", "z_mri_")

    X_list, y_list, o_list = [], [], []
    for Xb, yb, ob in [(Xr, yr, or_ids), (Xx, yx, ox_ids), (Xm, ym, om_ids)]:
        if Xb is None:
            continue
        X_list.append(Xb)
        y_list.append(yb)
        o_list.append(ob)

    if not X_list:
        raise RuntimeError("[DATA] No valid data found for any modality.")

    X = np.concatenate(X_list, axis=0)
    y = np.concatenate(y_list, axis=0)
    o = np.concatenate(o_list, axis=0)

    # Final paranoia clean
    mask_all = np.isfinite(X).all(axis=1) & np.isfinite(y) & np.isfinite(o)
    X = X[mask_all]
    y = y[mask_all]
    o = o[mask_all]

    # Shuffle once
    idx = np.random.permutation(len(y))
    X = X[idx]
    y = y[idx]
    o = o[idx]

    print("[DATA] Final: X={}, y={}, organs={}".format(X.shape, y.shape, o.shape))
    return X, y, o, organ_to_id


def main():
    """
    Command-line entry point for the v4.5 CrossFusion training run.

    Parses all hyperparameter flags, builds the dataset, constructs the
    model, and runs the training loop with validation and checkpointing.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str,
                        default="data/processed/aligned/v3_aligned_contrastive.parquet")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch", type=int, default=256)
    parser.add_argument("--val-batch", type=int, default=2048)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) // 2))
    parser.add_argument("--amp", action="store_true", help="Enable mixed precision (CUDA)")
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    args = parser.parse_args()

    print(f"[CONFIG] device={args.device}  batch={args.batch}  val_batch={args.val_batch}  workers={args.workers}  amp={args.amp}")

    if args.device == "cuda" and not torch.cuda.is_available():
        print("[WARN] CUDA not available, using CPU.")
        args.device = "cpu"

    # Enable cuDNN autotuner when input sizes are stable
    if args.device == "cuda":
        torch.backends.cudnn.benchmark = True

    df = pd.read_parquet(args.data)
    X, y, o, organ_to_id = build_dataset(df)

    Xt = torch.tensor(X, dtype=torch.float32)
    yt = torch.tensor(y, dtype=torch.float32)
    ot = torch.tensor(o, dtype=torch.long)

    N = len(y)
    split = int(0.85 * N)

    train_ds = TensorDataset(Xt[:split], yt[:split], ot[:split])
    val_ds = TensorDataset(Xt[split:], yt[split:], ot[split:])

    pin_memory = True if args.device == "cuda" else False
    train_dl = DataLoader(
        train_ds,
        batch_size=args.batch,
        shuffle=True,
        drop_last=False,
        num_workers=args.workers,
        pin_memory=pin_memory,
    )
    val_dl = DataLoader(
        val_ds,
        batch_size=args.val_batch,
        shuffle=False,
        drop_last=False,
        num_workers=max(1, min(4, args.workers)),
        pin_memory=pin_memory,
    )

    model = CrossFusionV45(
        emb_dim=256,
        organ_dim=64,
        n_organs=len(organ_to_id),
        d_model=256,
    ).to(args.device)

    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # Mixed precision scaler (only used when amp enabled and cuda)
    use_amp = args.amp and args.device == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    best_nll = float("inf")
    outdir = Path("models/v4_5")
    outdir.mkdir(parents=True, exist_ok=True)
    outpath = outdir / "fusion_cross_v4_5.pt"

    print("[TRAIN] Starting v4.5 training...")

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()
        model.train()

        train_loss_sum = 0.0
        train_loss_count = 0

        # Training loop
        for xb, yb, ob in train_dl:
            xb = xb.to(args.device, non_blocking=True)
            yb = yb.to(args.device, non_blocking=True)
            ob = ob.to(args.device, non_blocking=True)

            optim.zero_grad()
            with torch.cuda.amp.autocast(enabled=use_amp):
                mu, sigma = model(xb, ob)
                loss = gaussian_nll(mu, sigma, yb)

            if not torch.isfinite(loss):
                print("[WARN] Non-finite train loss encountered; skipping batch.")
                continue

            # Scale, backward, unscale for clipping, step
            scaler.scale(loss).backward()
            scaler.unscale_(optim)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optim)
            scaler.update()

            train_loss_sum += float(loss.detach().cpu().item())
            train_loss_count += 1

        # Validation
        model.eval()
        val_loss_sum = 0.0
        val_loss_count = 0
        val_sigma_sum = 0.0

        with torch.no_grad():
            for xb, yb, ob in val_dl:
                xb = xb.to(args.device, non_blocking=True)
                yb = yb.to(args.device, non_blocking=True)
                ob = ob.to(args.device, non_blocking=True)

                with torch.cuda.amp.autocast(enabled=use_amp):
                    mu, sigma = model(xb, ob)
                    vloss = gaussian_nll(mu, sigma, yb)

                if not torch.isfinite(vloss):
                    print("[WARN] Non-finite val loss encountered; skipping batch.")
                    continue

                val_loss_sum += float(vloss.cpu().item())
                val_loss_count += 1
                val_sigma_sum += float(sigma.mean().cpu().item())

        train_nll = (train_loss_sum / train_loss_count) if train_loss_count else float("nan")
        val_nll = (val_loss_sum / val_loss_count) if val_loss_count else float("nan")
        mean_sigma = (val_sigma_sum / val_loss_count) if val_loss_count else float("nan")

        epoch_sec = time.time() - epoch_start
        examples = (train_loss_count * args.batch)
        throughput = examples / epoch_sec if epoch_sec > 0 else float("nan")

        print(
            f"[E{epoch:02d}] train NLL={train_nll:.4f} | val NLL={val_nll:.4f} | mean σ={mean_sigma:.3f} | "
            f"time={epoch_sec:.1f}s | ~{throughput:.0f} ex/s"
        )

        if np.isfinite(val_nll) and val_nll < best_nll:
            best_nll = val_nll
            torch.save(model.state_dict(), outpath)
            print(f"[SAVE] Best model saved → {outpath}")


if __name__ == "__main__":
    main()
