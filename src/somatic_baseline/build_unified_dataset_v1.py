# scripts/70_build_unified_dataset.py
"""
Build a unified multimodal dataset from:
- GTEx v10 (RNA expression)
- CheXpert (chest X-ray embeddings)
- IXI (T1 MRI embeddings)

Output:
  data/processed/unified_gtex.parquet
  data/processed/unified_chexpert.parquet
  data/processed/unified_ixi.parquet
  data/processed/unified_all.parquet

Unified schema:
  sample_id : str
  subject_id: str
  age       : float
  sex       : str
  organ     : str      (heart, lung, brain, etc.)
  modality  : str      (rna, xray, mri)
  source    : str      (gtex_v10, chexpert, ixi)
  features  : np.ndarray (vector of floats)
"""

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(".")

# --------------------------- helpers ---------------------------

def clean_id(patient_id: str, study_id: str) -> str:
    """
    Convert patient00001 + study1 --> 'patient00001-study1'
    Must match scripts/11_chexpert_make_embeddings.py.
    """
    p = str(patient_id).strip()
    s = str(study_id).strip()
    return f"{p}-{s}"


def ensure_processed_dir():
    """Create and return the ``data/processed`` directory."""
    out = ROOT / "data" / "processed"
    out.mkdir(parents=True, exist_ok=True)
    return out


# --------------------------- GTEx loader ---------------------------

def load_gtex():
    """
    Load GTEx per-organ RNA expression from data/interim/{expression,metadata}.
    For each sample, we produce a single 'features' vector (genes).
    """
    expr_root = ROOT / "data" / "interim" / "expression"
    meta_root = ROOT / "data" / "interim" / "metadata"

    # organs used during preprocessing (scripts/02_gtex_process_all.py)
    organs = [
        "adipose",
        "brain_cortex",
        "colon",
        "heart",
        "kidney",
        "liver",
        "lung",
        "skeletal_muscle",
        "skin",
        "whole_blood",
    ]

    dfs = []
    print("[GTEx] Scanning organs...")
    for organ in organs:
        expr_path = expr_root / f"{organ}.parquet"
        meta_path = meta_root / f"{organ}.parquet"
        if not expr_path.exists() or not meta_path.exists():
            print(f"[GTEx][{organ}] Missing expression or metadata, skipping.")
            continue

        # expression is genes x samples; transpose to samples x genes
        E = pd.read_parquet(expr_path)
        E = E.T  # now rows = samples, cols = genes

        meta = pd.read_parquet(meta_path)  # index = sample_id
        # align to intersection
        common = E.index.intersection(meta.index)
        E = E.loc[common]
        meta = meta.loc[common]

        print(f"[GTEx] Loaded {organ}: shape={E.shape}")

        # build unified rows
        features = list(E.to_numpy(dtype=np.float32))
        df = pd.DataFrame(
            {
                "sample_id": common.astype(str),
                "subject_id": meta["subject_id"].astype(str),
                "age": meta["age_numeric"].astype(float),
                "sex": meta["sex_clean"].astype(str),
                "organ": organ,
                "modality": "rna",
                "source": "gtex_v10",
                "features": features,
            }
        )
        dfs.append(df)

    if not dfs:
        print("[GTEx] No organs loaded.")
        return pd.DataFrame()

    gtex = pd.concat(dfs, axis=0, ignore_index=True)
    print(f"[GTEx] Final GTEx unified shape: {gtex.shape}")
    return gtex


# --------------------------- CheXpert loader ---------------------------

def load_chexpert():
    """
    Load CheXpert embeddings from:
      data/interim/chexpert/embeddings_compressed/chexpert_embeddings.npy
      data/interim/chexpert/embeddings_compressed/chexpert_index.parquet
      data/interim/chexpert/metadata_train.parquet
      data/interim/chexpert/metadata_valid.parquet

    and build unified rows with 'features' = X-ray embedding.
    """
    base = ROOT / "data" / "interim" / "chexpert"
    emb_file = base / "embeddings_compressed" / "chexpert_embeddings.npy"
    index_file = base / "embeddings_compressed" / "chexpert_index.parquet"
    meta_train = base / "metadata_train.parquet"
    meta_valid = base / "metadata_valid.parquet"

    if not emb_file.exists() or not index_file.exists():
        print("[CheXpert] Missing compressed embeddings or index, skipping CheXpert.")
        return pd.DataFrame()

    if not meta_train.exists():
        print("[CheXpert] Missing metadata_train.parquet, skipping CheXpert.")
        return pd.DataFrame()

    emb = np.load(emb_file)  # (N, D)
    idx = pd.read_parquet(index_file)  # columns: sample_id, row
    print(f"[CheXpert] Loaded embedding matrix: {emb.shape}")
    print(f"[CheXpert] Loaded index: {idx.shape}")

    # metadata: train + valid
    meta = pd.read_parquet(meta_train)
    if meta_valid.exists():
        meta_val = pd.read_parquet(meta_valid)
        meta = pd.concat([meta, meta_val], axis=0, ignore_index=True)

    print("[CheXpert] Metadata columns:", list(meta.columns))

    required = {"patient_id", "study_id"}
    if not required.issubset(meta.columns):
        print("[CheXpert] Missing patient_id or study_id in metadata, skipping.")
        return pd.DataFrame()

    # reconstruct sample_id
    meta["sample_id"] = meta.apply(
        lambda r: clean_id(r["patient_id"], r["study_id"]), axis=1
    )
    meta = meta.drop_duplicates("sample_id")

    # join with index to get Age/Sex
    df = idx.merge(meta, on="sample_id", how="left")

    # attach embeddings
    df["features"] = df["row"].apply(lambda r: emb[int(r)].astype(np.float32))

    # clean label columns
    df["age"] = df.get("Age", np.nan).astype(float)
    df["sex"] = df.get("Sex", "U").astype(str)

    # subject_id = patient_id (string)
    df["subject_id"] = df.get("patient_id", "").astype(str)

    df["organ"] = "lung"
    df["modality"] = "xray"
    df["source"] = "chexpert"

    keep_cols = ["sample_id", "subject_id", "age", "sex", "organ", "modality", "source", "features"]
    df = df[keep_cols]

    print(f"[CheXpert] Final CheXpert unified shape: {df.shape}")
    return df


# --------------------------- IXI loader ---------------------------

def load_ixi():
    """
    Load per-subject IXI MRI embeddings from:
      data/processed/ixi/ixi_subject_embeddings.parquet

    Expected columns (from your scripts):
      - subject   (e.g. 'IXI002')
      - age
      - sex
      - embedding (vector)
    """
    emb_file = ROOT / "data" / "processed" / "ixi" / "ixi_subject_embeddings.parquet"

    if not emb_file.exists():
        print(f"[IXI] Embedding parquet not found at {emb_file}, skipping IXI.")
        return pd.DataFrame()

    df = pd.read_parquet(emb_file)
    print(f"[IXI] Loaded subject embeddings: shape={df.shape}")

    # Standardize column names
    if "subject" in df.columns and "subject_id" not in df.columns:
        df = df.rename(columns={"subject": "subject_id"})
    if "embedding" in df.columns:
        df = df.rename(columns={"embedding": "features"})

    df["sample_id"] = df["subject_id"].astype(str)
    df["age"] = df.get("age", np.nan).astype(float)
    df["sex"] = df.get("sex", "U").astype(str)

    df["organ"] = "brain"
    df["modality"] = "mri"
    df["source"] = "ixi"

    keep_cols = ["sample_id", "subject_id", "age", "sex", "organ", "modality", "source", "features"]
    df = df[keep_cols]

    print(f"[IXI] Final IXI unified shape: {df.shape}")
    return df


# --------------------------- main ---------------------------

def main():
    """
    Orchestrate loading of GTEx, CheXpert, and IXI data, save individual
    per-modality parquet files, then concatenate everything into a single
    ``data/processed/unified_all.parquet`` with the shared schema.
    """
    processed_dir = ensure_processed_dir()

    print("\n=== Building Unified Multimodal Dataset ===")

    # 1) GTEx
    gtex = load_gtex()
    if not gtex.empty:
        gtex.to_parquet(processed_dir / "unified_gtex.parquet")
        print(f"[SAVE] GTEx → {processed_dir / 'unified_gtex.parquet'}")

    # 2) CheXpert
    chexpert = load_chexpert()
    if not chexpert.empty:
        chexpert.to_parquet(processed_dir / "unified_chexpert.parquet")
        print(f"[SAVE] CheXpert → {processed_dir / 'unified_chexpert.parquet'}")

    # 3) IXI
    ixi = load_ixi()
    if not ixi.empty:
        ixi.to_parquet(processed_dir / "unified_ixi.parquet")
        print(f"[SAVE] IXI → {processed_dir / 'unified_ixi.parquet'}")

    # 4) Stack everything
    frames = [df for df in [gtex, chexpert, ixi] if not df.empty]
    if not frames:
        print("[WARN] No datasets loaded; nothing to unify.")
        return

    unified = pd.concat(frames, axis=0, ignore_index=True)

    # normalize ID types so parquet/pyarrow don't complain
    for col in ["sample_id", "subject_id"]:
        if col in unified.columns:
            unified[col] = unified[col].astype(str)

    out_all = processed_dir / "unified_all.parquet"
    unified.to_parquet(out_all)
    print(f"\n=== DONE: Unified dataset shape = {unified.shape} ===")
    print(f"Saved to: {out_all}")


if __name__ == "__main__":
    main()
