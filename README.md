# Organ-Age

**Multimodal Fusion of Transcriptomic and Radiological Signals for Organ-Resolved Biological Age Estimation**

Arnav Sangle — Independent Researcher, Ranchview High School, Irving TX

---

## Overview

Biological aging does not proceed uniformly across the body — different organs accumulate molecular, structural, and physiological damage at different rates. Organ-Age is a multimodal machine learning system that estimates organ-resolved biological age by jointly modeling gene expression and medical imaging data.

The current model (v4.5) achieves a mean absolute error of ~9.3 years and produces per-organ age predictions with residual "organ-age deltas" that expose patterns of accelerated and decelerated aging consistent with known tissue-level biology.

## Data Sources

| Modality | Dataset | Description |
|---|---|---|
| Gene expression (RNA-seq) | GTEx v10 | Bulk RNA-seq across 10+ tissue types |
| Chest X-ray | CheXpert | ~224K frontal chest radiographs |
| Brain MRI | IXI | T1-weighted brain MRI scans |

## Project Structure

```
organ-age/
├── src/
│   ├── vitalis_organage/     # Active v4.5 pipeline (training, inference, explainability)
│   ├── synapse_aligned/      # v3.5 cross-modality contrastive alignment
│   ├── figures/              # Figure generation scripts
│   ├── organon_multimodal/   # v3 multimodal baseline
│   ├── somatic_baseline/     # v1 PCA/MLP baseline
│   └── common/               # Shared utilities (gene map prep)
├── data/
│   ├── raw/                  # Original datasets (not tracked)
│   ├── interim/              # Intermediate preprocessing outputs
│   ├── processed/            # Final processed embeddings and features
│   └── analysis/             # Model outputs, QC, summaries
├── models/                   # Saved model weights (not tracked)
│   └── v4_5/                 # Production fusion transformer
├── figures/                  # Generated visualizations by version
├── reports/                  # PDF reports (clinical, ISEF, technical)
├── experiments/              # Training runs and logs
├── analysis/                 # Gene importance, integrated gradients outputs
├── config/                   # Project configuration
├── env/                      # Conda environment spec
├── docs/                     # Development notes
├── archive_unused/           # Superseded versions (v1, v2, v3 legacy)
└── Paper.tex / Paper.pdf     # Academic paper
```

## Model Architecture (v4.5)

- **Unimodal encoders**: Separate MLP encoders for RNA-seq, X-ray embeddings, and MRI embeddings
- **Contrastive alignment**: Cross-modal projectors trained with contrastive loss to align subject representations across modalities
- **Fusion transformer**: Cross-attention transformer that merges aligned multimodal tokens into a single age prediction per organ
- **Calibration**: Post-hoc isotonic regression calibration per organ

## Outputs

- Per-organ age predictions (brain cortex, heart, kidney, liver, lung, adipose, colon, skeletal muscle, skin, whole blood)
- Organ-age deltas (predicted age − chronological age)
- Gene importance scores and integrated gradients per organ (v4.5)
- Clinical, ISEF, and technical PDF reports per subject

## Setup

```bash
conda env create -f env/conda-env.yml
conda activate organ-age
```

## Version History

| Version | Module | Description |
|---|---|---|
| v1 | `somatic_baseline` | PCA + MLP unimodal baselines |
| v2 | `chimera_fusion` | Early contrastive fusion experiments |
| v3 | `organon_multimodal` | Multimodal transformer baseline |
| v3.5 | `synapse_aligned` | Cross-modality contrastive alignment |
| v4.5 | `vitalis_organage` | Production model with explainability |

Superseded versions are archived in `archive_unused/`.
