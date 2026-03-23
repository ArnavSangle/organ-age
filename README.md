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
├── papers/                   # All .tex, .bib, .pdf paper files
├── results/                  # Computed metrics, statistical tests
├── archive_unused/           # Superseded versions (v1, v2, v3 legacy)
└── run_pipeline.bat          # End-to-end reproducibility script
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
- Statistical validation (`results/`):
  - Bootstrap 95% CIs on MAE (overall, per-modality, per-organ; 10,000 replicates)
  - Wilcoxon signed-rank tests on per-organ age-gap distributions
  - Pairwise bootstrap MAE comparisons between organs
  - Ablation metrics (unimodal vs naive fusion vs aligned fusion)

## Setup

```bash
conda env create -f env/conda-env.yml
conda activate organ-age
```

## Reproducing the Study

Place the raw datasets under `data/raw/` (GTEx v10, CheXpert, IXI), then run the full pipeline:

```bash
run_pipeline.bat              # Full pipeline (GPU auto-detected)
run_pipeline.bat cpu          # CPU-only mode
run_pipeline.bat --from 5     # Resume from a specific stage
```

The pipeline runs 9 stages: unimodal encoder training, contrastive alignment, cross-fusion transformer training, normative table generation, calibration, inference, explainability, metrics/statistical tests, and figure generation. Each stage exits on failure with a clear error message.

## Version History

| Version | Module | Description |
|---|---|---|
| v1 | `somatic_baseline` | PCA + MLP unimodal baselines |
| v2 | `chimera_fusion` | Early contrastive fusion experiments |
| v3 | `organon_multimodal` | Multimodal transformer baseline |
| v3.5 | `synapse_aligned` | Cross-modality contrastive alignment |
| v4.5 | `vitalis_organage` | Production model with explainability |

Superseded versions are archived in `archive_unused/`.

## Citation

If you use this work, please cite:

```
Sangle, A. (2026). Organ-Age: A Multimodal Fusion of Transcriptomic and Radiological Signals
for Organ-Resolved Biological Age Estimation. Zenodo.
https://doi.org/10.5281/zenodo.19197649
```
