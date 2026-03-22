"""
synapse_aligned package.

This package implements the multimodal biological-age alignment and prediction
pipeline (v3 / v3.5).  It is organised into three sub-packages:

  alignment/    - scripts that learn a shared embedding space across RNA, X-ray
                  and MRI modalities (contrastive and supervised alignment).
  training/     - downstream fusion-transformer models that predict biological
                  age from the aligned embeddings.
  evaluation/   - evaluation scripts that compute MSE / MAE metrics, optionally
                  grouped by modality, organ, or age bin, and compare model
                  versions.
  visualization/ - UMAP plotting utilities for inspecting the embedding geometry.
"""
