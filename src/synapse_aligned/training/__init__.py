"""
training sub-package for synapse_aligned.

Contains scripts that train downstream fusion-transformer models on the
contrastive-aligned multimodal embeddings produced by the alignment scripts.

Modules
-------
train_fusion_transformer_cross_v3
    Trains the CrossFusion transformer that regresses biological age from the
    contrastive-aligned z_* embeddings (v3.5 pipeline).
"""
