"""
v4_5_crossfusion_model.py
==========================
Defines the ``CrossFusionV45`` transformer model and the heteroscedastic
Gaussian negative log-likelihood loss used to train it.

Architecture overview
---------------------
``CrossFusionV45`` is an organ-conditioned, uncertainty-aware age regressor.
It takes a fused multi-modal embedding and an organ identifier as inputs and
predicts a Gaussian distribution over biological age (mu, sigma):

  1. The organ ID is embedded via a learned ``nn.Embedding`` and projected to
     ``d_model`` dimensions (the *organ token*).
  2. The input embedding is projected to ``d_model`` dimensions (the
     *embedding token*).
  3. Both tokens are stacked to form a 2-token sequence and passed through a
     multi-layer Transformer encoder (pre-norm, GELU activations).
  4. The embedding-token output drives two linear heads:
       - ``head_mu``    → predicted mean age (mu)
       - ``head_sigma`` → raw logit for aleatoric uncertainty, softplus-ed to
                          produce sigma > 0.

Loss
----
``gaussian_nll`` computes the heteroscedastic negative log-likelihood:

    NLL = 0.5 * (log(sigma^2) + (y - mu)^2 / sigma^2)

with safety clamping on variance and NaN/Inf sanitisation of all inputs.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossFusionV45(nn.Module):
    """
    Organ-conditioned, uncertainty-aware CrossFusion (v4.5)

    Inputs:
        x_emb : (B, D) fused aligned embedding (D=256)
        organ_id : (B,) long tensor in [0, n_organs-1]

    Outputs:
        mu : predicted organ age
        sigma : predicted aleatoric uncertainty (>0)
    """

    def __init__(
        self,
        emb_dim=256,
        organ_dim=64,
        n_organs=11,
        d_model=256,
        n_heads=8,
        n_layers=4,
        ff_mult=4,
    ):
        """
        Initialise ``CrossFusionV45``.

        Parameters
        ----------
        emb_dim : int
            Dimensionality of the input fused embedding (D).  Must match the
            number of ``z_rna_*`` / ``z_xray_*`` / ``z_mri_*`` columns.
        organ_dim : int
            Dimensionality of the organ embedding lookup table.
        n_organs : int
            Total number of distinct organs (vocabulary size of the organ
            embedding).
        d_model : int
            Internal transformer width.  Both tokens are projected to this
            size before being fed to the encoder.
        n_heads : int
            Number of multi-head attention heads.
        n_layers : int
            Number of ``TransformerEncoderLayer`` blocks stacked in the
            encoder.
        ff_mult : int
            Feed-forward hidden-size multiplier relative to ``d_model``.
        """
        super().__init__()

        # Organ embedding token (learned)
        self.organ_embed = nn.Embedding(n_organs, organ_dim)

        # Project organ token to transformer dimension
        self.organ_proj = nn.Linear(organ_dim, d_model)

        # Project the fused aligned embedding to d_model
        self.input_proj = nn.Linear(emb_dim, d_model)

        # Transformer encoder (2-token sequence)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=ff_mult * d_model,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=n_layers,
            enable_nested_tensor=False,
        )

        # Gaussian regression head
        self.head_mu = nn.Linear(d_model, 1)
        self.head_sigma = nn.Linear(d_model, 1)

        # Initialize sigma head so initial σ is reasonable (~10 years)
        nn.init.zeros_(self.head_sigma.weight)
        nn.init.constant_(self.head_sigma.bias, 2.0)  # softplus(2) ~ 2.1

    def forward(self, x_emb, organ_id):
        """
        x_emb: (B,256)
        organ_id: (B,)
        """
        B = x_emb.size(0)

        # Token 1: ORGAN
        organ_vec = self.organ_embed(organ_id)  # (B,64)
        organ_tok = self.organ_proj(organ_vec)  # (B,256)

        # Token 2: EMBEDDING
        emb_tok = self.input_proj(x_emb)  # (B,256)

        # Build sequence: (B,2,256)
        x = torch.stack([organ_tok, emb_tok], dim=1)

        # Transformer
        h = self.encoder(x)  # (B,2,256)

        # Use embedding token's output for prediction (index 1)
        h_emb = h[:, 1, :]  # (B,256)

        mu = self.head_mu(h_emb).squeeze(1)  # (B,)
        sigma_raw = self.head_sigma(h_emb).squeeze(1)

        # Softplus to enforce positive sigma
        sigma = F.softplus(sigma_raw) + 1e-4  # ensure >0

        return mu, sigma


def gaussian_nll(mu, sigma, target):
    """
    Heteroscedastic Gaussian negative log-likelihood with safety clamps.
    """
    # Clean any NaNs/infs
    mu = torch.nan_to_num(mu, nan=0.0, posinf=0.0, neginf=0.0)
    target = torch.nan_to_num(target, nan=0.0, posinf=0.0, neginf=0.0)
    sigma = torch.nan_to_num(sigma, nan=1.0, posinf=10.0, neginf=1.0)

    # Clamp variance to safe range
    var = torch.clamp(sigma ** 2, min=1e-3, max=1e3)
    return 0.5 * (torch.log(var) + (target - mu) ** 2 / var).mean()
