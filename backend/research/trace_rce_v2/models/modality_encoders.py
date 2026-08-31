"""
TRACE-RCE v2 — Modality Encoders
===================================
Projects each modality's raw feature vector into a shared d-dimensional
embedding space using a two-layer MLP with LayerNorm.

Architecture Rationale
----------------------
Each modality has a different raw feature dimension (3–4) and a different
semantic meaning (visual activations vs. weather measurements vs. history).
Before cross-attention can operate meaningfully, all modalities must live in
the same representation space. We use a simple but effective:

  Linear(dim_in, d) → GELU → LayerNorm → Dropout → Linear(d, d) → LayerNorm

Why GELU over ReLU?
GELU has smoother gradients near zero and empirically outperforms ReLU for
small transformer-like architectures (Hendrycks & Gimpel, 2016).
ReLU creates dead neurons at zero; GELU's gradient is non-zero everywhere.

Why LayerNorm after the projection?
LayerNorm ensures all modality embeddings are on the same scale before
cross-attention computes Q·K^T. Without normalization, a modality with
larger absolute activations would dominate the attention scores.

Why NOT BatchNorm?
BatchNorm behaves differently at train vs. eval and is sensitive to small
batch sizes. With batch_size=32 and 5 modalities, LayerNorm is safer.

Parameter Count per Encoder
-----------------------------
  dim_in=4, d=32: 4×32 + 32×32 = 128 + 1024 = 1,152 params
  For 5 encoders: ~5,760 params (visual, env, sev, know, hist)
  The historical encoder has dim_in=3: slightly fewer params.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class ModalityEncoder(nn.Module):
    """
    Two-layer MLP encoder projecting a single modality into ℝ^d.

    Forward pass: x ∈ ℝ^{dim_in} → h ∈ ℝ^d

    Args:
        dim_in: Input feature dimension for this modality
        d_model: Target embedding dimension (shared across all modalities)
        dropout: Dropout rate
    """

    def __init__(self, dim_in: int, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim_in, d_model),
            nn.GELU(),
            nn.LayerNorm(d_model),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input features, shape (batch, dim_in)
        Returns:
            Embedding, shape (batch, d_model)
        """
        return self.net(x)


class ModalityEncoderBank(nn.Module):
    """
    Container for all five per-modality encoders.

    Accepts a batch dict and returns a stacked modality sequence
    tensor suitable for cross-attention.

    Output: H ∈ ℝ^{batch × 5 × d_model}
    (5 modalities, each embedded into d_model dimensions)
    """

    MODALITY_DIMS = {
        "visual":    4,
        "env":       4,
        "severity":  4,
        "knowledge": 4,
        "historical": 3,
    }

    MODALITY_ORDER = ["visual", "env", "severity", "knowledge", "historical"]

    def __init__(self, d_model: int = 32, dropout: float = 0.1):
        super().__init__()

        self.encoders = nn.ModuleDict({
            name: ModalityEncoder(dim, d_model, dropout)
            for name, dim in self.MODALITY_DIMS.items()
        })

        self.d_model = d_model

    def forward(self, batch: dict) -> torch.Tensor:
        """
        Encode all modalities and stack into a sequence.

        Args:
            batch: Dict with keys 'visual', 'env', 'severity', 'knowledge', 'historical'
                   Each value has shape (batch_size, dim_m)

        Returns:
            H: shape (batch_size, 5, d_model) — the modality sequence
        """
        embeddings = []
        for name in self.MODALITY_ORDER:
            h_m = self.encoders[name](batch[name])  # (batch, d_model)
            embeddings.append(h_m.unsqueeze(1))     # (batch, 1, d_model)

        # Stack: (batch, 5, d_model)
        return torch.cat(embeddings, dim=1)

    @property
    def n_modalities(self) -> int:
        return len(self.MODALITY_ORDER)
