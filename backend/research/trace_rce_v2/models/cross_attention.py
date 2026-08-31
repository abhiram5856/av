"""
TRACE-RCE v2 — Cross-Attention Fusion Layer
===============================================
Applies multi-head self-attention over the sequence of 5 modality embeddings,
allowing each modality to attend to all other modalities.

Mathematical Operation
----------------------
Given the modality sequence H ∈ ℝ^{B×5×d}, for each head h:

    Q_h = H · W_Q_h    (B, 5, d_k)
    K_h = H · W_K_h    (B, 5, d_k)
    V_h = H · W_V_h    (B, 5, d_v)

    A_h = softmax(Q_h · K_h^T / √d_k)    (B, 5, 5) — attention weights
    O_h = A_h · V_h                        (B, 5, d_v)

    O = concat(O_1, ..., O_n_heads) · W_O   (B, 5, d)

    H' = LayerNorm(H + Dropout(O))           (B, 5, d)  — residual connection

The attention matrix A ∈ ℝ^{B×5×5} is the KEY EXPLAINABILITY ARTIFACT.
Entry A[b, i, j] represents: "When reasoning about the current context,
how much did modality i attend to modality j?"

For example, if visual consistently attends strongly to env, the model has
learned that GradCAM coverage is correlated with weather conditions —
a cross-modal relationship that v1's linear fusion cannot capture.

Design Choices
--------------
- We use PyTorch's built-in nn.MultiheadAttention for correctness and efficiency.
- We set batch_first=True so H is (batch, seq, d_model).
- We return attention weights for explainability (average_attn_weights=False
  returns per-head weights; we average over heads for visualization).
- The FFN (Feed-Forward Network) after attention uses d_ffn = 4 × d_model,
  following the canonical Transformer design ratio.

Why Self-Attention (not cross-attention between query and key from different sets)?
Our formulation is technically "self-attention" over modalities, which allows
every modality to become both a query (seeking information from others) and a
key/value (providing information to others). This is exactly the right inductive
bias: we don't know a priori which modality should be the "query" — we let the
model learn this.
"""

from __future__ import annotations
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossModalAttention(nn.Module):
    """
    Multi-head self-attention over the 5-modality sequence + position-wise FFN.

    Implements one Transformer encoder layer, applied to the modality sequence
    rather than a word sequence. Includes:
      - Multi-head self-attention (with residual + LayerNorm)
      - Position-wise FFN (with residual + LayerNorm)
      - Attention weight extraction for explainability

    Args:
        d_model: Embedding dimension (same as encoder output)
        n_heads: Number of attention heads (d_model must be divisible by n_heads)
        dropout: Dropout rate
        d_ffn: Dimension of FFN hidden layer (default: 4 × d_model)
    """

    def __init__(
        self,
        d_model: int = 32,
        n_heads: int = 2,
        dropout: float = 0.1,
        d_ffn: int = None,
    ):
        super().__init__()
        assert d_model % n_heads == 0, f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"

        d_ffn = d_ffn or (4 * d_model)

        # Multi-head self-attention
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)

        # Position-wise Feed-Forward Network
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ffn),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ffn, d_model),
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)

    def forward(
        self,
        H: torch.Tensor,
        return_attention: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            H: Modality sequence, shape (batch, 5, d_model)
            return_attention: Whether to return attention weights

        Returns:
            H': Fused modality sequence, shape (batch, 5, d_model)
            attn_weights: Attention matrix, shape (batch, 5, 5)
                         (averaged across heads for interpretability)
        """
        # Self-attention: each modality queries all others
        # nn.MultiheadAttention with need_weights=True returns (output, attn_weights)
        attn_output, attn_weights = self.attn(
            query=H, key=H, value=H,
            need_weights=True,
            average_attn_weights=True,  # Average over heads → (batch, 5, 5)
        )

        # Residual + LayerNorm (Pre-LN variant is more stable for small models)
        H = self.norm1(H + self.dropout1(attn_output))

        # Position-wise FFN
        ffn_output = self.ffn(H)
        H = self.norm2(H + self.dropout2(ffn_output))

        return H, attn_weights
