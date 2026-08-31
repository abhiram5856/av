"""
TRACE-RCE v2 — Cause-Conditioned Ranking Head
================================================
Computes a relevance score for each of the 13 causal hypotheses by
comparing the fused context representation z with learnable cause embeddings.

Mathematical Formulation
------------------------
Let z ∈ ℝ^d be the mean-pooled fused context (output of cross-attention).
Let e_j ∈ ℝ^d be the learnable embedding for cause j.

For each cause j:
    interaction = z ⊙ e_j             (element-wise product)
    s_j = MLP([z; e_j; interaction])  ∈ ℝ

where MLP: ℝ^{3d} → ℝ is [Linear(3d, d), GELU, Dropout, Linear(d, 1)].

Why the element-wise product interaction term?
The concatenation [z; e_j] captures individual features of the context and
cause, but the interaction z ⊙ e_j captures JOINT feature relationships:
"Does high humidity (in z) co-occur with the humidity-related cause embedding?"
This is equivalent to a factored bilinear model and is more expressive than
simple concatenation alone (He et al., 2017 — Neural Collaborative Filtering).

Temperature Scaling for Confidence Calibration
----------------------------------------------
After computing raw scores s_j, we apply temperature scaling:
    p_j = σ(s_j / T)

where T is a learnable scalar parameter initialized to 1.0. Temperature
scaling is the simplest and most effective post-hoc calibration method
(Guo et al., 2017 — On Calibration of Modern Neural Networks).

Alternative considered: Platt scaling (a × s + b). Rejected because:
- Temperature scaling is a special case of Platt scaling with b=0
- It preserves ranking order (monotone transformation)
- It has only 1 learnable parameter vs. 2, reducing overfitting risk
  on our small dataset

Cause Embedding Initialization
--------------------------------
Cause embeddings are initialized with a small normal distribution N(0, 0.02),
following the convention of BERT and GPT models for token embeddings.
We do NOT use one-hot encodings because:
- The ontology has semantic structure (FUNGAL vs BACTERIAL causes are related)
- Learnable embeddings can capture these relationships during training
- Dimensionality 32 is sufficient to encode semantic differences
"""

from __future__ import annotations
from typing import Tuple

import torch
import torch.nn as nn


class CauseRankingHead(nn.Module):
    """
    Cause-conditioned MLP scoring head with learnable cause embeddings
    and temperature-scaled confidence calibration.

    Args:
        d_model: Context representation dimension (from cross-attention output)
        n_causes: Number of causal hypotheses (13 in v1 ontology)
        dropout: Dropout rate
    """

    def __init__(self, d_model: int = 32, n_causes: int = 13, dropout: float = 0.1):
        super().__init__()

        self.n_causes = n_causes
        self.d_model = d_model

        # Learnable cause embeddings: shape (n_causes, d_model)
        self.cause_embeddings = nn.Embedding(n_causes, d_model)
        nn.init.normal_(self.cause_embeddings.weight, mean=0.0, std=0.02)

        # Interaction MLP: [z; e_j; z⊙e_j] ∈ ℝ^{3d} → score ∈ ℝ
        self.scoring_mlp = nn.Sequential(
            nn.Linear(3 * d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )

        # Temperature scaling parameter (learnable scalar)
        # Initialize to 1.0 (identity calibration at start)
        self.log_temperature = nn.Parameter(torch.zeros(1))

    @property
    def temperature(self) -> torch.Tensor:
        """Temperature T = exp(log_T), constrained to be positive."""
        return torch.exp(self.log_temperature).clamp(min=0.1, max=10.0)

    def forward(
        self,
        z: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute scores and calibrated confidences for all 13 causes.

        Args:
            z: Fused context representation, shape (batch, d_model)

        Returns:
            scores: Raw logit scores, shape (batch, n_causes)
                    Used directly in ListMLE loss (no sigmoid needed for ranking)
            confidences: Calibrated probabilities, shape (batch, n_causes)
                        Values in (0, 1), used in Brier loss and output
        """
        batch_size = z.size(0)

        # Get all cause embeddings: (n_causes, d_model)
        cause_idx = torch.arange(self.n_causes, device=z.device)
        E = self.cause_embeddings(cause_idx)  # (n_causes, d_model)

        # Expand z to match: (batch, n_causes, d_model)
        z_expanded = z.unsqueeze(1).expand(-1, self.n_causes, -1)
        E_expanded = E.unsqueeze(0).expand(batch_size, -1, -1)

        # Interaction term: element-wise product
        interaction = z_expanded * E_expanded  # (batch, n_causes, d_model)

        # Concatenate and score
        combined = torch.cat([z_expanded, E_expanded, interaction], dim=-1)  # (batch, n_causes, 3d)
        scores = self.scoring_mlp(combined).squeeze(-1)  # (batch, n_causes)

        # Temperature-scaled confidence calibration
        confidences = torch.sigmoid(scores / self.temperature)  # (batch, n_causes)

        return scores, confidences
