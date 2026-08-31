"""
TRACE-RCE v2 — Top-Level Model
=================================
Assembles all components into a single nn.Module with a clean forward pass
and a production-compatible predict() method.

Full Architecture Summary
-------------------------
Input: Batch dict with 5 modality tensors

  [visual (4)] [env (4)] [severity (4)] [knowledge (4)] [historical (3)]
       ↓             ↓          ↓              ↓                ↓
  ModalityEncoder × 5   (each → ℝ^d,  d=32)
       ↓
  Stack → H ∈ ℝ^{B×5×d}
       ↓
  CrossModalAttention    (H → H', A)
  H' ∈ ℝ^{B×5×d},  A ∈ ℝ^{B×5×5}
       ↓
  Mean pool across modalities → z ∈ ℝ^{B×d}
       ↓
  CauseRankingHead(z, cause_embeddings)
       ↓
  scores ∈ ℝ^{B×13}   (for ListMLE)
  confidences ∈ ℝ^{B×13}  (for Brier)

Output:
  {
    "scores":      (batch, 13)   — raw logits for ranking
    "confidences": (batch, 13)   — calibrated probabilities
    "attn_weights": (batch, 5, 5) — modality attention matrix
    "z":           (batch, d)    — fused context (for IG explainability)
  }

Total Parameter Count (d=32, n_heads=2, n_causes=13):
  Modality Encoders:  4×(4×32 + 32×32) + 1×(3×32 + 32×32) ≈ 5,760 params
  Cross-Attention:    MHA(32, 2) + FFN(32, 128) ≈ 12,800 params
  Ranking Head:       Cause embed(13×32) + MLP(96→32→1) ≈ 3,700 params
  Temperature:        1 param
  ─────────────────────────────────────────────────────────────────────────
  TOTAL:              ≈ 22,261 params   (extremely compact — no overfitting risk)
"""

from __future__ import annotations
from typing import Dict, Tuple, Optional, List

import torch
import torch.nn as nn

from backend.research.trace_rce_v2.models.modality_encoders import ModalityEncoderBank
from backend.research.trace_rce_v2.models.cross_attention import CrossModalAttention
from backend.research.trace_rce_v2.models.ranking_head import CauseRankingHead
from backend.research.trace_rce_v2.dataset.dataset import CAUSE_IDS, N_CAUSES


class TRACERCEv2(nn.Module):
    """
    TRACE-RCE v2: Cross-Attention Multimodal Fusion + ListMLE Ranking.

    This is the primary nn.Module for the research implementation.
    It is designed to be:
      - Fully trainable: all parameters receive gradients
      - Explainable: returns attention weights for visualization
      - Compatible: predict() returns RootCause-compatible dicts
      - Reproducible: deterministic given a seed (no stochastic inference)

    Args:
        d_model: Shared embedding dimension (default: 32)
        n_heads: Number of attention heads (default: 2)
        n_causes: Number of causal hypotheses (default: 13)
        dropout: Dropout rate during training (default: 0.1)
    """

    def __init__(
        self,
        d_model: int = 32,
        n_heads: int = 2,
        n_causes: int = N_CAUSES,
        dropout: float = 0.1,
    ):
        super().__init__()

        # Component 1: Modality Encoders
        self.encoder_bank = ModalityEncoderBank(d_model=d_model, dropout=dropout)

        # Component 2: Cross-Modal Attention
        self.cross_attention = CrossModalAttention(
            d_model=d_model,
            n_heads=n_heads,
            dropout=dropout,
        )

        # Component 3: Cause-Conditioned Ranking Head
        self.ranking_head = CauseRankingHead(
            d_model=d_model,
            n_causes=n_causes,
            dropout=dropout,
        )

        self.d_model = d_model
        self.n_causes = n_causes

        # Initialize weights (Xavier for Linear, normal for embeddings)
        self._init_weights()

    def _init_weights(self) -> None:
        """Apply Xavier uniform initialization to all Linear layers."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(
        self,
        batch: Dict[str, torch.Tensor],
        return_attention: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """
        Full forward pass: features → scores + confidences + attention.

        Args:
            batch: Dict with modality tensors (visual, env, severity, knowledge, historical)
            return_attention: If True, include attention weights in output

        Returns:
            dict with keys:
              'scores':       (batch, n_causes) — raw logits for ListMLE
              'confidences':  (batch, n_causes) — calibrated probabilities
              'attn_weights': (batch, 5, 5) — cross-modal attention matrix
              'z':            (batch, d_model) — fused context embedding
        """
        # Step 1: Encode all modalities → H ∈ ℝ^{B×5×d}
        H = self.encoder_bank(batch)

        # Step 2: Cross-modal attention → H' ∈ ℝ^{B×5×d}, A ∈ ℝ^{B×5×5}
        H_prime, attn_weights = self.cross_attention(H, return_attention=True)

        # Step 3: Mean pool across modality dimension → z ∈ ℝ^{B×d}
        z = H_prime.mean(dim=1)

        # Step 4: Score all causes
        scores, confidences = self.ranking_head(z)

        output = {
            "scores": scores,
            "confidences": confidences,
            "z": z,
        }
        if return_attention:
            output["attn_weights"] = attn_weights

        return output

    def predict(
        self,
        batch: Dict[str, torch.Tensor],
        top_k: int = 5,
    ) -> List[Dict]:
        """
        Production-compatible inference method.

        Given a single sample or a batch, returns a ranked list of root cause
        predictions in a format compatible with the RootCauseResult schema.

        Args:
            batch: Preprocessed feature dict (from feature_extractor.py)
            top_k: Number of top causes to return

        Returns:
            List of dicts (one per sample in batch), each containing:
            [
                {
                    "rank": int,
                    "cause_id": str,
                    "cause_label": str (from ontology),
                    "score": float,
                    "confidence": float,
                }
            ]
        """
        self.eval()
        with torch.no_grad():
            output = self.forward(batch, return_attention=False)

        scores      = output["scores"]      # (batch, n_causes)
        confidences = output["confidences"] # (batch, n_causes)

        results = []
        for b in range(scores.size(0)):
            sorted_indices = torch.argsort(scores[b], descending=True)
            top_causes = []
            for rank, idx in enumerate(sorted_indices[:top_k], start=1):
                cause_id = CAUSE_IDS[idx.item()]
                top_causes.append({
                    "rank": rank,
                    "cause_id": cause_id,
                    "score": scores[b, idx].item(),
                    "confidence": confidences[b, idx].item(),
                })
            results.append(top_causes)

        return results

    def count_parameters(self) -> int:
        """Returns the total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def parameter_summary(self) -> str:
        """Returns a human-readable parameter count breakdown."""
        enc_params  = sum(p.numel() for p in self.encoder_bank.parameters() if p.requires_grad)
        attn_params = sum(p.numel() for p in self.cross_attention.parameters() if p.requires_grad)
        rank_params = sum(p.numel() for p in self.ranking_head.parameters() if p.requires_grad)
        total = enc_params + attn_params + rank_params

        return (
            f"Parameter Summary:\n"
            f"  Modality Encoders:     {enc_params:,}\n"
            f"  Cross-Attention:       {attn_params:,}\n"
            f"  Cause Ranking Head:    {rank_params:,}\n"
            f"  ─────────────────────────────────\n"
            f"  TOTAL:                 {total:,}"
        )


def build_model(config: dict = None) -> TRACERCEv2:
    """
    Factory function to construct TRACERCEv2 from a config dict.

    Args:
        config: Model configuration dict (from default.yaml['model'])
                If None, uses default values.

    Returns:
        Initialized TRACERCEv2 model
    """
    if config is None:
        config = {}

    model = TRACERCEv2(
        d_model=config.get("d_model", 32),
        n_heads=config.get("n_heads", 2),
        n_causes=config.get("n_causes", N_CAUSES),
        dropout=config.get("dropout", 0.1),
    )
    return model
