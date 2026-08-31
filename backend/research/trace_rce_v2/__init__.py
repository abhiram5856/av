"""
TRACE-RCE v2 — Trainable Multimodal Root Cause Engine
======================================================
Research Module for NOVA v2.0

This package implements a trainable PyTorch neural network for root-cause
reasoning in plant disease diagnosis. It is architecturally independent from
TRACE-RCE v1 (rule-based) and does NOT modify any production code.

Architecture: Cross-Attention Multimodal Fusion + ListMLE Ranking
Training: AdamW + CosineAnnealingLR + Early Stopping
Explainability: Attention visualization + Integrated Gradients

Author: NOVA Research Team
Version: 2.0.0
"""

__version__ = "2.0.0"
__algorithm__ = "TRACE-RCE-v2.0"
