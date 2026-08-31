"""
TRACE-RCE v2 — Unit Tests Suite
==================================
Verifies the correctness of the trainable TRACE-RCE v2 pipeline:
1. Feature Extractor: Ensures AIContext objects map correctly to standard dimensions.
2. Dataset & DataLoader: Assesses batch collating, splits, and augmentation behavior.
3. Model Architecture: Checks dimension matching, parameters, and deterministic forward pass.
4. Loss functions: Verifies ListMLELoss, BrierLoss, and gradient flow.
5. Inference Engine: Tests analyze() interface compatibility and schema matching.
"""

import os
import torch
import pytest
import numpy as np

from backend.research.root_cause_engine.dataset import generate_evaluation_dataset
from backend.research.trace_rce_v2.dataset.dataloader import create_dataloaders
from backend.research.trace_rce_v2.dataset.feature_extractor import extract_features
from backend.research.trace_rce_v2.models.trace_rce_v2 import build_model
from backend.research.trace_rce_v2.training.losses import CombinedLoss
from backend.research.trace_rce_v2.inference.engine import TRACERootCauseEngineV2
from backend.research.trace_rce_v2.dataset.dataset import N_CAUSES

# =============================================================================
# 1. Feature Extractor Tests
# =============================================================================

def test_feature_extractor():
    """Verify that feature extractor constructs tensors of the correct shapes and values."""
    scenarios = generate_evaluation_dataset()
    sample_ctx = scenarios[0].context
    
    features = extract_features(sample_ctx)
    
    assert features.visual.shape == (4,)
    assert features.env.shape == (4,)
    assert features.severity.shape == (4,)
    assert features.knowledge.shape == (4,)
    assert features.historical.shape == (3,)
    
    # Assert normalized bounds [0, 1]
    for tensor in [features.visual, features.env, features.severity, features.knowledge, features.historical]:
        assert torch.all(tensor >= 0.0)
        assert torch.all(tensor <= 1.0)


# =============================================================================
# 2. Dataset & Dataloader Tests
# =============================================================================

def test_dataloaders():
    """Verify train, validation, and test dataloaders shapes and split stratification."""
    train_loader, val_loader, test_loader = create_dataloaders(
        augmentation_factor=2,  # 2 augmented copies per sample
        batch_size=16,
        train_ratio=0.7,
        val_ratio=0.15,
        seed=42
    )
    
    # Check loaders sizes (300 base scenarios divided ~ 70/15/15)
    train_size = len(train_loader.dataset)
    val_size = len(val_loader.dataset)
    test_size = len(test_loader.dataset)
    
    assert train_size % (1 + 2) == 0  # must be divisible by (1 + augmentation_factor)
    assert abs(train_size // 3 - 210) <= 5  # approximately 70% of 300
    assert abs(val_size - 45) <= 5         # approximately 15% of 300
    assert abs(test_size - 45) <= 5        # approximately 15% of 300

    # Check collating
    batch = next(iter(train_loader))
    assert batch["visual"].shape == (16, 4)
    assert batch["env"].shape == (16, 4)
    assert batch["severity"].shape == (16, 4)
    assert batch["knowledge"].shape == (16, 4)
    assert batch["historical"].shape == (16, 3)
    assert batch["cause_ranking"].shape == (16, N_CAUSES)
    assert batch["binary_labels"].shape == (16, N_CAUSES)


# =============================================================================
# 3. Model Architecture Tests
# =============================================================================

def test_model_forward():
    """Test standard model forward pass shapes and attention extraction."""
    model = build_model()
    
    # Simulate a batch of 8 samples
    batch = {
        "visual": torch.randn(8, 4),
        "env": torch.randn(8, 4),
        "severity": torch.randn(8, 4),
        "knowledge": torch.randn(8, 4),
        "historical": torch.randn(8, 3)
    }
    
    outputs = model(batch, return_attention=True)
    
    assert outputs["scores"].shape == (8, N_CAUSES)
    assert outputs["confidences"].shape == (8, N_CAUSES)
    assert outputs["attn_weights"].shape == (8, 5, 5)
    
    # Assert calibrated probabilities are bounded
    assert torch.all(outputs["confidences"] >= 0.0)
    assert torch.all(outputs["confidences"] <= 1.0)


def test_model_determinism():
    """Test that model predictions are deterministic given the same inputs."""
    torch.manual_seed(42)
    model = build_model()
    model.eval()  # Crucial: Put in eval mode to disable dropout, ensuring determinism
    
    batch = {
        "visual": torch.randn(4, 4),
        "env": torch.randn(4, 4),
        "severity": torch.randn(4, 4),
        "knowledge": torch.randn(4, 4),
        "historical": torch.randn(4, 3)
    }
    
    out1 = model(batch, return_attention=False)
    out2 = model(batch, return_attention=False)
    
    assert torch.allclose(out1["scores"], out2["scores"])
    assert torch.allclose(out1["confidences"], out2["confidences"])


# =============================================================================
# 4. Loss Function Tests
# =============================================================================

def test_losses_gradient_flow():
    """Verify CombinedLoss calculates gradients correctly and updates weights."""
    model = build_model()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    loss_fn = CombinedLoss()

    batch = {
        "visual": torch.randn(2, 4),
        "env": torch.randn(2, 4),
        "severity": torch.randn(2, 4),
        "knowledge": torch.randn(2, 4),
        "historical": torch.randn(2, 3)
    }
    ranks = torch.tensor([[1, 2] + [3]*11, [2, 1] + [3]*11], dtype=torch.long)
    labels = torch.tensor([[1.0, 1.0] + [0.0]*11, [1.0, 1.0] + [0.0]*11], dtype=torch.float32)

    # Initial forward/backward pass
    model.train()
    outputs = model(batch)
    loss, l_rank, l_cal = loss_fn(outputs["scores"], outputs["confidences"], ranks, labels)
    
    assert loss.item() > 0.0
    loss.backward()
    
    # Verify gradients computed for all weights
    for name, param in model.named_parameters():
        assert param.grad is not None, f"Gradient for {name} was not computed!"
        
    optimizer.step()


# =============================================================================
# 5. Inference Engine Tests
# =============================================================================

def test_inference_engine():
    """Verify that the AbstractRootCauseEngine interface wrapper runs and matches schema."""
    import asyncio
    
    scenarios = generate_evaluation_dataset()
    sample_ctx = scenarios[0].context
    
    engine = TRACERootCauseEngineV2()
    result = asyncio.run(engine.analyze(sample_ctx))
    
    # Assert type
    from backend.research.root_cause_engine.models import RootCauseResult
    assert isinstance(result, RootCauseResult)
    
    # Assert structural content
    assert result.disease_name == sample_ctx.vision.predicted_disease
    assert len(result.ranked_causes) > 0
    assert result.ranked_causes[0].rank == 1
    assert result.ranked_causes[0].cas_raw is not None
    assert len(result.reasoning_chain) > 0
    assert len(result.evidence_graph.nodes) > 0
