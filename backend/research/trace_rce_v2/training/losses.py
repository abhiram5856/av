"""
TRACE-RCE v2 — Loss Functions
===============================
Implements the loss functions used to train TRACE-RCE v2:
1. ListMLELoss: Listwise ranking loss to optimize the cause ranking.
2. BrierLoss: Mean squared error of probabilities to calibrate confidences.
3. CombinedLoss: Weighted combination of ranking and calibration losses.

Scientific Rationale
--------------------
- ListMLE is chosen over pairwise ranking losses (like RankNet or margin ranking loss)
  because it considers the entire permutation of causes simultaneously. Pairwise
  losses optimize pairs independently, which can lead to suboptimal global rankings
  (e.g., incorrect top-1 even if pairs are mostly correct). ListMLE directly
  maximizes the probability of the ideal ranking sequence (primary > secondary > others).
- Brier Score is the proper scoring rule for probability calibration. Optimizing it
  minimizes Expected Calibration Error (ECE) and ensures predicted confidences reflect
  true probabilities.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class ListMLELoss(nn.Module):
    """
    Listwise Maximum Likelihood Estimation (ListMLE) Loss.
    Optimizes the model to output scores that rank causes in the ground-truth order.
    
    Target ranking:
      - Primary cause: rank 1 (lowest value)
      - Secondary cause: rank 2
      - Other causes: rank 3 (highest value)
      
    For a single sample:
      Let s be the scores, and y be the target ranks.
      We sort s by y in ascending order (best first). Let this be s_sorted.
      ListMLE Loss = -sum_{i=1}^{N} log( exp(s_sorted[i]) / sum_{j=i}^{N} exp(s_sorted[j]) )
                   = sum_{i=1}^{N} ( logsumexp(s_sorted[i:]) - s_sorted[i] )
    """
    def __init__(self):
        super().__init__()

    def forward(self, scores: torch.Tensor, target_ranks: torch.Tensor) -> torch.Tensor:
        """
        Args:
            scores: Predicted scores, shape (batch_size, n_causes)
            target_ranks: Ground truth ranks, shape (batch_size, n_causes)
                          where primary=1, secondary=2, others=3.
        Returns:
            loss: Mean ListMLE loss across the batch.
        """
        batch_size, n_causes = scores.shape
        
        # Sort scores according to target ranks (ascending order: rank 1, then 2, then 3)
        sorted_indices = torch.argsort(target_ranks, dim=1)
        s_sorted = torch.gather(scores, dim=1, index=sorted_indices)  # (batch_size, n_causes)
        
        # Expand to (batch_size, n_causes, n_causes) to compute suffix sums in parallel
        # s_expanded[b, i, j] = s_sorted[b, j]
        s_expanded = s_sorted.unsqueeze(1).expand(-1, n_causes, -1)
        
        # Create upper triangular mask of shape (n_causes, n_causes)
        # mask[i, j] = 1 if j >= i else 0
        mask = torch.triu(torch.ones(n_causes, n_causes, device=scores.device), diagonal=0)
        
        # Mask out elements where j < i (set to -inf so they don't affect logsumexp)
        s_masked = s_expanded.masked_fill(mask.unsqueeze(0) == 0, float('-inf'))
        
        # Compute logsumexp along the last dimension -> (batch_size, n_causes)
        # suffix_logsumexp[b, i] = logsumexp(s_sorted[b, i:])
        suffix_logsumexp = torch.logsumexp(s_masked, dim=2)
        
        # We only sum suffix terms up to n_causes - 2 (the last term has no choice, loss is 0)
        # Loss = sum_{i=0}^{n_causes-2} (suffix_logsumexp[:, i] - s_sorted[:, i])
        per_sample_loss = torch.sum(suffix_logsumexp[:, :-1] - s_sorted[:, :-1], dim=1)
        
        return torch.mean(per_sample_loss)

class BrierLoss(nn.Module):
    """
    Brier Score Loss (Mean Squared Error on probabilities).
    Used as an auxiliary loss for probability calibration.
    
    Formula:
      L_Brier = (1 / n_causes) * sum_{j=1}^{n_causes} (p_j - y_j)^2
    """
    def __init__(self):
        super().__init__()

    def forward(self, confidences: torch.Tensor, binary_labels: torch.Tensor) -> torch.Tensor:
        """
        Args:
            confidences: Calibrated probabilities, shape (batch_size, n_causes)
            binary_labels: Binary ground truth, shape (batch_size, n_causes)
        Returns:
            loss: Brier loss.
        """
        return F.mse_loss(confidences, binary_labels)

class CombinedLoss(nn.Module):
    """
    Combined Loss: ListMLE (Ranking) + Brier (Calibration).
    
    L_total = L_ListMLE + lambda_cal * L_Brier
    """
    def __init__(self, lambda_cal: float = 0.3):
        super().__init__()
        self.list_mle = ListMLELoss()
        self.brier = BrierLoss()
        self.lambda_cal = lambda_cal

    def forward(
        self,
        scores: torch.Tensor,
        confidences: torch.Tensor,
        target_ranks: torch.Tensor,
        binary_labels: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            scores: shape (batch_size, n_causes)
            confidences: shape (batch_size, n_causes)
            target_ranks: shape (batch_size, n_causes)
            binary_labels: shape (batch_size, n_causes)
        Returns:
            total_loss, ranking_loss, calibration_loss
        """
        l_rank = self.list_mle(scores, target_ranks)
        l_cal = self.brier(confidences, binary_labels)
        total = l_rank + self.lambda_cal * l_cal
        return total, l_rank, l_cal
