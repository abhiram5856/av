import torch
import torch.nn as nn

class EvidenceConsistencyEngine(nn.Module):
    """
    Module 4: Evidence Consistency Engine (ECE)
    -------------------------------------------
    Acts as the central neuro-symbolic coordinator.
    Compares observations from VEE, ERE, and KE.
    Outputs agreement and conflict scores, and a reasoning state.
    """
    def __init__(self, latent_dim: int = 16, num_states: int = 4):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_states = num_states
        
        # Interactive representation dimension: visual + env + knowledge + absolute differences
        # Concatenation size = latent_dim * 3
        # Differences size = latent_dim (abs(visual - env))
        input_dim = latent_dim * 3 + latent_dim
        
        self.shared_layer = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU()
        )
        
        # Heads
        self.agreement_head = nn.Linear(32, 1)   # Agreement score [0, 1]
        self.conflict_head = nn.Linear(32, 1)    # Conflict score [0, 1]
        self.state_head = nn.Linear(32, num_states) # Softmax reasoning state logits
        
    def forward(self, visual_feats: torch.Tensor, env_feats: torch.Tensor, knowledge_feats: torch.Tensor) -> dict:
        """
        Args:
            visual_feats: Latent visual vector of shape (batch_size, latent_dim)
            env_feats: Latent environmental vector of shape (batch_size, latent_dim)
            knowledge_feats: Latent knowledge vector of shape (batch_size, latent_dim)
        Returns:
            A dictionary containing:
                - agreement_score: (batch_size,) similarity/agreement score
                - conflict_score: (batch_size,) conflict degree
                - reasoning_state: (batch_size, num_states) probability over reasoning modes
        """
        # Compute interactive features: concatenation + absolute visual-env discrepancy
        discrepancy = torch.abs(visual_feats - env_feats)
        x = torch.cat([visual_feats, env_feats, knowledge_feats, discrepancy], dim=-1)
        
        shared = self.shared_layer(x)
        
        agreement = torch.sigmoid(self.agreement_head(shared)).squeeze(-1)
        conflict = torch.sigmoid(self.conflict_head(shared)).squeeze(-1)
        state_logits = self.state_head(shared)
        state_probs = torch.softmax(state_logits, dim=-1)
        
        return {
            "agreement_score": agreement,
            "conflict_score": conflict,
            "reasoning_state": state_probs,
            "state_logits": state_logits
        }
