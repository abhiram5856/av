import torch
import torch.nn as nn

class KnowledgeEngine(nn.Module):
    """
    Module 3: Knowledge Engine (KE)
    -------------------------------
    Embeds preprocessed RAG metadata into a joint knowledge vector space.
    """
    def __init__(self, input_dim: int = 4, latent_dim: int = 16):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, latent_dim),
            nn.ReLU()
        )
        
    def forward(self, knowledge_features: torch.Tensor) -> dict:
        """
        Args:
            knowledge_features: Knowledge RAG features of shape (batch_size, 4)
        Returns:
            A dictionary containing:
                - knowledge_features: (batch_size, latent_dim) latent embedding
        """
        features = self.encoder(knowledge_features)
        return {
            "knowledge_features": features
        }
