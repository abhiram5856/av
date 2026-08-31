import torch
import torch.nn as nn

class EnvironmentalRiskEngine(nn.Module):
    """
    Module 2: Environmental Risk Engine (ERE)
    -----------------------------------------
    Estimates environmental conduciveness, NOT disease diagnosis.
    Strict boundary: Never inspects images or their features.
    
    Accepts environmental (weather) and crop history inputs and maps them to
    pathogen-family risk profiles (Fungal, Bacterial, Viral, Pest, Abiotic) and
    a latent environmental representation.
    """
    def __init__(self, input_dim: int = 7, latent_dim: int = 16, num_risks: int = 5):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.num_risks = num_risks
        
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, latent_dim),
            nn.ReLU()
        )
        
        # Outputs
        self.risk_head = nn.Linear(latent_dim, num_risks) # Pathogen risk scores
        
    def forward(self, env_features: torch.Tensor, hist_features: torch.Tensor) -> dict:
        """
        Args:
            env_features: Environmental tensor of shape (batch_size, 4)
            hist_features: Crop history tensor of shape (batch_size, 3)
        Returns:
            A dictionary containing:
                - risk_scores: (batch_size, num_risks) risk probabilities
                - env_features: (batch_size, latent_dim) latent embedding
        """
        # Concatenate environmental and historical features
        x = torch.cat([env_features, hist_features], dim=-1)
        features = self.encoder(x)
        
        # Multi-label probability output for risks
        risks = torch.sigmoid(self.risk_head(features))
        
        return {
            "risk_scores": risks,
            "env_features": features
        }
