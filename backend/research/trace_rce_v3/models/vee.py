import torch
import torch.nn as nn

class VisualEvidenceEngine(nn.Module):
    """
    Module 1: Visual Evidence Engine (VEE)
    --------------------------------------
    Observes only the leaf.
    Strict boundary: Absolutely no environmental, weather, or crop history inputs.
    
    Accepts visual representations (pre-extracted features or intermediate CNN embeddings)
    and maps them to disease presence, class probabilities, severity, confidence, and a
    latent visual feature vector.
    """
    def __init__(self, input_dim: int = 4, latent_dim: int = 16, num_classes: int = 36):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.num_classes = num_classes
        
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, latent_dim),
            nn.ReLU()
        )
        
        # Outputs
        self.present_head = nn.Linear(latent_dim, 1)      # Disease Present Probability
        self.class_head = nn.Linear(latent_dim, num_classes) # Disease Class Probabilities
        self.severity_head = nn.Linear(latent_dim, 1)     # Severity Score
        self.confidence_head = nn.Linear(latent_dim, 1)   # Visual Confidence Score
        
    def forward(self, x: torch.Tensor) -> dict:
        """
        Args:
            x: Input visual tensor of shape (batch_size, input_dim)
        Returns:
            A dictionary containing:
                - disease_present_prob: (batch_size,) presence probability
                - disease_class_probs: (batch_size, num_classes) class distribution
                - severity_score: (batch_size,) severity
                - visual_confidence: (batch_size,) visual confidence
                - visual_features: (batch_size, latent_dim) latent embedding
        """
        features = self.encoder(x)
        
        present_prob = torch.sigmoid(self.present_head(features)).squeeze(-1)
        class_logits = self.class_head(features)
        class_probs = torch.softmax(class_logits, dim=-1)
        severity = torch.sigmoid(self.severity_head(features)).squeeze(-1)
        confidence = torch.sigmoid(self.confidence_head(features)).squeeze(-1)
        
        return {
            "disease_present_prob": present_prob,
            "disease_class_probs": class_probs,
            "severity_score": severity,
            "visual_confidence": confidence,
            "visual_features": features
        }
