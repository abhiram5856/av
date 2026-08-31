import torch
import torch.nn as nn
from backend.research.trace_rce_v3.models.vee import VisualEvidenceEngine
from backend.research.trace_rce_v3.models.ere import EnvironmentalRiskEngine
from backend.research.trace_rce_v3.models.ke import KnowledgeEngine
from backend.research.trace_rce_v3.models.ece import EvidenceConsistencyEngine

class TRACERCEv3(nn.Module):
    """
    TRACE-RCE v3: Multimodal Causal Reasoning Engine
    ------------------------------------------------
    Implements the first-principles modular reasoning architecture.
    Couples VEE, ERE, KE, and ECE with a learnable confidence calibration layer
    and final cause ranking head.
    """
    def __init__(self, latent_dim: int = 16, num_classes: int = 36, num_risks: int = 5, num_causes: int = 13):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_classes = num_classes
        self.num_risks = num_risks
        self.num_causes = num_causes
        
        # Modules
        self.vee = VisualEvidenceEngine(input_dim=4, latent_dim=latent_dim, num_classes=num_classes)
        self.ere = EnvironmentalRiskEngine(input_dim=7, latent_dim=latent_dim, num_risks=num_risks)
        self.ke = KnowledgeEngine(input_dim=4, latent_dim=latent_dim)
        self.ece = EvidenceConsistencyEngine(latent_dim=latent_dim, num_states=4)
        
        # Confidence Calibration Network
        self.calibration_net = nn.Sequential(
            nn.Linear(3, 8),
            nn.ReLU(),
            nn.Linear(8, 1)
        )
        
        # Cause Ranking Head: consumes visual, env, and consistency states
        # Concat size: latent_dim * 2 (vee + ere) + 32 (ece shared representation proxy or outputs)
        # To keep it simple: concat size = latent_dim * 2 (visual + env) + 4 (reasoning state probs)
        ranking_input_dim = latent_dim * 2 + 4
        self.ranking_head = nn.Sequential(
            nn.Linear(ranking_input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, num_causes)
        )
        
    def forward(self, batch_inputs: dict) -> dict:
        """
        Args:
            batch_inputs: Dict containing tensors for:
                - visual: (batch_size, 4)
                - env: (batch_size, 4)
                - historical: (batch_size, 3)
                - knowledge: (batch_size, 4)
        Returns:
            Dictionary containing:
                - disease_present_prob: (batch_size,) presence probability (from VEE)
                - disease_class_probs: (batch_size, num_classes) class probs (from VEE)
                - severity_score: (batch_size,) severity index (from VEE)
                - visual_confidence: (batch_size,) visual confidence (from VEE)
                - risk_scores: (batch_size, num_risks) env risk estimates (from ERE)
                - agreement_score: (batch_size,) (from ECE)
                - conflict_score: (batch_size,) (from ECE)
                - reasoning_state: (batch_size, 4) (from ECE)
                - calibrated_confidence: (batch_size,) final calibrated confidence
                - scores: (batch_size, num_causes) raw ranking scores
                - confidences: (batch_size, num_causes) calibrated cause probabilities
        """
        # 1. Observations
        vee_out = self.vee(batch_inputs["visual"])
        ere_out = self.ere(batch_inputs["env"], batch_inputs["historical"])
        ke_out = self.ke(batch_inputs["knowledge"])
        
        # 2. Consistency Analysis
        ece_out = self.ece(
            vee_out["visual_features"],
            ere_out["env_features"],
            ke_out["knowledge_features"]
        )
        
        # 3. Confidence Calibration
        # Input features: [visual_confidence, agreement_score, conflict_score]
        cal_input = torch.stack([
            vee_out["visual_confidence"],
            ece_out["agreement_score"],
            ece_out["conflict_score"]
        ], dim=-1)
        
        # Learnable scaling
        calibrated_conf = torch.sigmoid(self.calibration_net(cal_input)).squeeze(-1)
        
        # 4. Cause Ranking
        ranking_feats = torch.cat([
            vee_out["visual_features"],
            ere_out["env_features"],
            ece_out["reasoning_state"]
        ], dim=-1)
        
        raw_scores = self.ranking_head(ranking_feats)
        
        # Output calibrated cause confidences by scaling raw scores with final calibrated confidence
        # Cause confidences represent probabilities of each cause being a root cause
        cause_confidences = torch.sigmoid(raw_scores) * calibrated_conf.unsqueeze(-1)
        
        # Assemble complete outputs
        outputs = {
            "disease_present_prob": vee_out["disease_present_prob"],
            "disease_class_probs": vee_out["disease_class_probs"],
            "severity_score": vee_out["severity_score"],
            "visual_confidence": vee_out["visual_confidence"],
            "risk_scores": ere_out["risk_scores"],
            "agreement_score": ece_out["agreement_score"],
            "conflict_score": ece_out["conflict_score"],
            "reasoning_state": ece_out["reasoning_state"],
            "state_logits": ece_out["state_logits"],
            "calibrated_confidence": calibrated_conf,
            "scores": raw_scores,
            "confidences": cause_confidences
        }
        
        return outputs
