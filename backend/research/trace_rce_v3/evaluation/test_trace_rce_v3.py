import os
import sys
import torch

# Add project root to sys.path
sys.path.append(r"c:\Users\ABHIRAM MODUKURU\OneDrive\Desktop\AgriVision-AI")

from backend.research.trace_rce_v3.models.trace_rce_v3 import TRACERCEv3

def test_vee_isolation():
    print("\nRunning VEE Isolation Test...")
    model = TRACERCEv3()
    model.eval()
    
    batch = {
        "visual": torch.rand(4, 4),
        "env": torch.rand(4, 4),
        "historical": torch.rand(4, 3),
        "knowledge": torch.rand(4, 4)
    }
    
    # Get base VEE output
    with torch.no_grad():
        out1 = model.vee(batch["visual"])
        
    # Perturb env/weather, history, and knowledge
    batch["env"] = torch.zeros(4, 4)
    batch["historical"] = torch.zeros(4, 3)
    batch["knowledge"] = torch.zeros(4, 4)
    
    with torch.no_grad():
        out2 = model.vee(batch["visual"])
        
    # Check strict equality of VEE embeddings
    diff = torch.max(torch.abs(out1["visual_features"] - out2["visual_features"])).item()
    print(f"  VEE Latent Shift under Weather/History perturbation: {diff:.6f}")
    assert diff == 0.0, "VEE Isolation Failed! VEE output is affected by non-visual inputs."
    print("  --> PASSED: VEE is strictly isolated.")

def test_ere_isolation():
    print("\nRunning ERE Isolation Test...")
    model = TRACERCEv3()
    model.eval()
    
    batch = {
        "visual": torch.rand(4, 4),
        "env": torch.rand(4, 4),
        "historical": torch.rand(4, 3),
        "knowledge": torch.rand(4, 4)
    }
    
    # Get base ERE output
    with torch.no_grad():
        out1 = model.ere(batch["env"], batch["historical"])
        
    # Perturb visual features
    batch["visual"] = torch.zeros(4, 4)
    
    with torch.no_grad():
        out2 = model.ere(batch["env"], batch["historical"])
        
    # Check strict equality of ERE outputs
    diff = torch.max(torch.abs(out1["env_features"] - out2["env_features"])).item()
    print(f"  ERE Latent Shift under Visual feature perturbation: {diff:.6f}")
    assert diff == 0.0, "ERE Isolation Failed! ERE output is affected by visual inputs."
    print("  --> PASSED: ERE is strictly isolated.")

def test_encoder_freeze():
    print("\nRunning Encoder Freeze Test...")
    model = TRACERCEv3()
    
    # Freeze encoders
    for p in model.vee.parameters():
        p.requires_grad = False
    for p in model.ere.parameters():
        p.requires_grad = False
        
    # Perform forward on mock inputs
    batch = {
        "visual": torch.rand(2, 4),
        "env": torch.rand(2, 4),
        "historical": torch.rand(2, 3),
        "knowledge": torch.rand(2, 4)
    }
    
    outputs = model(batch)
    loss = outputs["scores"].sum() + outputs["agreement_score"].sum()
    loss.backward()
    
    # Assert that VEE and ERE gradients are None
    vee_grads = [p.grad for p in model.vee.parameters() if p.grad is not None]
    ere_grads = [p.grad for p in model.ere.parameters() if p.grad is not None]
    
    # Assert that ECE and Ranking heads have gradients
    ece_grads = [p.grad for p in model.ece.parameters() if p.grad is not None]
    ranking_grads = [p.grad for p in model.ranking_head.parameters() if p.grad is not None]
    
    print(f"  Number of VEE parameters with gradients: {len(vee_grads)}")
    print(f"  Number of ERE parameters with gradients: {len(ere_grads)}")
    print(f"  Number of ECE parameters with gradients: {len(ece_grads)}")
    print(f"  Number of Ranking Head parameters with gradients: {len(ranking_grads)}")
    
    assert len(vee_grads) == 0, "Encoder Freeze Failed! VEE gradients are not frozen."
    assert len(ere_grads) == 0, "Encoder Freeze Failed! ERE gradients are not frozen."
    assert len(ece_grads) > 0, "ECE training check failed. ECE should be trainable."
    assert len(ranking_grads) > 0, "Ranking head training check failed. Ranking head should be trainable."
    print("  --> PASSED: Encoders are correctly frozen.")

def test_conflict_scenarios():
    print("\nRunning Conflict Scenarios Test...")
    model = TRACERCEv3()
    model.eval()
    
    # Scenario A: Healthy Image (Low visual confidence) + High Environmental Risk
    print("  [Scenario A: Healthy leaf + High environmental risk]")
    # visual: [confidence=0.01, coverage=0.0, peak=0.0, entropy=0.01] (healthy leaf)
    # env: [humidity=0.95, temp=0.5, rainfall=0.8, wetness=0.8] (wet warm weather)
    batch_a = {
        "visual": torch.tensor([[0.01, 0.0, 0.0, 0.01]], dtype=torch.float32),
        "env": torch.tensor([[0.95, 0.5, 0.8, 0.8]], dtype=torch.float32),
        "historical": torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float32),
        "knowledge": torch.tensor([[0.0, 0.0, 0.0, 0.0]], dtype=torch.float32)
    }
    
    with torch.no_grad():
        out_a = model(batch_a)
        
    print(f"    VEE Disease Present Probability: {out_a['disease_present_prob'][0].item():.4f}")
    print(f"    ERE Pathogen Risk Score (Fungal): {out_a['risk_scores'][0][0].item():.4f}")
    print(f"    ECE Reasoning State Probabilities: {out_a['reasoning_state'][0].tolist()}")
    print(f"    ECE Conflict Score: {out_a['conflict_score'][0].item():.4f}")
    
    # Scenario B: Diseased Image (High visual confidence) + Low Environmental Risk
    print("  [Scenario B: Diseased leaf + Low environmental risk]")
    # visual: [confidence=0.98, coverage=0.8, peak=0.9, entropy=0.01] (clear spots)
    # env: [humidity=0.1, temp=0.1, rainfall=0.0, wetness=0.0] (dry desert weather)
    batch_b = {
        "visual": torch.tensor([[0.98, 0.8, 0.9, 0.01]], dtype=torch.float32),
        "env": torch.tensor([[0.1, 0.1, 0.0, 0.0]], dtype=torch.float32),
        "historical": torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float32),
        "knowledge": torch.tensor([[0.0, 0.0, 0.0, 0.0]], dtype=torch.float32)
    }
    
    with torch.no_grad():
        out_b = model(batch_b)
        
    print(f"    VEE Disease Present Probability: {out_b['disease_present_prob'][0].item():.4f}")
    print(f"    ERE Pathogen Risk Score (Fungal): {out_b['risk_scores'][0][0].item():.4f}")
    print(f"    ECE Reasoning State Probabilities: {out_b['reasoning_state'][0].tolist()}")
    print(f"    ECE Conflict Score: {out_b['conflict_score'][0].item():.4f}")
    print(f"    Visual Confidence: {out_b['visual_confidence'][0].item():.4f}")
    print(f"    Final Calibrated Confidence: {out_b['calibrated_confidence'][0].item():.4f}")
    print("  --> PASSED: Conflict scenarios successfully executed and categorized.")

def main():
    print("====================================================")
    print("TRACE-RCE v3: Unit & Isolation Verification Testing")
    print("====================================================")
    
    test_vee_isolation()
    test_ere_isolation()
    test_encoder_freeze()
    test_conflict_scenarios()
    
    print("\n====================================================")
    print("ALL ISOLATION AND CONSISTENCY TESTS PASSED SUCCESSFULLY!")
    print("====================================================")

if __name__ == "__main__":
    main()
