import os
import sys
import json
import torch
import torch.nn as nn
import torch.optim as optim

# Add project root to sys.path
sys.path.append(r"c:\Users\ABHIRAM MODUKURU\OneDrive\Desktop\AgriVision-AI")

from backend.research.trace_rce_v3.dataset.novarcd_v3_dataset import create_novarcd_v3_dataloaders
from backend.research.trace_rce_v3.models.trace_rce_v3 import TRACERCEv3

# Custom ListMLE loss for ranking
class ListMLELoss(nn.Module):
    def forward(self, scores: torch.Tensor, target_ranks: torch.Tensor) -> torch.Tensor:
        batch_size, n_causes = scores.shape
        sorted_indices = torch.argsort(target_ranks, dim=1)
        s_sorted = torch.gather(scores, dim=1, index=sorted_indices)
        s_expanded = s_sorted.unsqueeze(1).expand(-1, n_causes, -1)
        mask = torch.triu(torch.ones(n_causes, n_causes, device=scores.device), diagonal=0)
        s_masked = s_expanded.masked_fill(mask.unsqueeze(0) == 0, float('-inf'))
        suffix_logsumexp = torch.logsumexp(s_masked, dim=2)
        per_sample_loss = torch.sum(suffix_logsumexp[:, :-1] - s_sorted[:, :-1], dim=1)
        return torch.mean(per_sample_loss)

def set_seed(seed: int):
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def train_novarcd_v3():
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training TRACE-RCE v3 on {device}...")
    
    # 1. Load data
    train_loader, val_loader, _ = create_novarcd_v3_dataloaders(batch_size=32, augmentation_factor=2)
    
    # 2. Build model
    model = TRACERCEv3().to(device)
    
    # Define loss criteria
    bce_loss = nn.BCELoss()
    mse_loss = nn.MSELoss()
    ce_loss = nn.CrossEntropyLoss()
    list_mle_loss = ListMLELoss()
    
    # Define optimizer parameters for Stage 1
    # ----------------------------------------------------
    # STAGE 1: Train Visual Evidence Engine (VEE)
    # ----------------------------------------------------
    print("\n[STAGE 1] Training Visual Evidence Engine (VEE) independently...")
    optimizer_vee = optim.AdamW(model.vee.parameters(), lr=1e-3, weight_decay=1e-4)
    model.train()
    for epoch in range(40):
        total_loss = 0.0
        for batch in train_loader:
            optimizer_vee.zero_grad()
            
            visual_in = batch["visual"].to(device)
            target_present = batch["disease_present"].to(device)
            target_severity = batch["severity_target"].to(device)
            
            vee_out = model.vee(visual_in)
            
            loss_present = bce_loss(vee_out["disease_present_prob"], target_present)
            loss_sev = mse_loss(vee_out["severity_score"], target_severity)
            loss = loss_present + loss_sev
            
            loss.backward()
            optimizer_vee.step()
            total_loss += loss.item()
            
        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1:02d}/40 | VEE Loss: {total_loss/len(train_loader):.4f}")
            
    # ----------------------------------------------------
    # STAGE 2: Train Environmental Risk Engine (ERE)
    # ----------------------------------------------------
    print("\n[STAGE 2] Training Environmental Risk Engine (ERE) independently...")
    optimizer_ere = optim.AdamW(model.ere.parameters(), lr=1e-3, weight_decay=1e-4)
    for epoch in range(40):
        total_loss = 0.0
        for batch in train_loader:
            optimizer_ere.zero_grad()
            
            env_in = batch["env"].to(device)
            hist_in = batch["historical"].to(device)
            target_risk = batch["risk_profile"].to(device)
            
            ere_out = model.ere(env_in, hist_in)
            
            loss = bce_loss(ere_out["risk_scores"], target_risk)
            
            loss.backward()
            optimizer_ere.step()
            total_loss += loss.item()
            
        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1:02d}/40 | ERE Loss: {total_loss/len(train_loader):.4f}")
            
    # ----------------------------------------------------
    # STAGE 3: Freeze VEE and ERE Encoders
    # ----------------------------------------------------
    print("\n[STAGE 3] Freezing VEE and ERE encoders...")
    for p in model.vee.parameters():
        p.requires_grad = False
    for p in model.ere.parameters():
        p.requires_grad = False
        
    # ----------------------------------------------------
    # STAGE 4: Train KE, ECE, and Final Reasoning Head
    # ----------------------------------------------------
    print("\n[STAGE 4] Training KE, ECE, Calibration, and Reasoning Heads...")
    # Select only trainable parameters
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer_final = optim.AdamW(trainable_params, lr=1e-3, weight_decay=1e-4)
    
    for epoch in range(80):
        total_loss = 0.0
        total_rank_loss = 0.0
        total_ece_loss = 0.0
        for batch in train_loader:
            optimizer_final.zero_grad()
            
            # Map batch inputs
            inputs = {
                "visual": batch["visual"].to(device),
                "env": batch["env"].to(device),
                "historical": batch["historical"].to(device),
                "knowledge": batch["knowledge"].to(device)
            }
            
            target_ranks = batch["cause_ranking"].to(device)
            target_labels = batch["binary_labels"].to(device)
            target_state = batch["reasoning_state"].to(device)
            
            outputs = model(inputs)
            
            # Compute losses
            # A. ListNet ranking loss
            l_rank = list_mle_loss(outputs["scores"], target_ranks)
            
            # B. Brier calibration loss on causes
            # Cause predictions = outputs["confidences"], targets = target_labels
            l_cal = mse_loss(outputs["confidences"], target_labels)
            
            # C. ECE reasoning state loss
            l_state = ce_loss(outputs["state_logits"], target_state)
            
            # D. Agreement / Conflict score regression loss
            # Construct target agreement/conflict based on target state
            target_agree = torch.where(target_state == 0, 1.0, 0.0)
            target_conflict = torch.where(target_state >= 2, 1.0, 0.0)
            
            l_agree = mse_loss(outputs["agreement_score"], target_agree)
            l_conflict = mse_loss(outputs["conflict_score"], target_conflict)
            
            # Total multi-task loss
            loss = l_rank + 0.5 * l_cal + 0.3 * l_state + 0.1 * l_agree + 0.1 * l_conflict
            
            loss.backward()
            optimizer_final.step()
            
            total_loss += loss.item()
            total_rank_loss += l_rank.item()
            total_ece_loss += l_state.item()
            
        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1:02d}/80 | Total: {total_loss/len(train_loader):.4f} | Rank: {total_rank_loss/len(train_loader):.4f} | ECE: {total_ece_loss/len(train_loader):.4f}")
            
    # Save checkpoint
    checkpoint_dir = r"c:\Users\ABHIRAM MODUKURU\OneDrive\Desktop\AgriVision-AI\backend\research\trace_rce_v3\checkpoints"
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, "best_novarcd_model.pt")
    
    # Save both model weights and metadata config
    torch.save({
        "model_state_dict": model.state_dict(),
        "config": {
            "latent_dim": model.latent_dim,
            "num_classes": model.num_classes,
            "num_risks": model.num_risks,
            "num_causes": model.num_causes
        }
    }, checkpoint_path)
    print(f"\nTRACE-RCE v3 training complete. Checkpoint saved to {checkpoint_path}")

if __name__ == "__main__":
    train_novarcd_v3()
