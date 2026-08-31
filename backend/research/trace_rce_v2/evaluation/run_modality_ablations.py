import os
import sys
import json
import torch
from typing import Dict, Any, List

# Add root directory to sys path
sys.path.append(r"c:\Users\ABHIRAM MODUKURU\OneDrive\Desktop\AgriVision-AI")

from backend.research.trace_rce_v2.dataset.dataset import CAUSE_IDS
from backend.research.trace_rce_v2.dataset.novarcd_dataset import create_novarcd_dataloaders
from backend.research.trace_rce_v2.models.trace_rce_v2 import build_model
from backend.research.trace_rce_v2.evaluation.evaluator import run_v2_evaluation

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
WEIGHTS_PATH = r"c:\Users\ABHIRAM MODUKURU\OneDrive\Desktop\AgriVision-AI\backend\research\trace_rce_v2\checkpoints\best_novarcd_model.pt"

class AblatedLoader:
    def __init__(self, loader, modality):
        self.loader = loader
        self.modality = modality
        self.dataset = loader.dataset

    def __iter__(self):
        for batch in self.loader:
            if self.modality:
                batch[self.modality] = torch.zeros_like(batch[self.modality])
            yield batch

def main():
    print("========================================")
    print("NOVA-RCD: Test-Time Modality Ablation Study")
    print("========================================")
    
    # 1. Load splits
    _, _, test_loader = create_novarcd_dataloaders(batch_size=32, augmentation_factor=0)
    
    # 2. Load model
    model = build_model()
    checkpoint = torch.load(WEIGHTS_PATH, map_location=DEVICE)
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)
    model.to(DEVICE)
    model.eval()

    # 3. Evaluate full model (No Ablation)
    print("Evaluating Full Model...")
    full_metrics, _ = run_v2_evaluation(model, test_loader, DEVICE)
    results = {"Full Model": full_metrics}
    
    # 4. Evaluate each modality ablated
    modalities = ["visual", "env", "severity", "knowledge", "historical"]
    for m in modalities:
        print(f"Evaluating model with ablated modality: {m}...")
        ablated_loader = AblatedLoader(test_loader, m)
        metrics, _ = run_v2_evaluation(model, ablated_loader, DEVICE)
        results[f"Ablated {m}"] = metrics
        
    print("\n[Ablation Results Summary]")
    print(f"{'Configuration':<20} | {'NDCG@3':<8} | {'MRR':<8} | {'P@1':<8} | {'R@3':<8} | {'ECE':<8}")
    print("-" * 70)
    for name, m in results.items():
        print(f"{name:<20} | {m['ndcg_at_3']:.4f}   | {m['mrr']:.4f}   | {m['precision_at_1']:.4f}   | {m['recall_at_3']:.4f}   | {m['expected_calibration_error']:.4f}")
        
    # Save results
    save_path = r"c:\Users\ABHIRAM MODUKURU\OneDrive\Desktop\AgriVision-AI\backend\research\trace_rce_v2\evaluation\novarcd_modality_ablations.json"
    with open(save_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nAblation results saved to {save_path}")

if __name__ == "__main__":
    main()
