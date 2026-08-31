import torch
from torchvision import models
import torch.nn as nn

try:
    weights_path = r"C:\Users\ABHIRAM MODUKURU\OneDrive\Desktop\AgriVision-AI\experiments\exp_C\best_model.pth"
    state_dict = torch.load(weights_path, map_location="cpu")
    print("State dict loaded.")
    
    # Check output dimension from the weight tensor of the final layer
    final_weight = state_dict['classifier.3.weight']
    out_dim, in_dim = final_weight.shape
    print(f"Final layer output dimension: {out_dim}")
    print(f"Final layer input dimension: {in_dim}")
    
    if out_dim == 27:
        print("VERIFIED: Model output dimension matches 27 classes.")
    else:
        print(f"WARNING: Expected 27 classes, got {out_dim}")
except Exception as e:
    print(f"Error during verification: {e}")
