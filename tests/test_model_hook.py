import sys
import os
import traceback
import torch
import torchvision.models as tv_models
import torch.nn as nn

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from backend.models.explainability import GradCAM

def test():
    print("Initializing...")
    try:
        model = tv_models.mobilenet_v3_small(weights=None)
        in_features = model.classifier[3].in_features
        model.classifier[3] = nn.Linear(in_features, 36)
        model.eval()
        
        target_layer = model.features[-1]
        print(f"Target Layer: {target_layer}")
        
        cam = GradCAM(model, target_layer)
        x = torch.randn(1, 3, 224, 224)
        
        print("Generating heatmap...")
        h, o, c = cam.generate_heatmap(x, class_idx=0)
        print(f"Success! Heatmap shape: {h.shape}")
    except Exception as e:
        traceback.print_exc()

if __name__ == "__main__":
    test()
