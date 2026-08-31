import os
import torch
import torch.nn as nn
from torchvision import models
from backend.models.class_registry import CLASS_NAMES, NUM_CLASSES

WEIGHTS_DIR = os.path.normpath(r"c:\Users\ABHIRAM MODUKURU\OneDrive\Desktop\AgriVision-AI\backend\models\weights")
OLD_CHECKPOINT = os.path.join(WEIGHTS_DIR, "nova_mobilenet_v3_24_classes.pth")
NEW_CHECKPOINT = os.path.join(WEIGHTS_DIR, "nova_mobilenet_v3_26_classes_init.pth")

OLD_NUM_CLASSES = 24
NEW_NUM_CLASSES = 26

def expand_checkpoint():
    print(f"Loading {OLD_NUM_CLASSES}-class baseline: {OLD_CHECKPOINT}")
    
    # Load old state dict
    old_state = torch.load(OLD_CHECKPOINT, map_location="cpu")
    
    # Build new 26-class model
    print(f"Building new {NEW_NUM_CLASSES}-class architecture...")
    model = models.mobilenet_v3_small(weights=None)
    in_features = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(in_features, NEW_NUM_CLASSES)
    
    new_state = model.state_dict()
    
    print("Performing surgical weight transfer...")
    
    for name, param in old_state.items():
        if "classifier.3" not in name:
            # For backbone and other layers, exact copy
            new_state[name].copy_(param)
        else:
            # Handle classification head
            is_weight = "weight" in name
            new_param = new_state[name]
            
            # Indices mapping logic:
            # Old classes: 0, 1, 2 (corn) -> New classes: 0, 1, 2
            # New classes 3, 4 (cotton) are randomly initialized
            # Old classes 3..23 -> New classes 5..25
            
            # Copy first 3 (corn)
            new_param[:3] = param[:3]
            
            # Copy remaining 21 classes, shifted by 2
            new_param[5:] = param[3:]
            
            # The middle 2 (cotton) remain randomly initialized by nn.Linear
            
    # Verify exact match for a shifted class (e.g., pepper)
    diff = torch.sum(torch.abs(new_state['classifier.3.weight'][5] - old_state['classifier.3.weight'][3]))
    assert diff.item() < 1e-6, "Weight transfer assertion failed!"
    
    # Save the expanded checkpoint
    torch.save(new_state, NEW_CHECKPOINT)
    print(f"Surgery complete! Expanded checkpoint saved to: {NEW_CHECKPOINT}")

if __name__ == "__main__":
    expand_checkpoint()
