import sys
import os
import torch
import torchvision.models as models

sys.path.append(os.path.abspath('backend'))
from backend.models.class_registry import CLASS_NAMES, NUM_CLASSES

checkpoint_path = 'backend/models/weights/nova_mobilenet_v3_34_classes.pth'
state_dict = torch.load(checkpoint_path, map_location='cpu')

# In MobileNetV3-Small, the classifier is a Sequential. 
# We need to find its exact output dimension from the state_dict.
classifier_weight_key = 'classifier.3.weight'
if classifier_weight_key in state_dict:
    out_dim = state_dict[classifier_weight_key].shape[0]
else:
    # try to find the last weight
    last_key = list(state_dict.keys())[-1]
    out_dim = state_dict[last_key].shape[0] if len(state_dict[last_key].shape) > 0 else "unknown"

print('=== MODEL ARCHITECTURE ===')
print('Architecture hint: MobileNetV3-Small')
print('Output dimension (from state dict):', out_dim)
print('Registry NUM_CLASSES:', NUM_CLASSES)
print('CLASS_NAMES length:', len(CLASS_NAMES))

match = (out_dim == NUM_CLASSES)
print('Output dim matches registry:', match)
