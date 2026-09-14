import torch
from torchvision import transforms
from PIL import Image
from backend.models.class_registry import NUM_CLASSES, idx_to_class
from backend.api.diagnose import _build_mobilenet_v3_small, tta_transforms, WEIGHTS_PATH

DEVICE = torch.device('cpu')
model = _build_mobilenet_v3_small(NUM_CLASSES)
model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=DEVICE))
model.eval()

images = {
    'Diseased (Lab)': r'c:\Users\ABHIRAM MODUKURU\OneDrive\Desktop\AgriVision-AI\backend\data\processed_dataset\tomato_late_blight\plantvillagedataset_0.JPG',
    'Healthy (Lab)': r'c:\Users\ABHIRAM MODUKURU\OneDrive\Desktop\AgriVision-AI\backend\data\processed_dataset\tomato_healthy\plantvillagedataset_0.JPG',
    'Field': r'c:\Users\ABHIRAM MODUKURU\OneDrive\Desktop\AgriVision-AI\backend\data\processed_field_dataset\rice_leaf_blast\DSC_0101.JPG'
}

for name, path in images.items():
    try:
        img = Image.open(path).convert('RGB')
        
        # Test TTA #1
        t_img = tta_transforms[0](img).unsqueeze(0)
        
        with torch.no_grad():
            out = model(t_img)
            probs = torch.nn.functional.softmax(out[0], dim=0)
            conf, idx = torch.max(probs, dim=0)
            
        print(f'{name} -> {idx_to_class(idx.item())} ({conf.item():.2f})')
    except Exception as e:
        print(f'{name} -> Failed: {e}')
