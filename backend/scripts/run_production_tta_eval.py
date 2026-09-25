import torch
import json
import os
import numpy as np
import time
from PIL import Image
from sklearn.metrics import accuracy_score, f1_score
from torchvision import models, transforms

import sys
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(REPO_ROOT)

from backend.models.class_registry import NUM_CLASSES, CLASS_TO_IDX, MODEL_CONFIG
from backend.api.diagnose import tta_transforms

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    model = models.mobilenet_v3_small(weights=None)
    model.classifier[3] = torch.nn.Linear(model.classifier[3].in_features, NUM_CLASSES)
    model_path = os.path.join(REPO_ROOT, 'backend', 'models', 'weights', 'nova_mobilenet_v3_34_classes.pth')
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model = model.to(device)
    model.eval()

    single_pass_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=MODEL_CONFIG['normalize_mean'], std=MODEL_CONFIG['normalize_std'])
    ])

    def evaluate_dataset(paths):
        targs, single_preds, tta_preds = [], [], []
        single_probs_all, tta_probs_all = [], []
        
        start = time.perf_counter()
        
        with torch.no_grad():
            for p in paths:
                cls_name = os.path.basename(os.path.dirname(p))
                if cls_name not in CLASS_TO_IDX: continue
                
                p_full = p
                if not os.path.isabs(p_full):
                    p_field = os.path.join(REPO_ROOT, 'backend', 'data', 'processed_field_dataset', p)
                    p_lab = os.path.join(REPO_ROOT, p)
                    if os.path.exists(p_field):
                        p_full = p_field
                    elif os.path.exists(p_lab):
                        p_full = p_lab
                    
                if not os.path.exists(p_full): continue
                
                img = Image.open(p_full).convert('RGB')
                targs.append(CLASS_TO_IDX[cls_name])
                
                # Single pass
                s_tensor = single_pass_transform(img).unsqueeze(0).to(device)
                s_out = model(s_tensor)
                s_prob = torch.nn.functional.softmax(s_out, dim=1)
                single_probs_all.append(s_prob[0].cpu().numpy())
                single_preds.append(torch.argmax(s_prob, dim=1).item())
                
                # TTA pass
                tta_outs = []
                for t in tta_transforms:
                    tensor = t(img).unsqueeze(0).to(device)
                    tta_outs.append(torch.nn.functional.softmax(model(tensor), dim=1))
                t_prob = torch.stack(tta_outs).mean(dim=0)
                tta_probs_all.append(t_prob[0].cpu().numpy())
                tta_preds.append(torch.argmax(t_prob, dim=1).item())
                
        def compute_metrics(y_true, y_pred, y_prob):
            if len(y_true) == 0: return 0, 0, 0, 0
            acc = accuracy_score(y_true, y_pred)
            mac = f1_score(y_true, y_pred, average='macro', zero_division=0)
            wei = f1_score(y_true, y_pred, average='weighted', zero_division=0)
            top3 = sum([1 if t in np.argsort(p)[-3:] else 0 for t, p in zip(y_true, y_prob)]) / len(y_true)
            return acc, mac, wei, top3
            
        s_acc, s_mac, s_wei, s_top3 = compute_metrics(targs, single_preds, single_probs_all)
        t_acc, t_mac, t_wei, t_top3 = compute_metrics(targs, tta_preds, tta_probs_all)
        
        lat = (time.perf_counter() - start) / len(targs) if len(targs) > 0 else 0
        
        return (s_acc, s_mac, s_wei, s_top3, lat), (t_acc, t_mac, t_wei, t_top3, lat)

    # Load Field Data
    print('Loading datasets...')
    with open(os.path.join(REPO_ROOT, 'backend', 'data', 'field_splits.json')) as f:
        field_paths = json.load(f)['test']

    # Load Lab Data
    lab_paths = []
    with open(os.path.join(REPO_ROOT, 'evaluation', 'clean_test_split.json')) as f:
        lab_data = json.load(f)
        if isinstance(lab_data, list):
            lab_paths = [item['path'] for item in lab_data if item['class_name'] in CLASS_TO_IDX]
        elif isinstance(lab_data, dict):
            for k, v in lab_data.items():
                if k in CLASS_TO_IDX:
                    lab_paths.extend(v)

    print('Evaluating Field Test Set (Untouched)...')
    field_single, field_tta = evaluate_dataset(field_paths)

    print('Evaluating Lab Test Set (Authoritative)...')
    lab_single, lab_tta = evaluate_dataset(lab_paths)

    print('\n=============================================')
    print('AUTHORITATIVE LIVE PRODUCTION FIELD:')
    print(f'Accuracy:    {field_tta[0]:.4f}')
    print(f'Macro F1:    {field_tta[1]:.4f}')
    print(f'Weighted F1: {field_tta[2]:.4f}')
    print(f'Top-3:       {field_tta[3]:.4f}')

    print('\nAUTHORITATIVE LIVE PRODUCTION LAB:')
    print(f'Accuracy:    {lab_tta[0]:.4f}')
    print(f'Macro F1:    {lab_tta[1]:.4f}')
    print(f'Weighted F1: {lab_tta[2]:.4f}')
    print(f'Top-3:       {lab_tta[3]:.4f}')

    print('\n=============================================')
    print('SINGLE-PASS FIELD (FOR REFERENCE ONLY):')
    print(f'Accuracy:    {field_single[0]:.4f}')
    print(f'Macro F1:    {field_single[1]:.4f}')

    print('\nSINGLE-PASS LAB (FOR REFERENCE ONLY):')
    print(f'Accuracy:    {lab_single[0]:.4f}')
    print(f'Macro F1:    {lab_single[1]:.4f}')

if __name__ == '__main__':
    main()
