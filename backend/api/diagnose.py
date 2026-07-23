import os
import base64
import cv2
import numpy as np
from fastapi import APIRouter, File, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse
import torch
from torchvision import transforms
from PIL import Image
import io

from backend.models.vision_training import build_model
from backend.models.explainability import GradCAM
from backend.models.severity_scorer import SeverityScoringEngine

router = APIRouter()

severity_engine = SeverityScoringEngine()
NUM_CLASSES = 35
WEIGHTS_PATH = r"c:\Users\ABHIRAM MODUKURU\OneDrive\Desktop\AgriVision-AI\backend\models\weights\nova_mobilenet_v3.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

_model = None
_cam = None

# Base TTA Augmentations (These are applied dynamically at test time)
tta_transforms = [
    transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]),
    transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=1.0),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]),
    transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomVerticalFlip(p=1.0),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]),
    transforms.Compose([
        transforms.Resize((240, 240)),
        transforms.CenterCrop((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]),
    transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]),
]

def load_ai_pipeline():
    global _model, _cam
    if _model is None:
        if os.path.exists(WEIGHTS_PATH):
            _model = build_model(NUM_CLASSES)
            _model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=DEVICE))
            _model.eval()
            _cam = GradCAM(_model, _model.features[-1])
        else:
            return False 
    return True

def image_to_base64(img_array):
    _, buffer = cv2.imencode('.jpg', img_array)
    b64_string = base64.b64encode(buffer).decode('utf-8')
    return f"data:image/jpeg;base64,{b64_string}"

@router.post("/")
async def run_diagnosis(
    image: UploadFile = File(...),
    temperature: float = Form(25.0), 
    humidity: float = Form(60.0),    
    ph_level: float = Form(6.5)      
):
    try:
        contents = await image.read()
        pil_image = Image.open(io.BytesIO(contents)).convert('RGB')
        
        has_real_model = load_ai_pipeline()
        
        if has_real_model:
            # --- TEST TIME AUGMENTATION (TTA) ---
            # Instead of 1 prediction, we run the image through 5 different augmented states
            tta_outputs = []
            with torch.no_grad():
                for t in tta_transforms:
                    input_tensor = t(pil_image).unsqueeze(0).to(DEVICE)
                    output = _model(input_tensor)
                    probabilities = torch.nn.functional.softmax(output[0], dim=0)
                    tta_outputs.append(probabilities)
            
            # Average the probabilities across all 5 augmented inferences
            avg_probs = torch.stack(tta_outputs).mean(dim=0)
            class_idx = torch.argmax(avg_probs).item()
            confidence = avg_probs[class_idx].item()
            
            # --- GRAD-CAM ---
            # We generate the Grad-CAM purely on the standard base image (tta_transforms[0]) 
            # so the heatmap overlays perfectly onto the farmer's raw photo.
            base_tensor = tta_transforms[0](pil_image).unsqueeze(0).to(DEVICE)
            heatmap, _, _ = _cam.generate_heatmap(base_tensor, class_idx=class_idx)
            
            predicted_disease = "tomato_late_blight" # Placeholder
            
            lesion_ratio = _cam.calculate_lesion_area_ratio(heatmap)
            
            cv_img = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
            heatmap_resized = cv2.resize(heatmap, (cv_img.shape[1], cv_img.shape[0]))
            heatmap_uint8 = np.uint8(255 * heatmap_resized)
            heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
            superimposed_img = heatmap_color * 0.4 + cv_img * 0.6
            
            heatmap_b64 = image_to_base64(superimposed_img)
            
        else:
            predicted_disease = "tomato_late_blight"
            confidence = 0.92
            lesion_ratio = 0.45
            
            cv_img = cv2.cvtColor(np.array(pil_image.resize((224, 224))), cv2.COLOR_RGB2BGR)
            heatmap_color = np.zeros_like(cv_img)
            cv2.circle(heatmap_color, (112, 112), 50, (0, 0, 255), -1)
            superimposed_img = heatmap_color * 0.4 + cv_img * 0.6
            heatmap_b64 = image_to_base64(superimposed_img)
            
        severity_result = severity_engine.calculate_final_severity(
            disease_name=predicted_disease,
            ml_confidence=confidence,
            lesion_area_ratio=lesion_ratio,
            humidity=humidity,
            temp_c=temperature,
            ph_level=ph_level
        )
        
        return JSONResponse(content={
            "status": "success",
            "is_mock_data": not has_real_model,
            "prediction": {
                "disease": predicted_disease,
                "confidence": round(confidence * 100, 2),
                "tta_enabled": True # Letting the frontend know TTA was used
            },
            "severity_assessment": severity_result,
            "gradcam_heatmap_b64": heatmap_b64
        })
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
