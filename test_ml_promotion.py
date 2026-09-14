import sys
from pathlib import Path
import json

REPO_ROOT = Path(r"c:\Users\ABHIRAM MODUKURU\OneDrive\Desktop\AgriVision-AI")
sys.path.append(str(REPO_ROOT))

from backend.models.inference import load_model, predict
from backend.models.cam import generate_gradcam
from backend.models.concern_scoring import calculate_concern_score, RAG_AVAILABLE
from backend.models.class_registry import CLASS_NAMES

print("\n--- 1. Class Registry Verification ---")
print(f"Total Classes: {len(CLASS_NAMES)}")
print("Matches 34 classes exactly: ", len(CLASS_NAMES) == 34)
print("First 3 classes:", CLASS_NAMES[:3])

print("\n--- 2. Loading Model ---")
try:
    model = load_model()
    print("Model loaded successfully from production path (nova_mobilenet_v3_34_classes.pth)")
except Exception as e:
    print(f"Failed to load model: {e}")
    sys.exit(1)

print("\n--- 3. Testing Prediction & TTA ---")
image_path = REPO_ROOT / "backend/data/processed_dataset/tomato_late_blight/plantvillagedataset_0.JPG"
with open(image_path, "rb") as f:
    img_bytes = f.read()

try:
    pred_class, conf, all_probs = predict(img_bytes, model, use_tta=True)
    print(f"Prediction successful (TTA=True).")
    print(f"Predicted Class: {pred_class}")
    print(f"Confidence: {conf:.4f}")
except Exception as e:
    print(f"Prediction failed: {e}")

print("\n--- 4. Testing Grad-CAM ---")
try:
    cam_base64 = generate_gradcam(img_bytes, model)
    print(f"Grad-CAM generated successfully. Base64 length: {len(cam_base64)}")
except Exception as e:
    print(f"Grad-CAM failed: {e}")

print("\n--- 5. Testing Concern Score ---")
try:
    # Dummy inputs simulating IoT
    score, context = calculate_concern_score(
        disease_name=pred_class,
        confidence=conf,
        temperature=26.5,
        humidity=85.0
    )
    print(f"Concern Score generated: {score:.2f}/10.0")
    print(f"RAG Used: {context['rag_used']}")
except Exception as e:
    print(f"Concern score calculation failed: {e}")

print("\n=== PROMOTION TEST COMPLETE ===")
