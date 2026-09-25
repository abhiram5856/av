import os
import base64
import uuid
import cv2
import numpy as np
from fastapi import APIRouter, File, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models
from PIL import Image
import io
import time
import json

from backend.models.class_registry import (
    CLASS_NAMES, NUM_CLASSES, MODEL_CONFIG,
    idx_to_class, class_to_display, is_healthy
)
from backend.models.explainability import GradCAM
from backend.models.concern_scorer import MultimodalConcernScorer
from backend.services.context_builder import AIContextBuilder
from backend.services.weather import WeatherService
from backend.services.multimodal_evidence_engine import MultimodalEvidenceEngine
from backend.services.image_quality_gate import ImageQualityGate
from backend.research.trace_rce_v3.inference.engine import TRACERootCauseEngineV3
from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession
from backend.core.database import get_db
from backend.data.crud import get_or_create_user, create_diagnosis
from backend.auth.security import verify_token
from backend.core_logging.logger import api_logger

from pathlib import Path

router = APIRouter()

BASE_DIR = Path(__file__).resolve().parent.parent.parent
RULES_PATH = BASE_DIR / "knowledge_base" / "agronomic_rules.json"
TRACE_RCE_CKPT = BASE_DIR / "backend" / "research" / "trace_rce_v3" / "checkpoints" / "best_model.pt"

concern_scorer = MultimodalConcernScorer()
evidence_engine = MultimodalEvidenceEngine(str(RULES_PATH))
image_gate = ImageQualityGate()

root_cause_engine = TRACERootCauseEngineV3(
    checkpoint_path=str(TRACE_RCE_CKPT) if TRACE_RCE_CKPT.exists() else None
)

# ─────────────────────────────────────────────────────────────────────────────
# Model configuration — pulled exclusively from class_registry.py
# ─────────────────────────────────────────────────────────────────────────────
WEIGHTS_PATH = os.path.join(
    os.path.dirname(__file__), "..", "models", "weights",
    MODEL_CONFIG["checkpoint_filename"]
)
WEIGHTS_PATH = os.path.normpath(WEIGHTS_PATH)

if os.environ.get("NOVA_FORCE_CPU") == "true":
    DEVICE = torch.device("cpu")
else:
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

_model: nn.Module = None
_cam: GradCAM = None


# ─────────────────────────────────────────────────────────────────────────────
# Test-Time Augmentation (TTA) transforms
# These must match the val/inference transforms used during training.
# ─────────────────────────────────────────────────────────────────────────────
_MEAN = MODEL_CONFIG["normalize_mean"]
_STD  = MODEL_CONFIG["normalize_std"]
_SIZE = MODEL_CONFIG["input_size"][0]  # 224



tta_transforms = [
    # 1. Standard centre crop
    transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(_MEAN, _STD),
    ]),
    # 2. Horizontal flip
    transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(_SIZE),
        transforms.RandomHorizontalFlip(p=1.0),
        transforms.ToTensor(),
        transforms.Normalize(_MEAN, _STD),
    ]),
    # 3. Vertical flip
    transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(_SIZE),
        transforms.RandomVerticalFlip(p=1.0),
        transforms.ToTensor(),
        transforms.Normalize(_MEAN, _STD),
    ]),
    # 4. Slightly larger crop
    transforms.Compose([
        transforms.Resize(240),
        transforms.CenterCrop(_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(_MEAN, _STD),
    ]),
    # 5. Even larger crop
    transforms.Compose([
        transforms.Resize(272),
        transforms.CenterCrop(_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(_MEAN, _STD),
    ]),
]

def _build_mobilenet_v3_small(num_classes: int) -> nn.Module:
    """
    Build MobileNetV3-Small with a custom classification head.
    Architecture exactly matches the saved checkpoint (nova_mobilenet_v3.pth).
    """
    model = models.mobilenet_v3_small(weights=None)
    # MobileNetV3-Small classifier: [Linear, Hardswish, Dropout, Linear]
    in_features = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(in_features, num_classes)
    return model


def load_ai_pipeline() -> bool:
    """
    Loads the MobileNetV3-Small checkpoint into _model and initialises GradCAM.
    Returns True if real model is loaded, False if weights file is missing.
    Idempotent — subsequent calls are no-ops.
    """
    global _model, _cam
    if _model is not None:
        return True

    if not os.path.exists(WEIGHTS_PATH):
        api_logger.warning(
            f"Model weights not found at {WEIGHTS_PATH}. "
            "Running in demo mode — predictions will NOT be real."
        )
        return False

    try:
        api_logger.info(f"Loading MobileNetV3-Small checkpoint from {WEIGHTS_PATH}")
        model = _build_mobilenet_v3_small(NUM_CLASSES)
        state_dict = torch.load(WEIGHTS_PATH, map_location=DEVICE)
        model.load_state_dict(state_dict)
        model.to(DEVICE)
        model.eval()

        # GradCAM targets the last convolutional feature block
        cam = GradCAM(model, model.features[-1])

        _model = model
        _cam   = cam
        api_logger.info(
            f"Model loaded successfully. "
            f"Architecture=MobileNetV3-Small, Classes={NUM_CLASSES}, Device={DEVICE}"
        )
        return True

    except Exception as exc:
        api_logger.error(f"Failed to load model checkpoint: {exc}", exc_info=True)
        return False


def _image_to_base64(img_array: np.ndarray) -> str:
    _, buffer = cv2.imencode(".jpg", img_array)
    return "data:image/jpeg;base64," + base64.b64encode(buffer).decode("utf-8")


# ─────────────────────────────────────────────────────────────────────────────
# Diagnosis endpoint
# ─────────────────────────────────────────────────────────────────────────────

@router.post("/")
async def run_diagnosis(
    image:     UploadFile = File(...),
    ph_level:  float = Form(6.5),
    latitude:  float = Form(17.3850),
    longitude: float = Form(78.4867),
    growth_stage: str = Form("Unknown"),
    user_id:   str = Depends(verify_token),
    db:        AsyncSession = Depends(get_db),
):
    request_id = f"req_{uuid.uuid4()}"
    t_start    = time.perf_counter()

    # ── Live Weather ───────────────────────────────────────────────────────
    weather_data = await WeatherService.fetch_current_weather(latitude, longitude)
    temperature  = weather_data["temperature"]
    humidity     = weather_data["humidity"]

    inference_time = 0.0
    gradcam_time   = 0.0
    rag_time       = 0.001

    # ── Mobile Safety Guardrails ─────────────────────────────────────────
    # 1. Size limit (10 MB)
    MAX_SIZE = 10 * 1024 * 1024
    
    # 2. MIME type check
    if image.content_type not in ["image/jpeg", "image/png", "image/webp"]:
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload a JPEG or PNG image.")

    try:
        contents  = await image.read()
        if len(contents) > MAX_SIZE:
            raise HTTPException(status_code=413, detail="File too large. Maximum size is 10MB.")
            
        try:
            pil_image = Image.open(io.BytesIO(contents)).convert("RGB")
        except Exception as e:
            raise HTTPException(status_code=400, detail="Corrupted image file. Please upload a valid image.")
            
        width, height = pil_image.size

        has_real_model = load_ai_pipeline()

        if has_real_model:
            # ── TTA Inference ──────────────────────────────────────────────
            t_inf_start = time.perf_counter()
            tta_outputs = []
            with torch.no_grad():
                for t in tta_transforms:
                    input_tensor = t(pil_image).unsqueeze(0).to(DEVICE)
                    logits       = _model(input_tensor)
                    probs        = F.softmax(logits[0], dim=0)
                    tta_outputs.append(probs)

            avg_probs    = torch.stack(tta_outputs).mean(dim=0)   # shape: (NUM_CLASSES,)
            class_idx    = int(torch.argmax(avg_probs).item())
            confidence   = float(avg_probs[class_idx].item())
            inference_time = time.perf_counter() - t_inf_start

            # ── Map class_idx → class name using registry ──────────────────
            # RF1 FIX: class name comes from the actual model prediction,
            # NOT a hardcoded constant.
            predicted_disease = idx_to_class(class_idx)

            # Build top-5 predictions for diagnostics
            topk_probs, topk_indices = torch.topk(avg_probs, k=min(5, NUM_CLASSES))
            topk_predictions = {
                idx_to_class(int(i)): float(p)
                for i, p in zip(topk_indices.tolist(), topk_probs.tolist())
            }

            # ── Grad-CAM for the ACTUAL predicted class ────────────────────
            t_cam_start  = time.perf_counter()
            base_tensor  = tta_transforms[0](pil_image).unsqueeze(0).to(DEVICE)
            heatmap, _, _ = _cam.generate_heatmap(base_tensor, class_idx=class_idx)
            lesion_ratio = _cam.calculate_lesion_area_ratio(heatmap)

            cv_img           = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
            heatmap_resized  = cv2.resize(heatmap, (cv_img.shape[1], cv_img.shape[0]))
            heatmap_uint8    = np.uint8(255 * heatmap_resized)
            heatmap_color    = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
            superimposed_img = heatmap_color * 0.4 + cv_img * 0.6
            heatmap_b64      = _image_to_base64(superimposed_img)
            gradcam_time     = time.perf_counter() - t_cam_start

        else:
            api_logger.error(f"Model weights missing. Cannot process request {request_id}")
            raise HTTPException(status_code=503, detail="Service Unavailable: AI model is currently offline.")

        # ── Confidence guard ───────────────────────────────────────────────
        HIGH_CONFIDENCE_THRESHOLD = 0.80
        LOW_CONFIDENCE_THRESHOLD = 0.60
        
        if not has_real_model:
            confidence_category = "MODERATE CONFIDENCE"
        elif confidence >= HIGH_CONFIDENCE_THRESHOLD:
            confidence_category = "HIGH CONFIDENCE"
        elif confidence >= LOW_CONFIDENCE_THRESHOLD:
            confidence_category = "MODERATE CONFIDENCE"
        else:
            confidence_category = "UNCERTAIN"
            
        is_low_confidence = (confidence_category == "UNCERTAIN")
        
        if is_low_confidence:
            predicted_disease = "Unknown"

        # ── Severity & Concern Assessment ──────────────────────────────────
        quality_assessment = image_gate.assess(pil_image)
        
        # Enforce Image Quality and OOD Gate
        if not quality_assessment["allow_analysis"]:
            return JSONResponse(content={
                "status": "rejected",
                "request_id": request_id,
                "is_demo_mode": not has_real_model,
                "diagnosis": {
                    "disease": "OOD",
                    "display_name": "Unknown/Non-Leaf Object",
                    "confidence": 0,
                    "topk": {},
                    "is_healthy": False,
                    "low_confidence_warning": quality_assessment["warnings"][0] if quality_assessment["warnings"] else "Image rejected."
                },
                "visual_evidence": {"gradcam_available": False},
                "concern": {"level": "Uncertain"},
                "recommendation": {"rag_response": {"cause": "Upload a clear picture of a plant leaf.", "solution": "No plant detected."}},
                "image_quality": quality_assessment
            }, status_code=400)
            
        env_evidence = evidence_engine.evaluate(
            disease_name=predicted_disease,
            temperature=temperature,
            humidity=humidity,
            ph=ph_level
        )
        
        concern_result = concern_scorer.calculate_concern(
            disease_name=predicted_disease,
            is_healthy=is_healthy(predicted_disease),
            ml_confidence=confidence,
            lesion_area_ratio=lesion_ratio,
            environmental_evidence=env_evidence,
            growth_stage=growth_stage,
            image_quality_status=quality_assessment["status"]
        )

        # ── AI Context Builder ─────────────────────────────────────────────
        builder = AIContextBuilder(request_id=request_id, user_id=user_id)
        builder.set_image_metadata(
            filename=image.filename or "leaf.jpg",
            w=width, h=height,
            fmt=image.content_type or "image/jpeg",
            blur_score=350.0,
            is_valid=True,
        )
        builder.set_vision_results(
            disease=predicted_disease,
            confidence=confidence,
            topk=topk_predictions,
            concepts={"necrotic_lesions": lesion_ratio, "heatmap_coverage": lesion_ratio},
        )
        builder.set_gradcam_results(
            coverage=lesion_ratio,
            peak=float(np.max(heatmap)) if has_real_model else 0.5,
            layer="features[-1]",
            url=heatmap_b64[:30] + "...",
        )
        builder.set_concern_results(
            base          = concern_result.get("score", 0),
            env_factor    = 1.0,
            soil_factor   = 1.0,
            concern_score = concern_result.get("score", 0),
            concern_level = concern_result.get("level", "Unknown"),
        )
        builder.set_weather_data(
            lat=latitude, lon=longitude,
            temp_avg=temperature, hum_avg=humidity,
            precip=12.5, wetness=10.0,
            raw={"temp": temperature, "humidity": humidity},
        )
        rule = evidence_engine.rules.get(predicted_disease, {})
        if is_healthy(predicted_disease):
            treatment_text = rule.get("treatment", "Crop is healthy. Continue standard maintenance.")
            source_text = "General Agronomy Guidelines"
        else:
            treatment_text = rule.get("treatment", "Consult local agricultural extension for specific treatments.")
            source_text = "AgriVision Plant Pathology Database"

        builder.set_knowledge_context(
            chunk_ids=[f"chk_{predicted_disease}"],
            sources=[source_text],
            text_block=treatment_text,
        )

        ai_context = builder.build()

        # ── Evidence Consistency Engine ────────────────────────────────────
        root_cause_result = await root_cause_engine.analyze(ai_context)

        # ── Monitoring log ─────────────────────────────────────────────────
        elapsed = time.perf_counter() - t_start
        monitoring_payload = {
            "request_id":               request_id,
            "user_id":                  user_id,
            "model_version":            "1.0.0",
            "model_architecture":       MODEL_CONFIG["architecture"],
            "dataset_version":          MODEL_CONFIG["dataset"],
            "predicted_class_idx":      class_idx,
            "predicted_disease":        predicted_disease,
            "confidence":               round(confidence, 4),
            "is_demo_mode":             not has_real_model,
            "is_low_confidence":        is_low_confidence,
            "ai_context_hash":          ai_context.get_content_hash(),
            "device":                   str(DEVICE),
            "inference_latency_s":      round(inference_time, 4),
            "gradcam_latency_s":        round(gradcam_time, 4),
            "total_latency_s":          round(elapsed, 4),
            "request_status":           "success",
        }
        api_logger.info(f"[MONITOR] {json.dumps(monitoring_payload)}")

        # ── Database persistence ───────────────────────────────────────────
        try:
            await get_or_create_user(db, user_id=user_id)
            diagnosis_data = {
                "disease_name":    predicted_disease,
                "confidence":      confidence,
                "concern_level":   concern_result.get("level", "Unknown"),
                "concern_score":   concern_result.get("score", 0),
                "temperature":     temperature,
                "humidity":        humidity,
                "ph_level":        ph_level,
                "gradcam_heatmap": heatmap_b64,
                "root_cause_json": root_cause_result.model_dump(mode="json"),
            }
            await create_diagnosis(db, user_id=user_id, diagnosis_data=diagnosis_data)
        except Exception as db_exc:
            api_logger.warning(f"Database persistence skipped/failed: {db_exc}. Continuing diagnosis response.")


        # ── Response ───────────────────────────────────────────────────────
        return JSONResponse(content={
            "status":          "success",
            "request_id":      request_id,
            "context_hash":    ai_context.get_content_hash(),
            "is_demo_mode":    not has_real_model,
            "is_low_confidence": is_low_confidence,
            
            "diagnosis": {
                "class_idx":       class_idx,
                "disease":         predicted_disease if not is_low_confidence else "Unknown",
                "display_name":    class_to_display(predicted_disease) if not is_low_confidence else "Uncertain",
                "is_healthy":      is_healthy(predicted_disease) if not is_low_confidence else False,
                "confidence":      round(confidence * 100, 2),
                "confidence_category": confidence_category,
                "topk":            topk_predictions,
                "tta_enabled":     True,
                "low_confidence_warning": (
                    "Insufficient visual evidence for a reliable diagnosis. "
                    "Please retake the photo in better lighting or capture a closer leaf image."
                ) if is_low_confidence else None,
            },
            
            "visual_evidence": {
                "gradcam_available": True,
                "evidence_level": concern_result.get("visual_evidence_level", "Unknown"),
                "attention_indicator": lesion_ratio,
                "heatmap_b64": heatmap_b64
            },
            
            "environment": {
                "temperature": temperature,
                "humidity": humidity,
                "soil_moisture": None,
                "soil_ph": ph_level,
                "compatibility": env_evidence.get("overall_compatibility", "Insufficient Evidence"),
                "available_evidence": env_evidence.get("factors", [])
            },
            
            "growth_stage": {
                "selected_stage": growth_stage,
                "vulnerability": concern_result.get("growth_stage_vulnerability", "Unknown")
            },
            
            "concern": {
                "score": concern_result.get("score", 0),
                "level": concern_result.get("level", "Unknown"),
                "contributing_factors": concern_result.get("contributing_factors", []),
                "limiting_factors": concern_result.get("limiting_factors", []),
                "evidence_quality": concern_result.get("evidence_quality", "Good")
            },
            
            "recommendation": {
                "rag_response": root_cause_result.model_dump(mode="json"),
                "retrieved_sources": ai_context.knowledge.document_sources if getattr(ai_context, "knowledge", None) else []
            },
            
            # Legacy fields for backward compatibility
            "prediction": {
                "disease": predicted_disease,
                "confidence": round(confidence * 100, 2),
                "display_name": class_to_display(predicted_disease)
            },
            "root_cause_analysis": root_cause_result.model_dump(mode="json"),
            "gradcam_heatmap_b64": heatmap_b64,
            "ai_context": ai_context.to_dict(),
        })

    except Exception as exc:
        import traceback
        traceback.print_exc()
        error_payload = {
            "request_id":    request_id if "request_id" in dir() else "unknown",
            "user_id":       user_id,
            "request_status": f"failed: {str(exc)}",
        }
        api_logger.error(f"[MONITOR] {json.dumps(error_payload)}")
        raise HTTPException(status_code=500, detail=str(exc))

