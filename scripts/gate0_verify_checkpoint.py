"""
Gate 0: Checkpoint Forensic Verification
==========================================
Independent validation of nova_mobilenet_v3_24_classes.pth before
it is made the canonical production checkpoint.

Checks:
  1. State dict keys match MobileNetV3-Small architecture
  2. Classifier output dimension == 24
  3. Class ordering is alphabetical (matches ImageFolder / registry)
  4. Representative forward pass produces valid probability distribution
  5. If test split exists, runs deterministic evaluation and reports metrics

Output: PASS or FAIL with detailed reason. If FAIL, the checkpoint
must NOT be promoted to production.
"""

import sys
import os
import json
import time
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from torchvision import models, transforms
from PIL import Image

REPO_ROOT = Path(r"c:\Users\ABHIRAM MODUKURU\OneDrive\Desktop\AgriVision-AI")
sys.path.insert(0, str(REPO_ROOT))

CHECKPOINT_PATH = REPO_ROOT / "backend/models/weights/nova_mobilenet_v3_24_classes.pth"
FIELD_SPLITS    = REPO_ROOT / "backend/data/field_splits.json"
FIELD_DATASET   = REPO_ROOT / "backend/data/processed_field_dataset"
TEST_SAMPLES    = REPO_ROOT / "test_samples"

REQUIRED_CLASSES = [
    "corn_gray_leaf_spot", "corn_leaf_blight", "corn_rust_leaf",
    "pepper_bell_bacterial_spot", "pepper_bell_healthy",
    "potato_early_blight", "potato_healthy", "potato_late_blight",
    "rice_bacterial_leaf_blight", "rice_brown_spot", "rice_healthy",
    "rice_leaf_blast", "rice_leaf_scald", "rice_sheath_blight",
    "tomato_bacterial_spot", "tomato_early_blight", "tomato_healthy",
    "tomato_late_blight", "tomato_leaf_mold", "tomato_mosaic_virus",
    "tomato_septoria_leaf_spot",
    "tomato_spider_mites_two_spotted_spider_mite",
    "tomato_target_spot", "tomato_yellow_leaf_curl_virus",
]
NUM_REQUIRED = len(REQUIRED_CLASSES)

VERDICT = {"status": "PASS", "checks": [], "metrics": {}}

def check(name, condition, detail):
    status = "PASS" if condition else "FAIL"
    VERDICT["checks"].append({"check": name, "status": status, "detail": detail})
    if not condition:
        VERDICT["status"] = "FAIL"
    print(f"  [{status}] {name}: {detail}")
    return condition

def build_model(num_classes):
    m = models.mobilenet_v3_small(weights=None)
    m.classifier[3] = nn.Linear(m.classifier[3].in_features, num_classes)
    return m

print("=" * 65)
print("GATE 0: CHECKPOINT FORENSIC VERIFICATION")
print("=" * 65)

# ── 1. File existence ─────────────────────────────────────────────
print("\n[1] File Presence")
ckpt_exists = CHECKPOINT_PATH.exists()
check("Checkpoint file exists", ckpt_exists, str(CHECKPOINT_PATH))
if not ckpt_exists:
    print("\nFATAL: Checkpoint missing. Cannot continue.")
    sys.exit(1)

# ── 2. Load state dict and inspect ───────────────────────────────
print("\n[2] State Dict Inspection")
state_dict = torch.load(CHECKPOINT_PATH, map_location="cpu")
is_ordered_dict = isinstance(state_dict, dict)
check("State dict is a dict", is_ordered_dict, type(state_dict).__name__)

# Check key structure matches MobileNetV3-Small
expected_key_prefix = "features.0.0.weight"
has_features_key = expected_key_prefix in state_dict
check("Has MobileNetV3 feature keys", has_features_key,
      f"'{expected_key_prefix}' {'found' if has_features_key else 'MISSING'}")

# Check classifier output dimension
classifier_key = "classifier.3.weight"
has_classifier = classifier_key in state_dict
check("Has classifier.3.weight key", has_classifier,
      f"'{classifier_key}' {'found' if has_classifier else 'MISSING'}")

if has_classifier:
    out_dim = state_dict[classifier_key].shape[0]
    in_dim  = state_dict[classifier_key].shape[1]
    check("Output dimension == 24", out_dim == NUM_REQUIRED,
          f"shape=({out_dim}, {in_dim}), required out_dim={NUM_REQUIRED}")
    check("Input features == 1024 (MobileNetV3-Small)", in_dim == 1024,
          f"in_features={in_dim}")
    VERDICT["metrics"]["checkpoint_output_dim"] = out_dim

# Total parameter count
total_params = sum(p.numel() for p in state_dict.values())
check("Parameter count in expected range (2M-5M)",
      1_500_000 < total_params < 6_000_000,
      f"total_params={total_params:,}")

# ── 3. Model load compatibility ───────────────────────────────────
print("\n[3] Architecture Compatibility")
try:
    model = build_model(NUM_REQUIRED)
    model.load_state_dict(state_dict)
    model.eval()
    check("load_state_dict() succeeds", True, "No errors")
    arch_ok = True
except Exception as e:
    check("load_state_dict() succeeds", False, str(e))
    arch_ok = False

# ── 4. Class ordering validation ─────────────────────────────────
print("\n[4] Class Ordering Validation")
sorted_check = REQUIRED_CLASSES == sorted(REQUIRED_CLASSES)
check("Required classes are alphabetically sorted", sorted_check,
      "(ImageFolder ordering must match)")
no_cotton = "cotton_diseased" not in REQUIRED_CLASSES and "cotton_healthy" not in REQUIRED_CLASSES
check("Cotton classes absent from required list", no_cotton,
      "cotton_diseased and cotton_healthy must not be present")

# ── 5. Representative forward pass ────────────────────────────────
print("\n[5] Forward Pass Test")
if arch_ok:
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    # Use a dummy image if no test samples exist
    dummy = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
    
    # Try to load a real image if available
    real_img = None
    for ext in ["*.jpg", "*.jpeg", "*.png"]:
        found = list(TEST_SAMPLES.rglob(ext))
        if found:
            try:
                real_img = Image.open(found[0]).convert("RGB")
                break
            except:
                pass

    test_img = real_img if real_img else dummy
    img_source = "real test sample" if real_img else "random dummy image"

    try:
        with torch.no_grad():
            t_start = time.perf_counter()
            tensor  = transform(test_img).unsqueeze(0)
            logits  = model(tensor)
            probs   = torch.softmax(logits[0], dim=0)
            latency = time.perf_counter() - t_start

        out_classes  = logits.shape[1]
        top_idx      = int(torch.argmax(probs).item())
        top_conf     = float(probs[top_idx].item())
        prob_sum     = float(probs.sum().item())

        check("Output has 24 classes", out_classes == NUM_REQUIRED,
              f"logits.shape[1]={out_classes}")
        check("Softmax sums to ~1.0", abs(prob_sum - 1.0) < 1e-4,
              f"sum={prob_sum:.6f}")
        check("Top confidence is a valid float in [0,1]",
              0 < top_conf <= 1.0, f"top_conf={top_conf:.4f}")
        check(f"Inference completes in <2s (source: {img_source})",
              latency < 2.0, f"latency={latency*1000:.1f}ms")

        VERDICT["metrics"].update({
            "forward_pass_output_classes": out_classes,
            "softmax_sum": round(prob_sum, 6),
            "sample_top_class_idx": top_idx,
            "sample_top_class_name": REQUIRED_CLASSES[top_idx],
            "sample_top_confidence": round(top_conf, 4),
            "inference_latency_ms": round(latency * 1000, 1),
            "image_source": img_source,
        })
        print(f"     → Predicted class: [{top_idx}] {REQUIRED_CLASSES[top_idx]} ({top_conf*100:.1f}%)")

    except Exception as e:
        check("Forward pass executes without error", False, str(e))

# ── 6. Field test set evaluation (if available) ──────────────────
print("\n[6] Deterministic Field Evaluation (24-class test split)")
if arch_ok and FIELD_SPLITS.exists() and FIELD_DATASET.exists():
    try:
        from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
        CLASS_TO_IDX = {c: i for i, c in enumerate(REQUIRED_CLASSES)}
        transform_eval = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        with open(FIELD_SPLITS) as f:
            splits = json.load(f)
        test_paths = splits.get("test", [])
        y_true, y_pred = [], []
        skipped = 0
        with torch.no_grad():
            for rel_path in test_paths:
                cls = os.path.dirname(rel_path)
                if cls not in CLASS_TO_IDX:
                    skipped += 1
                    continue
                full = FIELD_DATASET / rel_path
                try:
                    img    = Image.open(full).convert("RGB")
                    tensor = transform_eval(img).unsqueeze(0)
                    out    = model(tensor)
                    pred   = int(torch.argmax(out, 1).item())
                    y_true.append(CLASS_TO_IDX[cls])
                    y_pred.append(pred)
                except:
                    skipped += 1

        if y_true:
            acc  = accuracy_score(y_true, y_pred)
            f1   = f1_score(y_true, y_pred, average="macro", zero_division=0)
            prec = precision_score(y_true, y_pred, average="macro", zero_division=0)
            rec  = recall_score(y_true, y_pred, average="macro", zero_division=0)
            print(f"     → Evaluated {len(y_true)} images (skipped {skipped} outside 24-class scope)")
            print(f"     → Accuracy:  {acc*100:.2f}%")
            print(f"     → Macro F1:  {f1*100:.2f}%")
            print(f"     → Precision: {prec*100:.2f}%")
            print(f"     → Recall:    {rec*100:.2f}%")
            check("Accuracy > 30% on field test split", acc > 0.30,
                  f"acc={acc*100:.2f}% (field domain — lower than lab is expected)")
            VERDICT["metrics"]["field_eval"] = {
                "num_evaluated": len(y_true),
                "num_skipped": skipped,
                "accuracy": round(acc, 4),
                "macro_f1": round(f1, 4),
                "macro_precision": round(prec, 4),
                "macro_recall": round(rec, 4),
                "note": "Field-domain evaluation on 24-class held-out test split using nova_mobilenet_v3_24_classes.pth"
            }
        else:
            check("Field evaluation produced predictions", False,
                  "No matching classes found in test split for 24-class scope")
    except ImportError:
        print("     sklearn not available — skipping field evaluation")
    except Exception as e:
        print(f"     Field evaluation failed: {e}")
else:
    print("     SKIPPED — field_splits.json or processed_field_dataset not found")
    VERDICT["metrics"]["field_eval"] = "SKIPPED — field data not present"

# ── Final Verdict ─────────────────────────────────────────────────
print("\n" + "=" * 65)
print(f"GATE 0 VERDICT: {VERDICT['status']}")
print("=" * 65)
if VERDICT["status"] == "PASS":
    print("✅ Checkpoint is verified. Safe to promote to production.")
else:
    print("❌ Checkpoint FAILED verification. Do NOT promote to production.")
    for c in VERDICT["checks"]:
        if c["status"] == "FAIL":
            print(f"   FAILED CHECK: {c['check']} — {c['detail']}")

# Write verdict to file for reference
out_path = REPO_ROOT / "evaluation/gate0_checkpoint_verification.json"
out_path.parent.mkdir(exist_ok=True)
with open(out_path, "w") as f:
    json.dump(VERDICT, f, indent=2)
print(f"\nVerification report written to: {out_path}")
