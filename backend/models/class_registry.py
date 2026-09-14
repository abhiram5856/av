"""
AgriVision AI — Centralised Class Registry
==========================================
SINGLE SOURCE OF TRUTH for all class definitions.

Every component MUST import CLASS_NAMES and related constants from here.
Never hardcode class names, indices, or counts elsewhere.

PROJECT SCOPE: Telangana-Focused Agricultural AI Prototype
  Supported crops: Rice, Tomato, Corn, Pepper, Potato
  Active classes:  24

Architecture:   MobileNetV3-Small
Checkpoint:     backend/models/weights/nova_mobilenet_v3_24_classes.pth
                (Verified by Gate 0 — output_dim=24, MobileNetV3-Small compatible)
Num classes:    24
Dataset source: PlantVillage (preprocessed; laboratory-style segmented images)
Training date:  2026-07-25

Gate 0 Verification Results (2026-08-26):
  Field zero-shot accuracy: 33.06% on 732 held-out images (expected — lab-to-field gap)
  Full report: evaluation/gate0_checkpoint_verification.json

Legacy 26-class checkpoints (with Cotton):
  ARCHIVED to: backend/models/weights/archive/
  Do NOT load these for production inference.

IMPORTANT: Class ordering here MUST match the ordering used by
torchvision.datasets.ImageFolder when it loaded the training data.
That ordering is alphabetical (case-insensitive) on the folder names.
The 24 class names below are sorted alphabetically — matching ImageFolder.

DO NOT reorder this list. The index position IS the class label.
"""

from typing import Dict, List, Optional

# ──────────────────────────────────────────────────────────────────────────────
# Canonical class list (alphabetical, matching ImageFolder discovery order)
# ──────────────────────────────────────────────────────────────────────────────

CLASS_NAMES: List[str] = [
    "chilli_healthy",
    "chilli_leaf_curl",
    "chilli_leaf_spot",
    "corn_gray_leaf_spot",
    "corn_leaf_blight",
    "corn_rust_leaf",
    "cotton_bacterial_blight",
    "cotton_healthy",
    "cotton_leaf_curl_virus",
    "groundnut_early_leaf_spot",
    "groundnut_healthy",
    "groundnut_late_leaf_spot",
    "groundnut_rust",
    "pepper_bell_bacterial_spot",
    "pepper_bell_healthy",
    "potato_early_blight",
    "potato_healthy",
    "potato_late_blight",
    "rice_bacterial_leaf_blight",
    "rice_brown_spot",
    "rice_healthy",
    "rice_leaf_blast",
    "rice_leaf_scald",
    "rice_sheath_blight",
    "tomato_bacterial_spot",
    "tomato_early_blight",
    "tomato_healthy",
    "tomato_late_blight",
    "tomato_leaf_mold",
    "tomato_mosaic_virus",
    "tomato_septoria_leaf_spot",
    "tomato_spider_mites_two_spotted_spider_mite",
    "tomato_target_spot",
    "tomato_yellow_leaf_curl_virus",
]

NUM_CLASSES: int = len(CLASS_NAMES)  # 34

# ──────────────────────────────────────────────────────────────────────────────
# Quick lookup maps
# ──────────────────────────────────────────────────────────────────────────────

CLASS_TO_IDX: Dict[str, int] = {name: idx for idx, name in enumerate(CLASS_NAMES)}
IDX_TO_CLASS: Dict[int, str] = {idx: name for idx, name in enumerate(CLASS_NAMES)}

# ──────────────────────────────────────────────────────────────────────────────
# Human-readable display names (for UI and reports)
# ──────────────────────────────────────────────────────────────────────────────

DISPLAY_NAMES: Dict[str, str] = {
    "chilli_healthy": "Chilli Healthy",
    "chilli_leaf_curl": "Chilli Leaf Curl",
    "chilli_leaf_spot": "Chilli Leaf Spot",
    "corn_gray_leaf_spot": "Corn Gray Leaf Spot",
    "corn_leaf_blight": "Corn Leaf Blight",
    "corn_rust_leaf": "Corn Rust Leaf",
    "cotton_bacterial_blight": "Cotton Bacterial Blight",
    "cotton_healthy": "Cotton Healthy",
    "cotton_leaf_curl_virus": "Cotton Leaf Curl Virus",
    "groundnut_early_leaf_spot": "Groundnut Early Leaf Spot",
    "groundnut_healthy": "Groundnut Healthy",
    "groundnut_late_leaf_spot": "Groundnut Late Leaf Spot",
    "groundnut_rust": "Groundnut Rust",
    "pepper_bell_bacterial_spot": "Pepper Bell Bacterial Spot",
    "pepper_bell_healthy": "Pepper Bell Healthy",
    "potato_early_blight": "Potato Early Blight",
    "potato_healthy": "Potato Healthy",
    "potato_late_blight": "Potato Late Blight",
    "rice_bacterial_leaf_blight": "Rice Bacterial Leaf Blight",
    "rice_brown_spot": "Rice Brown Spot",
    "rice_healthy": "Rice Healthy",
    "rice_leaf_blast": "Rice Leaf Blast",
    "rice_leaf_scald": "Rice Leaf Scald",
    "rice_sheath_blight": "Rice Sheath Blight",
    "tomato_bacterial_spot": "Tomato Bacterial Spot",
    "tomato_early_blight": "Tomato Early Blight",
    "tomato_healthy": "Tomato Healthy",
    "tomato_late_blight": "Tomato Late Blight",
    "tomato_leaf_mold": "Tomato Leaf Mold",
    "tomato_mosaic_virus": "Tomato Mosaic Virus",
    "tomato_septoria_leaf_spot": "Tomato Septoria Leaf Spot",
    "tomato_spider_mites_two_spotted_spider_mite": "Tomato Spider Mites Two Spotted Spider Mite",
    "tomato_target_spot": "Tomato Target Spot",
    "tomato_yellow_leaf_curl_virus": "Tomato Yellow Leaf Curl Virus",
}

# ──────────────────────────────────────────────────────────────────────────────
# Crop and disease-type metadata (for Evidence Consistency Engine)
# ──────────────────────────────────────────────────────────────────────────────

# True = this class is a disease; False = healthy leaf
IS_DISEASE: Dict[str, bool] = {
    "chilli_healthy": False,
    "chilli_leaf_curl": True,
    "chilli_leaf_spot": True,
    "corn_gray_leaf_spot": True,
    "corn_leaf_blight": True,
    "corn_rust_leaf": True,
    "cotton_bacterial_blight": True,
    "cotton_healthy": False,
    "cotton_leaf_curl_virus": True,
    "groundnut_early_leaf_spot": True,
    "groundnut_healthy": False,
    "groundnut_late_leaf_spot": True,
    "groundnut_rust": True,
    "pepper_bell_bacterial_spot": True,
    "pepper_bell_healthy": False,
    "potato_early_blight": True,
    "potato_healthy": False,
    "potato_late_blight": True,
    "rice_bacterial_leaf_blight": True,
    "rice_brown_spot": True,
    "rice_healthy": False,
    "rice_leaf_blast": True,
    "rice_leaf_scald": True,
    "rice_sheath_blight": True,
    "tomato_bacterial_spot": True,
    "tomato_early_blight": True,
    "tomato_healthy": False,
    "tomato_late_blight": True,
    "tomato_leaf_mold": True,
    "tomato_mosaic_virus": True,
    "tomato_septoria_leaf_spot": True,
    "tomato_spider_mites_two_spotted_spider_mite": True,
    "tomato_target_spot": True,
    "tomato_yellow_leaf_curl_virus": True,
}

CROP_FAMILY: Dict[str, str] = {
    "chilli_healthy": "chilli",
    "chilli_leaf_curl": "chilli",
    "chilli_leaf_spot": "chilli",
    "corn_gray_leaf_spot": "corn",
    "corn_leaf_blight": "corn",
    "corn_rust_leaf": "corn",
    "cotton_bacterial_blight": "cotton",
    "cotton_healthy": "cotton",
    "cotton_leaf_curl_virus": "cotton",
    "groundnut_early_leaf_spot": "groundnut",
    "groundnut_healthy": "groundnut",
    "groundnut_late_leaf_spot": "groundnut",
    "groundnut_rust": "groundnut",
    "pepper_bell_bacterial_spot": "pepper",
    "pepper_bell_healthy": "pepper",
    "potato_early_blight": "potato",
    "potato_healthy": "potato",
    "potato_late_blight": "potato",
    "rice_bacterial_leaf_blight": "rice",
    "rice_brown_spot": "rice",
    "rice_healthy": "rice",
    "rice_leaf_blast": "rice",
    "rice_leaf_scald": "rice",
    "rice_sheath_blight": "rice",
    "tomato_bacterial_spot": "tomato",
    "tomato_early_blight": "tomato",
    "tomato_healthy": "tomato",
    "tomato_late_blight": "tomato",
    "tomato_leaf_mold": "tomato",
    "tomato_mosaic_virus": "tomato",
    "tomato_septoria_leaf_spot": "tomato",
    "tomato_spider_mites_two_spotted_spider_mite": "tomato",
    "tomato_target_spot": "tomato",
    "tomato_yellow_leaf_curl_virus": "tomato",
}

# ──────────────────────────────────────────────────────────────────────────────
# Model configuration — single source of truth
# ──────────────────────────────────────────────────────────────────────────────

MODEL_CONFIG = {
    "architecture":        "mobilenet_v3_small",
    "num_classes":         NUM_CLASSES,
    "input_size":          (224, 224),
    "normalize_mean":      [0.485, 0.456, 0.406],
    "normalize_std":       [0.229, 0.224, 0.225],
    "checkpoint_filename": "nova_mobilenet_v3_34_classes.pth",
    "dataset":             "PlantVillage-Preprocessed-v1.0",
    "dataset_note": (
        "PlantVillage images are laboratory-segmented (white/black backgrounds). "
        "Real-world field accuracy is expected to be lower than benchmark accuracy."
    ),
}


# ──────────────────────────────────────────────────────────────────────────────
# Utility functions
# ──────────────────────────────────────────────────────────────────────────────

def idx_to_class(idx: int) -> str:
    """Return class name for a given integer index. Raises KeyError if out of range."""
    if idx not in IDX_TO_CLASS:
        raise KeyError(
            f"Class index {idx} is out of range [0, {NUM_CLASSES - 1}]. "
            "Possible architecture/checkpoint mismatch."
        )
    return IDX_TO_CLASS[idx]


def class_to_display(class_name: str) -> str:
    """Return human-readable display name. Falls back to class_name if not found."""
    return DISPLAY_NAMES.get(class_name, class_name.replace("_", " ").title())


def is_healthy(class_name: str) -> bool:
    """Returns True if the class represents a healthy leaf."""
    return not IS_DISEASE.get(class_name, True)


def validate_registry() -> None:
    """
    Sanity-check the registry at import time.
    Raises AssertionError if internal consistency is broken.
    Called automatically on module load.
    """
    assert len(CLASS_NAMES) == NUM_CLASSES, "NUM_CLASSES mismatch"
    assert len(CLASS_TO_IDX) == NUM_CLASSES, "CLASS_TO_IDX mismatch"
    assert len(IDX_TO_CLASS) == NUM_CLASSES, "IDX_TO_CLASS mismatch"
    assert len(DISPLAY_NAMES) == NUM_CLASSES, "DISPLAY_NAMES mismatch"
    assert len(IS_DISEASE) == NUM_CLASSES, "IS_DISEASE mismatch"
    assert len(CROP_FAMILY) == NUM_CLASSES, "CROP_FAMILY mismatch"

    # Verify round-trip consistency
    for idx, name in enumerate(CLASS_NAMES):
        assert CLASS_TO_IDX[name] == idx, f"CLASS_TO_IDX[{name}] != {idx}"
        assert IDX_TO_CLASS[idx] == name, f"IDX_TO_CLASS[{idx}] != {name}"

    # Verify alphabetical order (must match ImageFolder)
    assert CLASS_NAMES == sorted(CLASS_NAMES), (
        "CLASS_NAMES is not alphabetically sorted. "
        "This WILL cause a class index mismatch with ImageFolder."
    )


# Run validation at import time — fails loudly if registry is broken
validate_registry()
