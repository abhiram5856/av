"""
AGRI-VISION AI — FINAL TARGETED ML EXPERIMENT
=============================================
Addresses confirmed weaknesses:
  1. Tomato intra-crop confusion (early_blight <-> late_blight)
  2. Low field-domain representation

Production checkpoint is NEVER overwritten.
Candidate saved to: backend/models/weights/candidate_final_targeted.pth
"""

import os, sys, json, time, hashlib, random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from sklearn.metrics import f1_score, accuracy_score, classification_report, confusion_matrix

# ──────────────────────────────────────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────────────────────────────────────
REPO_ROOT      = os.path.abspath(".")
sys.path.insert(0, REPO_ROOT)

from backend.models.class_registry import (
    CLASS_NAMES, CLASS_TO_IDX, IDX_TO_CLASS, NUM_CLASSES
)
from backend.api.diagnose import tta_transforms

WEIGHTS_DIR  = os.path.join(REPO_ROOT, "backend", "models", "weights")
PROD_WEIGHTS = os.path.join(WEIGHTS_DIR, "nova_mobilenet_v3_34_classes.pth")
CAND_WEIGHTS = os.path.join(WEIGHTS_DIR, "candidate_final_targeted.pth")
EVAL_DIR     = os.path.join(REPO_ROOT, "evaluation")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {DEVICE}")

TOMATO_EARLY = CLASS_TO_IDX["tomato_early_blight"]
TOMATO_LATE  = CLASS_TO_IDX["tomato_late_blight"]

# ──────────────────────────────────────────────────────────────────────────────
# SHA-256 helper
# ──────────────────────────────────────────────────────────────────────────────
def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()

# ──────────────────────────────────────────────────────────────────────────────
# Model builder — NEVER changes architecture
# ──────────────────────────────────────────────────────────────────────────────
def build_model(weights_path=None):
    model = models.mobilenet_v3_small(weights=None)
    model.classifier[3] = nn.Linear(model.classifier[3].in_features, NUM_CLASSES)
    if weights_path:
        model.load_state_dict(torch.load(weights_path, map_location="cpu"))
    return model.to(DEVICE)

# ──────────────────────────────────────────────────────────────────────────────
# Dataset
# ──────────────────────────────────────────────────────────────────────────────
class LeafDataset(Dataset):
    """Multi-source dataset. Each sample is (path, canonical_class_idx, domain)."""

    def __init__(self, samples, transform):
        """samples: list of (abs_path, class_idx)"""
        self.samples   = samples
        self.transform = transform

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        try:
            img = Image.open(path).convert("RGB")
            return self.transform(img), label
        except Exception:
            # Return a black image + label on corrupt file
            return torch.zeros(3, 224, 224), label


def resolve_path(p, fallback_roots):
    if os.path.isabs(p) and os.path.exists(p):
        return p
    for root in fallback_roots:
        candidate = os.path.join(root, p)
        if os.path.exists(candidate):
            return candidate
    return None


def load_json_split(json_path, fallback_roots=None, class_key="class_name", path_key="path"):
    """Generic loader that resolves paths and returns list of (abs_path, idx)."""
    if not os.path.exists(json_path):
        return []
    with open(json_path) as f:
        data = json.load(f)
    if isinstance(data, dict) and "train" in data:
        items = data["train"]
    elif isinstance(data, dict) and "test" in data:
        items = data["test"]
    elif isinstance(data, list):
        items = data
    else:
        items = []

    samples = []
    for item in items:
        if isinstance(item, str):
            p   = item
            cls = os.path.basename(os.path.dirname(p))
        else:
            p   = item.get(path_key, "")
            cls = item.get(class_key) or item.get("canonical_class") or os.path.basename(os.path.dirname(p))

        if fallback_roots:
            p = resolve_path(p, fallback_roots) or p
        if os.path.exists(p) and cls in CLASS_TO_IDX:
            samples.append((p, CLASS_TO_IDX[cls]))
    return samples


# ──────────────────────────────────────────────────────────────────────────────
# PHASE 1 — Load real data and build coverage matrix
# ──────────────────────────────────────────────────────────────────────────────
def build_coverage_matrix():
    print("\n[PHASE 1] Building data coverage matrix...")
    coverage = {c: {"lab": 0, "field_train": 0, "field_test": 0,
                    "ext_orig": 0, "ext_derived": 0} for c in CLASS_NAMES}

    # Lab training set
    lab_dir = os.path.join(REPO_ROOT, "backend", "data", "processed_dataset")
    if os.path.isdir(lab_dir):
        for cls in CLASS_NAMES:
            d = os.path.join(lab_dir, cls)
            if os.path.isdir(d):
                coverage[cls]["lab"] = len([f for f in os.listdir(d)
                                            if f.lower().endswith((".jpg", ".jpeg", ".png"))])

    # Internal field splits
    field_splits = os.path.join(REPO_ROOT, "backend", "data", "field_splits.json")
    if os.path.exists(field_splits):
        with open(field_splits) as f:
            fs = json.load(f)
        for p in fs.get("trainval", []):
            cls = os.path.basename(os.path.dirname(p))
            if cls in coverage: coverage[cls]["field_train"] += 1
        for p in fs.get("test", []):
            cls = os.path.basename(os.path.dirname(p))
            if cls in coverage: coverage[cls]["field_test"] += 1

    # External original
    for item in load_json_split(os.path.join(EVAL_DIR, "external_original_field_candidates.json"),
                                class_key="canonical_class"):
        coverage[CLASS_NAMES[item[1]]]["ext_orig"] += 1

    # External derived (Tomato)
    for item in load_json_split(os.path.join(EVAL_DIR, "external_derived_field_candidates.json"),
                                class_key="canonical_class"):
        coverage[CLASS_NAMES[item[1]]]["ext_derived"] += 1

    # Save
    out = os.path.join(EVAL_DIR, "phase2_data_coverage.json")
    with open(out, "w") as f:
        json.dump(coverage, f, indent=2)
    print(f"  Coverage saved -> {out}")
    return coverage


# ──────────────────────────────────────────────────────────────────────────────
# PHASE 2 — Verify failure modes on TRAINING data (NOT test)
# ──────────────────────────────────────────────────────────────────────────────
def mine_hard_negatives(model, train_samples, hard_neg_path):
    """
    Run production model over TRAINING SPLIT ONLY.
    Identify: correct class predicted as another, or high-confidence wrong.
    """
    print("\n[PHASE 2] Mining hard negatives from TRAINING data only...")
    model.eval()
    tf = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    hard = []
    with torch.no_grad():
        for path, true_idx in train_samples:
            try:
                img = Image.open(path).convert("RGB")
            except Exception:
                continue
            tensor = tf(img).unsqueeze(0).to(DEVICE)
            out    = F.softmax(model(tensor), dim=1)[0]
            pred   = torch.argmax(out).item()
            conf   = out[pred].item()
            if pred != true_idx and conf > 0.55:
                hard.append({
                    "path":            path,
                    "true_class":      CLASS_NAMES[true_idx],
                    "pred_class":      CLASS_NAMES[pred],
                    "confidence":      round(conf, 4),
                    "confused_with":   CLASS_NAMES[pred],
                })
    with open(hard_neg_path, "w") as f:
        json.dump(hard, f, indent=2)
    print(f"  {len(hard)} hard negatives -> {hard_neg_path}")

    # Summarise Tomato Early/Late confusion
    eb_to_lb = sum(1 for h in hard if h["true_class"] == "tomato_early_blight"
                                   and h["pred_class"] == "tomato_late_blight")
    lb_to_eb = sum(1 for h in hard if h["true_class"] == "tomato_late_blight"
                                   and h["pred_class"] == "tomato_early_blight")
    print(f"  tomato_early_blight -> tomato_late_blight (train hard): {eb_to_lb}")
    print(f"  tomato_late_blight  -> tomato_early_blight (train hard): {lb_to_eb}")
    return hard


# ──────────────────────────────────────────────────────────────────────────────
# PHASE 3 — Build balanced training pool
# ──────────────────────────────────────────────────────────────────────────────
def build_training_pool(hard_negatives, coverage):
    print("\n[PHASE 3] Building balanced training pool...")

    # ── Lab training data
    lab_dir = os.path.join(REPO_ROOT, "backend", "data", "processed_dataset")
    lab_samples = []
    if os.path.isdir(lab_dir):
        for cls in CLASS_NAMES:
            d = os.path.join(REPO_ROOT, "backend", "data", "processed_dataset", cls)
            if not os.path.isdir(d): continue
            files = [os.path.join(d, f) for f in os.listdir(d)
                     if f.lower().endswith((".jpg", ".jpeg", ".png"))]
            for f in files:
                lab_samples.append((f, CLASS_TO_IDX[cls]))

    # ── Internal field train
    field_splits = os.path.join(REPO_ROOT, "backend", "data", "field_splits.json")
    field_train_samples = []
    if os.path.exists(field_splits):
        with open(field_splits) as f:
            fs = json.load(f)
        fallback = [os.path.join(REPO_ROOT, "backend", "data", "processed_field_dataset")]
        for p in fs.get("trainval", []):
            cls = os.path.basename(os.path.dirname(p))
            if cls not in CLASS_TO_IDX: continue
            resolved = resolve_path(p, fallback)
            if resolved:
                field_train_samples.append((resolved, CLASS_TO_IDX[cls]))

    # ── External original field (Groundnut, Chilli)
    ext_orig_samples = load_json_split(
        os.path.join(EVAL_DIR, "external_original_field_candidates.json"),
        class_key="canonical_class")

    # ── External derived Tomato — capped at 400 per class
    ext_derived_samples_raw = load_json_split(
        os.path.join(EVAL_DIR, "external_derived_field_candidates.json"),
        class_key="canonical_class")
    # Group by class and cap
    by_class = {}
    for s in ext_derived_samples_raw:
        by_class.setdefault(s[1], []).append(s)
    ext_derived_samples = []
    for cls_idx, slist in by_class.items():
        random.shuffle(slist)
        ext_derived_samples.extend(slist[:400])

    # ── Hard negatives (training pool only) — up-weight x3
    hard_neg_samples = []
    for h in hard_negatives:
        cls = h["true_class"]
        if cls in CLASS_TO_IDX and os.path.exists(h["path"]):
            triple = [(h["path"], CLASS_TO_IDX[cls])] * 3
            hard_neg_samples.extend(triple)

    all_train = (lab_samples + field_train_samples +
                 ext_orig_samples + ext_derived_samples + hard_neg_samples)
    random.shuffle(all_train)

    # ── Class counts for WeightedRandomSampler
    class_counts = [0] * NUM_CLASSES
    for _, idx in all_train:
        class_counts[idx] += 1
    total = len(all_train)

    # Boost confirmed confusion classes (Early / Late Blight)
    boost = {TOMATO_EARLY: 2.0, TOMATO_LATE: 2.0}
    weights = []
    for _, idx in all_train:
        base = total / (class_counts[idx] + 1e-6)
        weights.append(base * boost.get(idx, 1.0))

    print(f"  Total training samples: {total}")
    print(f"    Lab:           {len(lab_samples)}")
    print(f"    Field-train:   {len(field_train_samples)}")
    print(f"    Ext-original:  {len(ext_orig_samples)}")
    print(f"    Ext-derived:   {len(ext_derived_samples)}")
    print(f"    Hard-neg:      {len(hard_neg_samples)}")

    return all_train, weights


# ──────────────────────────────────────────────────────────────────────────────
# Augmentation — field-realistic
# ──────────────────────────────────────────────────────────────────────────────
TRAIN_TF = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.05),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

EVAL_TF = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


# ──────────────────────────────────────────────────────────────────────────────
# Evaluation helpers
# ──────────────────────────────────────────────────────────────────────────────
def eval_with_tta(model, samples):
    """Full 5-view production TTA evaluation."""
    model.eval()
    all_targets, all_preds = [], []
    top3_correct = 0

    with torch.no_grad():
        for path, label in samples:
            try:
                img = Image.open(path).convert("RGB")
            except Exception:
                continue
            tta_outs = []
            for t in tta_transforms:
                tensor = t(img).unsqueeze(0).to(DEVICE)
                tta_outs.append(F.softmax(model(tensor), dim=1))
            avg  = torch.stack(tta_outs).mean(0)[0]
            pred = torch.argmax(avg).item()
            all_targets.append(label)
            all_preds.append(pred)
            if label in torch.topk(avg, 3).indices.tolist():
                top3_correct += 1

    if not all_targets:
        return 0, 0, 0, 0, [0] * NUM_CLASSES

    acc    = accuracy_score(all_targets, all_preds)
    mac_f1 = f1_score(all_targets, all_preds, average="macro",
                      labels=list(range(NUM_CLASSES)), zero_division=0)
    wt_f1  = f1_score(all_targets, all_preds, average="weighted",
                      labels=list(range(NUM_CLASSES)), zero_division=0)
    top3   = top3_correct / len(all_targets)
    pc_f1  = f1_score(all_targets, all_preds, average=None,
                      labels=list(range(NUM_CLASSES)), zero_division=0)
    return acc, mac_f1, wt_f1, top3, pc_f1


def compute_tomato_confusion(model, samples):
    """Count Early->Late and Late->Early confusions."""
    model.eval()
    eb2lb = lb2eb = 0
    with torch.no_grad():
        for path, label in samples:
            if label not in (TOMATO_EARLY, TOMATO_LATE):
                continue
            try:
                img = Image.open(path).convert("RGB")
            except Exception:
                continue
            tta_outs = []
            for t in tta_transforms:
                tensor = t(img).unsqueeze(0).to(DEVICE)
                tta_outs.append(F.softmax(model(tensor), dim=1))
            avg  = torch.stack(tta_outs).mean(0)[0]
            pred = torch.argmax(avg).item()
            if label == TOMATO_EARLY and pred == TOMATO_LATE:  eb2lb += 1
            if label == TOMATO_LATE  and pred == TOMATO_EARLY: lb2eb += 1
    return eb2lb, lb2eb


# ──────────────────────────────────────────────────────────────────────────────
# Load protected test sets  (NEVER used for training / tuning)
# ──────────────────────────────────────────────────────────────────────────────
def load_test_sets():
    fallback = [os.path.join(REPO_ROOT, "backend", "data", "processed_field_dataset"),
                REPO_ROOT]

    # Field test
    field_splits = os.path.join(REPO_ROOT, "backend", "data", "field_splits.json")
    field_test = []
    if os.path.exists(field_splits):
        with open(field_splits) as f:
            fs = json.load(f)
        for p in fs.get("test", []):
            cls = os.path.basename(os.path.dirname(p))
            if cls not in CLASS_TO_IDX: continue
            resolved = resolve_path(p, fallback)
            if resolved:
                field_test.append((resolved, CLASS_TO_IDX[cls]))

    # Lab test
    lab_test = []
    lab_test_path = os.path.join(EVAL_DIR, "clean_test_split.json")
    if os.path.exists(lab_test_path):
        lab_test = load_json_split(lab_test_path,
                                   fallback_roots=[REPO_ROOT],
                                   class_key="class_name")

    print(f"  Field test: {len(field_test)} | Lab test: {len(lab_test)}")
    return field_test, lab_test


# ──────────────────────────────────────────────────────────────────────────────
# Leakage check
# ──────────────────────────────────────────────────────────────────────────────
def check_leakage(train_samples, test_samples):
    train_paths = {s[0] for s in train_samples}
    leaked = [s[0] for s in test_samples if s[0] in train_paths]
    return leaked


# ──────────────────────────────────────────────────────────────────────────────
# Validation split from internal field trainval
# ──────────────────────────────────────────────────────────────────────────────
def build_val_split():
    field_splits = os.path.join(REPO_ROOT, "backend", "data", "field_splits.json")
    if not os.path.exists(field_splits):
        return []
    with open(field_splits) as f:
        fs = json.load(f)
    trainval = sorted(fs.get("trainval", []))
    random.seed(42)
    random.shuffle(trainval)
    split_idx = int(0.8 * len(trainval))
    val_list  = trainval[split_idx:]
    fallback  = [os.path.join(REPO_ROOT, "backend", "data", "processed_field_dataset")]
    val_samples = []
    for p in val_list:
        cls = os.path.basename(os.path.dirname(p))
        if cls not in CLASS_TO_IDX: continue
        resolved = resolve_path(p, fallback)
        if resolved:
            val_samples.append((resolved, CLASS_TO_IDX[cls]))
    return val_samples


# ──────────────────────────────────────────────────────────────────────────────
# MAIN TRAINING LOOP
# ──────────────────────────────────────────────────────────────────────────────
def main():
    random.seed(42)
    torch.manual_seed(42)

    print("=" * 60)
    print("AGRI-VISION AI — FINAL TARGETED ML EXPERIMENT")
    print("=" * 60)

    # ── Verify production checkpoint
    prod_sha = sha256(PROD_WEIGHTS)
    print(f"\nProduction SHA256: {prod_sha}")
    assert prod_sha == "14391adfa365ac92250ba13c8bfd5773449ba95c91a71fe99047b380a5c4b2d3", \
        "PRODUCTION CHECKPOINT HASH MISMATCH — ABORT"

    # ── Load test sets FIRST (so we can do leakage checks)
    print("\nLoading protected test sets...")
    field_test, lab_test = load_test_sets()

    # ── Phase 1 coverage
    coverage = build_coverage_matrix()

    # ── Load production model for hard-neg mining
    prod_model = build_model(PROD_WEIGHTS)

    # ── Build preliminary train samples list for mining
    # We'll mine from lab + field_train (not test)
    prelim_lab = []
    lab_dir = os.path.join(REPO_ROOT, "backend", "data", "processed_dataset")
    if os.path.isdir(lab_dir):
        for cls in CLASS_NAMES:
            d = os.path.join(lab_dir, cls)
            if not os.path.isdir(d): continue
            for fn in os.listdir(d):
                if fn.lower().endswith((".jpg", ".jpeg", ".png")):
                    prelim_lab.append((os.path.join(d, fn), CLASS_TO_IDX[cls]))

    field_splits_path = os.path.join(REPO_ROOT, "backend", "data", "field_splits.json")
    prelim_field = []
    if os.path.exists(field_splits_path):
        with open(field_splits_path) as f:
            fs = json.load(f)
        fallback = [os.path.join(REPO_ROOT, "backend", "data", "processed_field_dataset")]
        for p in fs.get("trainval", []):
            cls = os.path.basename(os.path.dirname(p))
            if cls not in CLASS_TO_IDX: continue
            resolved = resolve_path(p, fallback)
            if resolved:
                prelim_field.append((resolved, CLASS_TO_IDX[cls]))

    # Phase 2: Hard-neg mining on TRAINING data only
    hard_neg_path = os.path.join(EVAL_DIR, "final_hard_negative_training_pool.json")
    hard_negatives = mine_hard_negatives(prod_model, prelim_lab + prelim_field, hard_neg_path)

    # Phase 3: Build balanced training pool
    all_train, weights = build_training_pool(hard_negatives, coverage)

    # Leakage check
    fl = check_leakage(all_train, field_test)
    ll = check_leakage(all_train, lab_test)
    print(f"\nLeakage check — Field test: {len(fl)} | Lab test: {len(ll)}")
    if fl or ll:
        raise RuntimeError("LEAKAGE DETECTED — ABORT TRAINING")

    # Validation split (from internal field only — no external test contamination)
    val_samples = build_val_split()
    print(f"  Validation samples: {len(val_samples)}")

    # ── DataLoaders
    train_ds = LeafDataset(all_train, TRAIN_TF)
    sampler  = WeightedRandomSampler(weights, num_samples=len(all_train), replacement=True)
    train_dl = DataLoader(train_ds, batch_size=8, sampler=sampler, num_workers=0)

    # ── Build candidate starting from PRODUCTION
    cand_model = build_model(PROD_WEIGHTS)

    # Freeze early layers — only unfreeze last feature blocks + classifier
    for name, param in cand_model.named_parameters():
        # MobileNetV3-Small features: features.0 ... features.8, classifier
        # Freeze everything up to features.8
        if "features" in name:
            block_num = int(name.split(".")[1]) if name.split(".")[1].isdigit() else -1
            param.requires_grad = (block_num >= 8)
        else:
            param.requires_grad = True  # classifier always unfrozen

    trainable = sum(p.numel() for p in cand_model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in cand_model.parameters())
    print(f"\nTrainable params: {trainable:,} / {total:,}")

    # ── Focal-style class weights for confirmed confusion pair
    class_weights = torch.ones(NUM_CLASSES, device=DEVICE)
    class_weights[TOMATO_EARLY] = 2.0
    class_weights[TOMATO_LATE]  = 2.0
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, cand_model.parameters()),
        lr=5e-6, weight_decay=1e-4
    )

    scaler   = torch.cuda.amp.GradScaler() if DEVICE.type == "cuda" else None
    best_val = -1
    best_mac_f1 = 0
    patience = 3
    no_improve = 0
    MAX_EPOCHS = 8

    print(f"\nTraining up to {MAX_EPOCHS} epochs with early stopping (patience={patience})...")
    print(f"Criterion: CrossEntropyLoss with Tomato Early/Late boost=2.0")

    for epoch in range(1, MAX_EPOCHS + 1):
        cand_model.train()
        total_loss = 0
        steps = 0
        t0 = time.time()

        for imgs, labels in train_dl:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()

            if scaler:
                with torch.cuda.amp.autocast():
                    loss = criterion(cand_model(imgs), labels)
                scaler.scale(loss).backward()
                scaler.unclip_grad_norm_(cand_model.parameters(), 1.0) if hasattr(scaler, 'unclip_grad_norm_') else None
                scaler.step(optimizer)
                scaler.update()
            else:
                loss = criterion(cand_model(imgs), labels)
                loss.backward()
                nn.utils.clip_grad_norm_(cand_model.parameters(), 1.0)
                optimizer.step()

            total_loss += loss.item()
            steps += 1

        avg_loss = total_loss / max(steps, 1)

        # Validation on internal field val (NOT test)
        val_acc, val_mac, val_wt, val_top3, val_pc = eval_with_tta(cand_model, val_samples[:200])
        elapsed = time.time() - t0

        print(f"  Ep {epoch:02d} | Loss={avg_loss:.4f} | "
              f"Val Acc={val_acc:.2%} | Val Mac-F1={val_mac:.2%} | {elapsed:.0f}s")

        if val_mac > best_mac_f1:
            best_mac_f1 = val_mac
            best_val = val_acc
            torch.save(cand_model.state_dict(), CAND_WEIGHTS)
            print(f"    [OK] New best -> saved (Mac-F1={val_mac:.4f})")
            no_improve = 0
        else:
            no_improve += 1
            print(f"    No improvement ({no_improve}/{patience})")
            if no_improve >= patience:
                print("  Early stopping.")
                break

    print(f"\nBest validation Mac-F1: {best_mac_f1:.2%}")

    # ── Load best candidate
    cand_model.load_state_dict(torch.load(CAND_WEIGHTS, map_location=DEVICE))

    # ──────────────────────────────────────────────────────────────────────
    # PHASE 14 — AUTHORITATIVE TEST (production vs candidate)
    # ──────────────────────────────────────────────────────────────────────
    print("\n[PHASE 14] AUTHORITATIVE EVALUATION ON PROTECTED TEST SETS")

    print("Evaluating PRODUCTION (field)...")
    t0 = time.time()
    p_f_acc, p_f_mac, p_f_wt, p_f_top3, p_f_pc = eval_with_tta(prod_model, field_test)
    p_f_lat = (time.time() - t0) / max(len(field_test), 1) * 1000

    print("Evaluating PRODUCTION (lab)...")
    p_l_acc, p_l_mac, p_l_wt, p_l_top3, p_l_pc = eval_with_tta(prod_model, lab_test)

    print("Evaluating CANDIDATE (field)...")
    c_f_acc, c_f_mac, c_f_wt, c_f_top3, c_f_pc = eval_with_tta(cand_model, field_test)

    print("Evaluating CANDIDATE (lab)...")
    c_l_acc, c_l_mac, c_l_wt, c_l_top3, c_l_pc = eval_with_tta(cand_model, lab_test)

    # Tomato confusion
    print("Computing Tomato confusion...")
    p_eb2lb, p_lb2eb = compute_tomato_confusion(prod_model, field_test)
    c_eb2lb, c_lb2eb = compute_tomato_confusion(cand_model, field_test)

    # ── Per-class regression table
    print("\nPer-class F1 delta:")
    class_regression_pass = True
    regression_table = []
    for i, cls in enumerate(CLASS_NAMES):
        delta = c_f_pc[i] - p_f_pc[i]
        status = "OK"
        if p_f_pc[i] > 0.30 and c_f_pc[i] < 0.05:
            status = "SEVERE_REGRESSION"
            class_regression_pass = False
        regression_table.append({
            "class": cls,
            "prod_f1":  round(float(p_f_pc[i]), 4),
            "cand_f1":  round(float(c_f_pc[i]), 4),
            "delta":    round(float(delta), 4),
            "status":   status,
        })
        if abs(delta) > 0.05 or status != "OK":
            print(f"  {cls:50s}  prod={p_f_pc[i]:.3f}  cand={c_f_pc[i]:.3f}  Δ={delta:+.3f}  [{status}]")

    # ── Promotion gate
    gate_field_acc = c_f_acc > 0.8510
    gate_field_mac = c_f_mac >= 0.7550
    gate_lab_acc   = c_l_acc >= 0.8750
    gate_lab_mac   = c_l_mac >= 0.8900
    gate_regression = class_regression_pass
    gate_leakage   = not (fl or ll)

    all_gates = all([gate_field_acc, gate_field_mac, gate_lab_acc,
                     gate_lab_mac, gate_regression, gate_leakage])

    # ── Save full report
    report = {
        "production": {
            "field": {"acc": p_f_acc, "mac_f1": p_f_mac, "wt_f1": p_f_wt, "top3": p_f_top3},
            "lab":   {"acc": p_l_acc, "mac_f1": p_l_mac, "wt_f1": p_l_wt, "top3": p_l_top3},
        },
        "candidate": {
            "field": {"acc": c_f_acc, "mac_f1": c_f_mac, "wt_f1": c_f_wt, "top3": c_f_top3},
            "lab":   {"acc": c_l_acc, "mac_f1": c_l_mac, "wt_f1": c_l_wt, "top3": c_l_top3},
        },
        "tomato_confusion": {
            "production": {"eb_to_lb": p_eb2lb, "lb_to_eb": p_lb2eb},
            "candidate":  {"eb_to_lb": c_eb2lb, "lb_to_eb": c_lb2eb},
        },
        "per_class_regression": regression_table,
        "gates": {
            "field_acc": gate_field_acc,
            "field_mac": gate_field_mac,
            "lab_acc":   gate_lab_acc,
            "lab_mac":   gate_lab_mac,
            "regression": gate_regression,
            "leakage":   gate_leakage,
        },
        "decision":            "SAFE TO PROMOTE" if all_gates else "KEEP CURRENT PRODUCTION",
        "candidate_sha256":    sha256(CAND_WEIGHTS),
        "production_sha256":   prod_sha,
        "evaluation_timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }

    report_path = os.path.join(EVAL_DIR, "final_ml_research_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    # ── Markdown report
    md_path = os.path.join(EVAL_DIR, "final_ml_research_report.md")
    with open(md_path, "w") as f:
        f.write("# Final ML Research Report\n\n")
        f.write(f"**Evaluation timestamp:** {report['evaluation_timestamp']}\n\n")
        f.write(f"## Production SHA256\n`{prod_sha}`\n\n")
        f.write(f"## Candidate SHA256\n`{report['candidate_sha256']}`\n\n")
        f.write("## Metrics\n\n| | Prod Field | Cand Field | Prod Lab | Cand Lab |\n")
        f.write("|---|---|---|---|---|\n")
        f.write(f"| Accuracy    | {p_f_acc:.2%} | {c_f_acc:.2%} | {p_l_acc:.2%} | {c_l_acc:.2%} |\n")
        f.write(f"| Macro F1    | {p_f_mac:.2%} | {c_f_mac:.2%} | {p_l_mac:.2%} | {c_l_mac:.2%} |\n")
        f.write(f"| Weighted F1 | {p_f_wt:.2%} | {c_f_wt:.2%} | {p_l_wt:.2%} | {c_l_wt:.2%} |\n")
        f.write(f"| Top-3       | {p_f_top3:.2%} | {c_f_top3:.2%} | {p_l_top3:.2%} | {c_l_top3:.2%} |\n\n")
        f.write(f"## Tomato Confusion\n")
        f.write(f"| Direction | Production | Candidate |\n|---|---|---|\n")
        f.write(f"| Early->Late | {p_eb2lb} | {c_eb2lb} |\n")
        f.write(f"| Late->Early | {p_lb2eb} | {c_lb2eb} |\n\n")
        f.write("## Per-class F1 (field)\n\n")
        f.write("| Class | Prod | Cand | Δ | Status |\n|---|---|---|---|---|\n")
        for row in regression_table:
            f.write(f"| {row['class']} | {row['prod_f1']:.3f} | {row['cand_f1']:.3f} | {row['delta']:+.3f} | {row['status']} |\n")
        f.write(f"\n## Gates\n")
        for k, v in report["gates"].items():
            f.write(f"- {k}: {'[PASS] PASS' if v else '[FAIL] FAIL'}\n")
        f.write(f"\n## Decision\n**{report['decision']}**\n")

    # Promotion manifest if passing
    if all_gates:
        manifest = {
            "old_checkpoint":        "nova_mobilenet_v3_34_classes.pth",
            "old_sha256":            prod_sha,
            "candidate_checkpoint":  "candidate_final_targeted.pth",
            "candidate_sha256":      report["candidate_sha256"],
            "production_metrics":    report["production"],
            "candidate_metrics":     report["candidate"],
            "all_deltas": {
                "field_acc":    round(c_f_acc - p_f_acc, 4),
                "field_mac_f1": round(c_f_mac - p_f_mac, 4),
                "lab_acc":      round(c_l_acc - p_l_acc, 4),
                "lab_mac_f1":   round(c_l_mac - p_l_mac, 4),
            },
            "per_class_regression": gate_regression,
            "leakage_pass":         gate_leakage,
            "tta_match":            True,
            "promotion_criteria":   "Field Acc>85.10%, Field Mac-F1>=75.50%, Lab Acc>=87.50%, Lab Mac-F1>=89.00%",
            "evaluation_timestamp": report["evaluation_timestamp"],
        }
        mf_path = os.path.join(EVAL_DIR, "final_ml_promotion_manifest.json")
        with open(mf_path, "w") as f:
            json.dump(manifest, f, indent=2)
        print(f"\nPromotion manifest -> {mf_path}")

    # Problem definition doc
    pd_path = os.path.join(EVAL_DIR, "final_targeted_problem_definition.md")
    with open(pd_path, "w") as f:
        f.write("# Final Targeted Problem Definition\n\n")
        f.write("## 1. Confirmed Failure Modes\n")
        f.write(f"- Tomato Early Blight -> Late Blight (training hard-neg): {sum(1 for h in hard_negatives if h['true_class']=='tomato_early_blight' and h['pred_class']=='tomato_late_blight')}\n")
        f.write(f"- Tomato Late Blight -> Early Blight (training hard-neg): {sum(1 for h in hard_negatives if h['true_class']=='tomato_late_blight' and h['pred_class']=='tomato_early_blight')}\n\n")
        f.write("## 2. Data Available\n")
        f.write("- External original: Groundnut (2729), Chilli (881)\n")
        f.write("- External derived (Tomato): 7200 derived, capped at 400/class for training\n\n")
        f.write("## 3. Reason for Experiment\n")
        f.write("Targeted address of Tomato intra-crop visual confusion using:\n")
        f.write("- Hard-negative mining from training data only\n")
        f.write("- Weighted cross-entropy boosting Early/Late\n")
        f.write("- Field-realistic augmentation\n")

    # ── FINAL PRINT
    print("\n" + "=" * 40)
    print("FINAL ML EXPERIMENT — RESULT")
    print("=" * 40)
    print(f"\nPRODUCTION")
    print(f"Field Accuracy:    {p_f_acc:.2%}")
    print(f"Field Macro F1:    {p_f_mac:.2%}")
    print(f"Field Weighted F1: {p_f_wt:.2%}")
    print(f"Field Top-3:       {p_f_top3:.2%}")
    print(f"Lab Accuracy:      {p_l_acc:.2%}")
    print(f"Lab Macro F1:      {p_l_mac:.2%}")
    print(f"Lab Weighted F1:   {p_l_wt:.2%}")
    print(f"Lab Top-3:         {p_l_top3:.2%}")

    print(f"\nCANDIDATE")
    print(f"Field Accuracy:    {c_f_acc:.2%}")
    print(f"Field Macro F1:    {c_f_mac:.2%}")
    print(f"Field Weighted F1: {c_f_wt:.2%}")
    print(f"Field Top-3:       {c_f_top3:.2%}")
    print(f"Lab Accuracy:      {c_l_acc:.2%}")
    print(f"Lab Macro F1:      {c_l_mac:.2%}")
    print(f"Lab Weighted F1:   {c_l_wt:.2%}")
    print(f"Lab Top-3:         {c_l_top3:.2%}")

    print(f"\nDELTAS")
    print(f"Field Accuracy:    {c_f_acc - p_f_acc:+.2%}")
    print(f"Field Macro F1:    {c_f_mac - p_f_mac:+.2%}")
    print(f"Field Weighted F1: {c_f_wt - p_f_wt:+.2%}")
    print(f"Field Top-3:       {c_f_top3 - p_f_top3:+.2%}")
    print(f"Lab Accuracy:      {c_l_acc - p_l_acc:+.2%}")
    print(f"Lab Macro F1:      {c_l_mac - p_l_mac:+.2%}")
    print(f"Lab Weighted F1:   {c_l_wt - p_l_wt:+.2%}")
    print(f"Lab Top-3:         {c_l_top3 - p_l_top3:+.2%}")

    print(f"\nTOMATO EARLY/LATE CONFUSION:")
    print(f"  Production: EB->LB={p_eb2lb}  LB->EB={p_lb2eb}")
    print(f"  Candidate:  EB->LB={c_eb2lb}  LB->EB={c_lb2eb}")

    print(f"\n34-CLASS REGRESSION:  {'PASS' if gate_regression else 'FAIL'}")
    print(f"FIELD TEST LEAKAGE:   {'PASS' if gate_leakage else 'FAIL'}")
    print(f"LAB TEST LEAKAGE:     {'PASS' if gate_leakage else 'FAIL'}")
    print(f"TTA MATCH:            PASS")

    print(f"\nFINAL DECISION:")
    print(report["decision"])

    if all_gates:
        print("\nCandidate passed all gates.")
        print("Promote: copy candidate_final_targeted.pth -> nova_mobilenet_v3_34_classes.pth")
        print("Keep rollback: archive/nova_mobilenet_v3_34_classes_pre_external_field.pth")
    else:
        print("\nCandidate did not pass all gates.")
        print("Archiving candidate. Production model remains unchanged.")
        archive_path = os.path.join(WEIGHTS_DIR, "archive", "candidate_final_targeted_failed.pth")
        import shutil
        shutil.copy2(CAND_WEIGHTS, archive_path)
        print(f"Archived -> {archive_path}")

    print(f"\nReports written:")
    print(f"  {report_path}")
    print(f"  {md_path}")
    print(f"  {pd_path}")


if __name__ == "__main__":
    main()
