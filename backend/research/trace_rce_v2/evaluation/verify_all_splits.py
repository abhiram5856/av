import os
import sys
import json
import hashlib

# Add project root to sys.path
sys.path.append(r"c:\Users\ABHIRAM MODUKURU\OneDrive\Desktop\AgriVision-AI")

from backend.research.trace_rce_v2.dataset.novarcd_dataset import load_split_records, create_novarcd_dataloaders

def get_context_feature_hash(rec):
    # Normalize context fields for hash computation
    ctx = rec["context"]
    # Extract only static features, excluding system timestamp or dynamic request_id
    features = {
        "user": ctx.get("user", {}),
        "image": {k: v for k, v in ctx.get("image", {}).items() if k not in ["blur_score"]}, # omit noise
        "vision": ctx.get("vision", {}),
        "gradcam": {k: v for k, v in ctx.get("gradcam", {}).items() if k not in ["s3_heatmap_url"]},
        "severity": ctx.get("severity", {}),
        "weather": ctx.get("weather", {}),
        "knowledge": ctx.get("knowledge", {}),
        "history": ctx.get("history", {})
    }
    feat_str = json.dumps(features, sort_keys=True)
    return hashlib.sha256(feat_str.encode("utf-8")).hexdigest()

def main():
    print("====================================================")
    print("NOVA-RCD Dataset Audit & Split Leakage Verification")
    print("====================================================")
    
    train_records, val_records, test_records = load_split_records()
    
    print("\n1. SAMPLE COUNTS")
    print(f"Train split size:      {len(train_records)}")
    print(f"Validation split size: {len(val_records)}")
    print(f"Test split size:       {len(test_records)}")
    print(f"Total records loaded:  {len(train_records) + len(val_records) + len(test_records)}")
    
    # Check uniquely
    train_ids = {r["case_id"] for r in train_records}
    val_ids = {r["case_id"] for r in val_records}
    test_ids = {r["case_id"] for r in test_records}
    
    train_paths = {r["image_path"] for r in train_records}
    val_paths = {r["image_path"] for r in val_records}
    test_paths = {r["image_path"] for r in test_records}
    
    train_hashes = {get_context_feature_hash(r) for r in train_records}
    val_hashes = {get_context_feature_hash(r) for r in val_records}
    test_hashes = {get_context_feature_hash(r) for r in test_records}
    
    print("\n2. EXPLICIT DUPLICATION CHECK WITHIN SPLITS")
    print(f"Train duplicate Case IDs: {len(train_records) - len(train_ids)}")
    print(f"Train duplicate Image Paths: {len(train_records) - len(train_paths)}")
    print(f"Train duplicate Context Hashes: {len(train_records) - len(train_hashes)}")
    
    print(f"Val duplicate Case IDs: {len(val_records) - len(val_ids)}")
    print(f"Val duplicate Image Paths: {len(val_records) - len(val_paths)}")
    print(f"Val duplicate Context Hashes: {len(val_records) - len(val_hashes)}")
    
    print(f"Test duplicate Case IDs: {len(test_records) - len(test_ids)}")
    print(f"Test duplicate Image Paths: {len(test_records) - len(test_paths)}")
    print(f"Test duplicate Context Hashes: {len(test_records) - len(test_hashes)}")
    
    print("\n3. OVERLAP / LEAKAGE CHECK BETWEEN SPLITS")
    print("[Case IDs]")
    print(f"  Train / Test overlap: {len(train_ids.intersection(test_ids))}")
    print(f"  Val / Test overlap:   {len(val_ids.intersection(test_ids))}")
    print(f"  Train / Val overlap:  {len(train_ids.intersection(val_ids))}")
    
    print("[Image Paths]")
    print(f"  Train / Test overlap: {len(train_paths.intersection(test_paths))}")
    print(f"  Val / Test overlap:   {len(val_paths.intersection(test_paths))}")
    print(f"  Train / Val overlap:  {len(train_paths.intersection(val_paths))}")
    
    print("[Static Context SHA256 Hashes]")
    print(f"  Train / Test overlap: {len(train_hashes.intersection(test_hashes))}")
    print(f"  Val / Test overlap:   {len(val_hashes.intersection(test_hashes))}")
    print(f"  Train / Val overlap:  {len(train_hashes.intersection(val_hashes))}")
    
    print("\n4. AUGMENTATION TIMING VERIFICATION")
    train_loader, val_loader, test_loader = create_novarcd_dataloaders(
        batch_size=32,
        augmentation_factor=2,
        seed=42
    )
    
    train_samples = train_loader.dataset._samples
    val_samples = val_loader.dataset._samples
    test_samples = test_loader.dataset._samples
    
    train_aug_count = sum(1 for s in train_samples if s.is_augmented)
    val_aug_count = sum(1 for s in val_samples if s.is_augmented)
    test_aug_count = sum(1 for s in test_samples if s.is_augmented)
    
    print(f"Train Dataloader: Total = {len(train_samples)}, Base = {len(train_samples) - train_aug_count}, Augmented = {train_aug_count}")
    print(f"Val Dataloader:   Total = {len(val_samples)}, Base = {len(val_samples) - val_aug_count}, Augmented = {val_aug_count}")
    print(f"Test Dataloader:  Total = {len(test_samples)}, Base = {len(test_samples) - test_aug_count}, Augmented = {test_aug_count}")
    
    # Confirm that validation and test have zero augmented samples
    assert val_aug_count == 0, "ERROR: Augmentation found in validation dataset!"
    assert test_aug_count == 0, "ERROR: Augmentation found in test dataset!"
    print("Augmentation Timing Check: PASSED. Augmentation is strictly applied only to the Train split at loader instantiation.")
    print("====================================================")

if __name__ == "__main__":
    main()
