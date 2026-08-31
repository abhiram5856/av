import os
import sys

# Add root directory to sys path
sys.path.append(r"c:\Users\ABHIRAM MODUKURU\OneDrive\Desktop\AgriVision-AI")

from backend.research.trace_rce_v2.dataset.novarcd_dataset import load_split_records

def main():
    print("========================================")
    print("NOVA-RCD: Data Leakage Verification")
    print("========================================")
    
    train_records, val_records, test_records = load_split_records()
    
    train_case_ids = {r["case_id"] for r in train_records}
    val_case_ids = {r["case_id"] for r in val_records}
    test_case_ids = {r["case_id"] for r in test_records}
    
    train_image_paths = {r["image_path"] for r in train_records}
    val_image_paths = {r["image_path"] for r in val_records}
    test_image_paths = {r["image_path"] for r in test_records}
    
    print("\n[Case IDs]")
    train_test_overlap = train_case_ids.intersection(test_case_ids)
    val_test_overlap = val_case_ids.intersection(test_case_ids)
    train_val_overlap = train_case_ids.intersection(val_case_ids)
    
    print(f"Train/Test Intersection: {len(train_test_overlap)}")
    print(f"Val/Test Intersection:   {len(val_test_overlap)}")
    print(f"Train/Val Intersection:  {len(train_val_overlap)}")
    
    print("\n[Image Paths]")
    train_test_img_overlap = train_image_paths.intersection(test_image_paths)
    val_test_img_overlap = val_image_paths.intersection(test_image_paths)
    train_val_img_overlap = train_image_paths.intersection(val_image_paths)
    
    print(f"Train/Test Intersection: {len(train_test_img_overlap)}")
    print(f"Val/Test Intersection:   {len(val_test_img_overlap)}")
    print(f"Train/Val Intersection:  {len(train_val_img_overlap)}")
    
    assert len(train_test_overlap) == 0, "DATA LEAKAGE: Case IDs overlap between Train and Test!"
    assert len(val_test_overlap) == 0, "DATA LEAKAGE: Case IDs overlap between Val and Test!"
    assert len(train_val_overlap) == 0, "DATA LEAKAGE: Case IDs overlap between Train and Val!"
    
    assert len(train_test_img_overlap) == 0, "DATA LEAKAGE: Image paths overlap between Train and Test!"
    assert len(val_test_img_overlap) == 0, "DATA LEAKAGE: Image paths overlap between Val and Test!"
    assert len(train_val_img_overlap) == 0, "DATA LEAKAGE: Image paths overlap between Train and Val!"
    
    print("\n========================================")
    print("VERIFICATION SUCCESS: No data leakage detected.")
    print("========================================")

if __name__ == "__main__":
    main()
