import json
import os
from collections import Counter

# Lab Splits
lab_train_path = r'c:\Users\ABHIRAM MODUKURU\OneDrive\Desktop\AgriVision-AI\evaluation\clean_train_split.json'
lab_val_path = r'c:\Users\ABHIRAM MODUKURU\OneDrive\Desktop\AgriVision-AI\evaluation\clean_val_split.json'
lab_test_path = r'c:\Users\ABHIRAM MODUKURU\OneDrive\Desktop\AgriVision-AI\evaluation\clean_test_split.json'

# Field Splits
field_splits_path = r'c:\Users\ABHIRAM MODUKURU\OneDrive\Desktop\AgriVision-AI\backend\data\field_splits.json'

with open(lab_train_path) as f: lab_train = json.load(f)
with open(lab_val_path) as f: lab_val = json.load(f)
with open(lab_test_path) as f: lab_test = json.load(f)

with open(field_splits_path) as f: field_splits = json.load(f)
field_train = field_splits.get('train', [])
field_val = field_splits.get('val', [])
field_test = field_splits.get('test', [])

# From class registry
import sys
sys.path.append(r'c:\Users\ABHIRAM MODUKURU\OneDrive\Desktop\AgriVision-AI')
from backend.models.class_registry import CLASS_NAMES
active_classes = set(CLASS_NAMES)

# Filter field splits to active classes
field_train = [x for x in field_train if os.path.basename(os.path.dirname(x)) in active_classes]
field_val = [x for x in field_val if os.path.basename(os.path.dirname(x)) in active_classes]
field_test = [x for x in field_test if os.path.basename(os.path.dirname(x)) in active_classes]

print("=== DATA SPLIT VERIFICATION ===")
print(f"Lab Train: {len(lab_train)}")
print(f"Lab Val: {len(lab_val)}")
print(f"Lab Test: {len(lab_test)}")
print(f"Field Train: {len(field_train)}")
print(f"Field Val: {len(field_val)}")
print(f"Field Test: {len(field_test)}")

print("\n=== PER CLASS FIELD COUNTS (TRAIN/VAL) ===")
field_train_classes = [os.path.basename(os.path.dirname(x)) for x in field_train]
field_val_classes = [os.path.basename(os.path.dirname(x)) for x in field_val]
train_counter = Counter(field_train_classes)
val_counter = Counter(field_val_classes)

for cls in sorted(active_classes):
    tc = train_counter.get(cls, 0)
    vc = val_counter.get(cls, 0)
    if tc > 0 or vc > 0:
        print(f"{cls:40s} Train: {tc:<5} Val: {vc:<5}")

print("\n=== LEAKAGE CHECK ===")
train_val_set = set(lab_train + lab_val + field_train + field_val)
field_test_set = set(field_test)
lab_test_set = set(lab_test)

field_leak = train_val_set.intersection(field_test_set)
lab_leak = train_val_set.intersection(lab_test_set)

print(f"Field Test Leakage (Overlap with Train/Val): {len(field_leak)}")
print(f"Lab Test Leakage (Overlap with Train/Val): {len(lab_leak)}")
