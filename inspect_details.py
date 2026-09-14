import json

with open('evaluation/final_field_metrics.json', 'r') as f:
    d = json.load(f)
    per_class = d.get('per_class', {})
    f1s = [(cls, metrics.get('f1-score', 0)) for cls, metrics in per_class.items() if isinstance(metrics, dict) and 'f1-score' in metrics]
    f1s.sort(key=lambda x: x[1])
    print('WORST FIELD CLASSES:')
    for cls, f1 in f1s[:5]: print(f"{cls}: {f1:.3f}")
    print('\nBEST FIELD CLASSES:')
    for cls, f1 in f1s[-5:]: print(f"{cls}: {f1:.3f}")

with open('backend/data/processed_dataset/dataset_audit.json', 'r') as f:
    d = json.load(f)
    print('\nDATASET AUDIT:')
    print('Corrupted:', len(d.get('corrupted_files', [])))
    print('Near Duplicates:', len(d.get('near_duplicate_groups', [])))
