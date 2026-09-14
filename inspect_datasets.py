import json
import os

files = [
    'evaluation/final_field_metrics.json',
    'evaluation/final_34_class_metrics.json',
    'backend/data/processed_dataset/dataset_audit.json',
    'evaluation/dataset_audit_report.json'
]

for f in files:
    if os.path.exists(f):
        print(f'\n--- {f} ---')
        with open(f, 'r') as fp:
            data = json.load(fp)
            if 'Accuracy' in data:
                print('Accuracy:', data['Accuracy'])
                print('Macro F1:', data.get('Macro F1', data.get('Macro_F1')))
                print('Top-3 Accuracy:', data.get('Top-3 Accuracy', data.get('Top3_Accuracy')))
            else:
                for k in list(data.keys())[:10]:
                    print(k, type(data[k]))
