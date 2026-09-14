import json

with open('evaluation/final_field_metrics.json', 'r') as f:
    d = json.load(f)
    print('FIELD ACCURACY:', d.get('accuracy'))
    print('FIELD TOP3:', d.get('top3_accuracy'))
    print('FIELD MACRO F1:', d.get('macro_f1'))
    print('FIELD SAMPLES:', d.get('total_samples'))

with open('evaluation/dataset_audit_report.json', 'r') as f:
    d = json.load(f)
    print('\nTOTAL LAB IMAGES:', d.get('total_images'))
    print('LAB SPLITS:', d.get('splits'))
