import re

with open(r'backend\api\diagnose.py', 'r', encoding='utf-8') as f:
    content = f.read()

tta_block = '''tta_transforms = [
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

'''

content = content.replace('def _build_mobilenet_v3_small', tta_block + 'def _build_mobilenet_v3_small')

with open(r'backend\api\diagnose.py', 'w', encoding='utf-8') as f:
    f.write(content)
