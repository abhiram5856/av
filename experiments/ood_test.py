import sys
import os
import cv2
import numpy as np
from PIL import Image

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(REPO_ROOT)

from backend.services.image_quality_gate import ImageQualityGate

def test_ood():
    gate = ImageQualityGate()
    
    # 1. Clear sky (Blue)
    img_sky = Image.new('RGB', (800, 800), color=(135, 206, 235))
    
    # 2. Dirt/Soil (Dark brown/black)
    img_dirt = Image.new('RGB', (800, 800), color=(60, 40, 20))
    
    # 3. Person/Skin tone
    img_skin = Image.new('RGB', (800, 800), color=(255, 205, 148))
    
    # 4. White wall
    img_wall = Image.new('RGB', (800, 800), color=(250, 250, 250))
    
    # 5. Night sky (Black)
    img_night = Image.new('RGB', (800, 800), color=(10, 10, 15))
    
    # 6. Actually a plant (Green)
    img_plant = Image.new('RGB', (800, 800), color=(34, 139, 34))

    tests = {
        "Clear Sky": img_sky,
        "Dark Soil": img_dirt,
        "Skin Tone": img_skin,
        "White Wall": img_wall,
        "Night Sky": img_night,
        "Synthetic Plant": img_plant
    }
    
    print("OOD Rejection Tests:")
    for name, img in tests.items():
        res = gate.assess(img)
        print(f"{name}: allow={res['allow_analysis']}, status={res['status']}")

if __name__ == '__main__':
    test_ood()
