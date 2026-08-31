"""
Image Quality Gate for AgriVision AI
====================================
Lightweight heuristic to check image blur, brightness, and resolution.
Does NOT do strict OOD detection. It provides warnings and optional rejections.
"""

import cv2
import numpy as np
from PIL import Image
from typing import Dict, Any

class ImageQualityGate:
    def __init__(self, 
                 min_blur_var=100, 
                 min_brightness=40, 
                 max_brightness=220,
                 min_resolution=(100, 100),
                 max_aspect_ratio=3.0):
        self.min_blur = min_blur_var
        self.min_bright = min_brightness
        self.max_bright = max_brightness
        self.min_res = min_resolution
        self.max_ar = max_aspect_ratio
        
    def assess(self, image: Image.Image) -> Dict[str, Any]:
        """
        Assess PIL image for basic quality heuristics.
        """
        warnings = []
        allow_analysis = True
        status = "acceptable"
        
        # Resolution & Aspect Ratio
        w, h = image.size
        if w < self.min_res[0] or h < self.min_res[1]:
            warnings.append(f"Low resolution ({w}x{h}). Minimum {self.min_res[0]}x{self.min_res[1]} recommended.")
            status = "poor"
            
        ar = max(w/h, h/w)
        if ar > self.max_ar:
            warnings.append(f"Extreme aspect ratio ({ar:.1f}:1). Make sure the leaf is centered.")
            status = "poor"
            
        # Convert to CV2 for Blur & Brightness
        cv_img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
        
        # Blur check (Laplacian variance)
        blur_val = cv2.Laplacian(gray, cv2.CV_64F).var()
        if blur_val < self.min_blur:
            warnings.append(f"Image appears blurry. Please hold camera steady.")
            status = "poor"
            
        # Brightness check
        mean_brightness = np.mean(gray)
        if mean_brightness < self.min_bright:
            warnings.append("Image is very dark. Please capture in better lighting.")
            status = "poor"
        elif mean_brightness > self.max_bright:
            warnings.append("Image is overexposed/too bright. Avoid direct glare.")
            status = "poor"
            
        return {
            "status": status,
            "allow_analysis": allow_analysis,
            "warnings": warnings,
            "metrics": {
                "blur_variance": round(blur_val, 1),
                "brightness": round(mean_brightness, 1),
                "resolution": f"{w}x{h}",
                "aspect_ratio": round(ar, 2)
            }
        }
