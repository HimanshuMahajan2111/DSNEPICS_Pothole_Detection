"""
Road Surface Categorized Images Preprocessing Module
Extracts, CLAHE-enhances, and standardizes categorized images into:
data/processed/vision/road_surface_classes/
"""

import os
import glob
import cv2
import numpy as np

DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "data"))
INPUT_BASE = os.path.join(DATA_DIR, "combined", "ROAD SURFACE", "Road Surface Image")
OUTPUT_BASE = os.path.join(DATA_DIR, "processed", "vision", "road_surface_classes")

def apply_clahe(image):
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    return cv2.cvtColor(cv2.merge((cl, a, b)), cv2.COLOR_LAB2BGR)

def resize_letterbox(image, target_size=(640, 640)):
    h, w = image.shape[:2]
    th, tw = target_size
    scale = min(tw / w, th / h)
    nw, nh = int(w * scale), int(h * scale)
    resized = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((th, tw, 3), 114, dtype=np.uint8)
    top = (th - nh) // 2
    left = (tw - nw) // 2
    canvas[top:top+nh, left:left+nw] = resized
    return canvas

def process_road_surface_classes(max_per_category=100):
    import numpy as np
    os.makedirs(OUTPUT_BASE, exist_ok=True)
    
    categories = {
        'paved_rainy_pothole': os.path.join(INPUT_BASE, "Paved Road", "Pothole in Rainy Season"),
        'paved_summer_pothole': os.path.join(INPUT_BASE, "Paved Road", "Pothole in Summer"),
        'paved_speed_breaker': os.path.join(INPUT_BASE, "Paved Road", "Speed braker_n_Roadside barrier"),
        'unpaved_rainy_pothole': os.path.join(INPUT_BASE, "Unpaved Road", "Pothole in Rainy"),
        'unpaved_summer_pothole': os.path.join(INPUT_BASE, "Unpaved Road", "Pothole in Summer")
    }
    
    summary = {}
    for cat_name, cat_dir in categories.items():
        dst_dir = os.path.join(OUTPUT_BASE, cat_name)
        os.makedirs(dst_dir, exist_ok=True)
        
        img_files = glob.glob(os.path.join(cat_dir, "**", "*.jpg"), recursive=True) + glob.glob(os.path.join(cat_dir, "**", "*.png"), recursive=True)
        img_files = [f for f in img_files if "rotated" not in f.lower()][:max_per_category]
        
        count = 0
        for f in img_files:
            img = cv2.imread(f)
            if img is None:
                continue
            enh = apply_clahe(img)
            lb = resize_letterbox(enh)
            cv2.imwrite(os.path.join(dst_dir, f"img_{count:04d}.jpg"), lb)
            count += 1
            
        summary[cat_name] = count
        print(f"Processed {count} images for category '{cat_name}'")
        
    print(f"Road Surface Categorized Images Complete: {summary}")

if __name__ == "__main__":
    process_road_surface_classes()
