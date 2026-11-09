"""
BharatPotHole Preprocessing Module
Applies CLAHE contrast enhancement for depth feature consistency,
verifies YOLO labels, and formats the dataset into:
data/processed/vision/bharat_pothole/
"""

import os
import glob
import cv2
import shutil

DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "data"))
INPUT_BASE = os.path.join(DATA_DIR, "combined", "BharatPotHole", "BharatPotHole")
OUTPUT_BASE = os.path.join(DATA_DIR, "processed", "vision", "bharat_pothole")

def apply_clahe(image):
    """Enhance pothole depth contrast using CLAHE on luminance."""
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    enhanced = cv2.merge((cl, a, b))
    return cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)

def process_bharat_pothole(max_train=1500, max_val=300, max_test=300):
    """
    Process train/val/test splits with CLAHE enhancement.
    Caps can be adjusted to balance dataset size.
    """
    splits = {
        'train': ('train', max_train),
        'val': ('valid', max_val),
        'test': ('test', max_test)
    }
    
    print(f"Starting BharatPotHole preprocessing from: {INPUT_BASE}")
    counts = {}
    
    for split_target, (split_src, limit) in splits.items():
        src_img_dir = os.path.join(INPUT_BASE, split_src, "images")
        src_lbl_dir = os.path.join(INPUT_BASE, split_src, "labels")
        
        dst_img_dir = os.path.join(OUTPUT_BASE, "images", split_target)
        dst_lbl_dir = os.path.join(OUTPUT_BASE, "labels", split_target)
        os.makedirs(dst_img_dir, exist_ok=True)
        os.makedirs(dst_lbl_dir, exist_ok=True)
        
        img_files = sorted(glob.glob(os.path.join(src_img_dir, "*.jpg")))
        if limit:
            img_files = img_files[:limit]
            
        print(f"Processing split '{split_target}': {len(img_files)} images...")
        processed_count = 0
        
        for img_path in img_files:
            base_name = os.path.splitext(os.path.basename(img_path))[0]
            lbl_path = os.path.join(src_lbl_dir, f"{base_name}.txt")
            
            if not os.path.exists(lbl_path):
                continue
                
            img = cv2.imread(img_path)
            if img is None:
                continue
                
            enhanced = apply_clahe(img)
            out_img = os.path.join(dst_img_dir, f"{base_name}.jpg")
            out_lbl = os.path.join(dst_lbl_dir, f"{base_name}.txt")
            
            cv2.imwrite(out_img, enhanced)
            shutil.copyfile(lbl_path, out_lbl)
            processed_count += 1
            
        counts[split_target] = processed_count
        print(f"Split '{split_target}' completed: {processed_count} images.")
        
    yaml_content = f"""# BharatPotHole YOLO Configuration
path: {OUTPUT_BASE}
train: images/train
val: images/val
test: images/test

nc: 1
names: ['pothole']
"""
    yaml_path = os.path.join(OUTPUT_BASE, "data.yaml")
    with open(yaml_path, "w") as f:
        f.write(yaml_content)
        
    print(f"BharatPotHole Preprocessing Complete: {counts}")
    print(f"YAML config saved to: {yaml_path}")

if __name__ == "__main__":
    process_bharat_pothole()

