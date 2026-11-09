"""
Vision Data Cleaning Module
Clamps out-of-boundary bounding boxes, removes sub-pixel annotation noise,
balances negative background frame ratios, and verifies image decodability.
"""

import os
import glob
import shutil
import cv2
import numpy as np

DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "data"))
VISION_BASE = os.path.join(DATA_DIR, "processed", "vision")

def clamp_and_filter_bbox(line):
    """
    Clamps box coordinates strictly to [0.0, 1.0] and discards sub-pixel boxes.
    Returns cleaned YOLO line string, or None if degenerate.
    """
    parts = line.strip().split()
    if len(parts) < 5:
        return None
        
    cls_id = parts[0]
    cx, cy, w, h = map(float, parts[1:5])
    
    # Calculate box bounds
    x1 = cx - w / 2.0
    x2 = cx + w / 2.0
    y1 = cy - h / 2.0
    y2 = cy + h / 2.0
    
    # Clamp to [0.0, 1.0]
    x1_clamped = max(0.0, min(1.0, x1))
    x2_clamped = max(0.0, min(1.0, x2))
    y1_clamped = max(0.0, min(1.0, y1))
    y2_clamped = max(0.0, min(1.0, y2))
    
    new_w = x2_clamped - x1_clamped
    new_h = y2_clamped - y1_clamped
    
    # Drop sub-pixel artifacts (< 0.005 corresponds to < 3.2 pixels on 640x640)
    if new_w < 0.005 or new_h < 0.005:
        return None
        
    new_cx = x1_clamped + new_w / 2.0
    new_cy = y1_clamped + new_h / 2.0
    
    return f"{cls_id} {new_cx:.6f} {new_cy:.6f} {new_w:.6f} {new_h:.6f}"

def clean_vision_annotations(dataset_name="bharat_pothole"):
    """
    Scans all YOLO label files, cleans OOB boxes, and eliminates sub-pixel noise.
    """
    base_dir = os.path.join(VISION_BASE, dataset_name)
    lbl_files = glob.glob(os.path.join(base_dir, "labels", "**", "*.txt"), recursive=True)
    
    fixed_oob = 0
    removed_tiny = 0
    cleaned_files = 0
    
    for f in lbl_files:
        with open(f, "r") as fp:
            lines = fp.readlines()
            
        new_lines = []
        for l in lines:
            parts = l.strip().split()
            if len(parts) >= 5:
                cx, cy, w, h = map(float, parts[1:5])
                is_oob = (cx - w/2 < 0) or (cx + w/2 > 1.0) or (cy - h/2 < 0) or (cy + h/2 > 1.0)
                is_tiny = (w < 0.005) or (h < 0.005)
                
                clean_l = clamp_and_filter_bbox(l)
                if clean_l is not None:
                    new_lines.append(clean_l)
                    if is_oob:
                        fixed_oob += 1
                elif is_tiny:
                    removed_tiny += 1
                    
        with open(f, "w") as fp:
            if new_lines:
                fp.write("\n".join(new_lines) + "\n")
            else:
                fp.write("") # Keep empty if no valid boxes
                
        cleaned_files += 1
        
    print(f"[{dataset_name}] Bounding Box Cleaning Complete:")
    print(f"   Clamped OOB Boxes: {fixed_oob} | Removed Sub-pixel Boxes: {removed_tiny} | Files Checked: {cleaned_files}")

def balance_background_frames(dataset_name="bharat_pothole", bg_ratio=0.15):
    """
    Rebalances datasets with extreme negative/empty frame surplus (e.g. 91% empty in bharat_pothole).
    Keeps a healthy ratio of negative frames (bg_ratio * positive_frames) and moves
    the excess empty images into a quarantine folder.
    """
    base_dir = os.path.join(VISION_BASE, dataset_name)
    quarantine_img_dir = os.path.join(base_dir, "background_quarantine", "images")
    quarantine_lbl_dir = os.path.join(base_dir, "background_quarantine", "labels")
    os.makedirs(quarantine_img_dir, exist_ok=True)
    os.makedirs(quarantine_lbl_dir, exist_ok=True)
    
    total_quarantined = 0
    
    for split in ["train", "val", "test"]:
        img_dir = os.path.join(base_dir, "images", split)
        lbl_dir = os.path.join(base_dir, "labels", split)
        if not os.path.exists(lbl_dir):
            continue
            
        lbl_files = sorted(glob.glob(os.path.join(lbl_dir, "*.txt")))
        
        pos_files = []
        empty_files = []
        
        for lf in lbl_files:
            bname = os.path.splitext(os.path.basename(lf))[0]
            with open(lf, "r") as fp:
                content = fp.read().strip()
            if content:
                pos_files.append(bname)
            else:
                empty_files.append(bname)
                
        max_allowed_empty = max(1, int(len(pos_files) * bg_ratio))
        to_quarantine = empty_files[max_allowed_empty:]
        
        for bname in to_quarantine:
            img_path = os.path.join(img_dir, f"{bname}.jpg")
            lbl_path = os.path.join(lbl_dir, f"{bname}.txt")
            
            if os.path.exists(img_path):
                shutil.move(img_path, os.path.join(quarantine_img_dir, f"{bname}.jpg"))
            if os.path.exists(lbl_path):
                shutil.move(lbl_path, os.path.join(quarantine_lbl_dir, f"{bname}.txt"))
                
            total_quarantined += 1
            
        retained_empty = len(empty_files) - len(to_quarantine)
        print(f"[{dataset_name}/{split}] Positive: {len(pos_files)} | Retained Background: {retained_empty} (Ratio: {retained_empty/max(1, len(pos_files)):.1%}) | Quarantined: {len(to_quarantine)}")
        
    print(f"[{dataset_name}] Background Balancing Complete: Quarantined {total_quarantined} redundant empty frames.")

def verify_all_images_decodability():
    """Verify all images can be cleanly decoded by OpenCV without byte truncation."""
    img_files = glob.glob(os.path.join(VISION_BASE, "**", "*.jpg"), recursive=True)
    corrupted_cnt = 0
    
    for ipath in img_files:
        try:
            img = cv2.imread(ipath)
            if img is None:
                corrupted_cnt += 1
        except Exception:
            corrupted_cnt += 1
            
    print(f"[Image Integrity Check] Checked {len(img_files):,} images. Corrupted/unreadable: {corrupted_cnt}")

def run_vision_cleaning():
    print("=" * 60)
    print("EXECUTING VISION DATA CLEANING PIPELINE")
    print("=" * 60)
    for ds in ["bharat_pothole", "rdd2022_d40", "indian_roads", "crddc2022"]:
        clean_vision_annotations(ds)
    balance_background_frames("bharat_pothole", bg_ratio=0.15)
    verify_all_images_decodability()
    print("=" * 60)
    print("VISION DATA CLEANING COMPLETED!")
    print("=" * 60)

if __name__ == "__main__":
    run_vision_cleaning()

