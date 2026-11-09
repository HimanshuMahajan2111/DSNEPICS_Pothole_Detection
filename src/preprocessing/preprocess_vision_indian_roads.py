"""
Indian Roads Dataset (Dataset3Class) Preprocessing Module
Applies CLAHE contrast enhancement, letterbox resizing (640x640),
transforms YOLO coordinates, and partitions into train/val/test splits.
Output saved to: data/processed/vision/indian_roads/
"""

import os
import glob
import random
import cv2
import numpy as np

DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "data"))
INPUT_BASE = os.path.join(DATA_DIR, "dashcam_data", "Indian Roads Dataset", "Dataset3Class")
OUTPUT_BASE = os.path.join(DATA_DIR, "processed", "vision", "indian_roads")

def apply_clahe(image):
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    enhanced = cv2.merge((cl, a, b))
    return cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)

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
    return canvas, scale, left, top

def adjust_yolo_bbox(line, orig_w, orig_h, scale, pad_x, pad_y, target_size=(640, 640)):
    parts = line.strip().split()
    if len(parts) < 5:
        return None
    cls_id = parts[0]
    cx, cy, w, h = map(float, parts[1:5])
    
    tw, th = target_size
    
    # Convert normalized back to original pixels
    x1 = (cx - w / 2.0) * orig_w
    x2 = (cx + w / 2.0) * orig_w
    y1 = (cy - h / 2.0) * orig_h
    y2 = (cy + h / 2.0) * orig_h
    
    # Transform to padded canvas
    x1_pad = x1 * scale + pad_x
    x2_pad = x2 * scale + pad_x
    y1_pad = y1 * scale + pad_y
    y2_pad = y2 * scale + pad_y
    
    new_w = (x2_pad - x1_pad) / tw
    new_h = (y2_pad - y1_pad) / th
    new_cx = (x1_pad + x2_pad) / (2.0 * tw)
    new_cy = (y1_pad + y2_pad) / (2.0 * th)
    
    return f"{cls_id} {new_cx:.6f} {new_cy:.6f} {new_w:.6f} {new_h:.6f}"

def process_indian_roads(max_samples=1200, train_split=0.8, val_split=0.1, seed=42):
    img_files = sorted(glob.glob(os.path.join(INPUT_BASE, "*.jpg")))
    print(f"Found {len(img_files)} images in Indian Roads Dataset...")
    
    random.seed(seed)
    random.shuffle(img_files)
    
    if max_samples:
        img_files = img_files[:max_samples]
        
    n_total = len(img_files)
    n_train = int(n_total * train_split)
    n_val = int(n_total * val_split)
    
    splits = {
        'train': img_files[:n_train],
        'val': img_files[n_train:n_train + n_val],
        'test': img_files[n_train + n_val:]
    }
    
    counts = {}
    for split_name, files in splits.items():
        dst_img_dir = os.path.join(OUTPUT_BASE, "images", split_name)
        dst_lbl_dir = os.path.join(OUTPUT_BASE, "labels", split_name)
        os.makedirs(dst_img_dir, exist_ok=True)
        os.makedirs(dst_lbl_dir, exist_ok=True)
        
        proc_cnt = 0
        for ipath in files:
            tpath = ipath.replace(".jpg", ".txt")
            if not os.path.exists(tpath):
                continue
                
            img = cv2.imread(ipath)
            if img is None:
                continue
                
            orig_h, orig_w = img.shape[:2]
            enhanced = apply_clahe(img)
            lb_img, scale, pad_x, pad_y = resize_letterbox(enhanced)
            
            with open(tpath) as f:
                lines = f.readlines()
                
            adjusted_lines = []
            for l in lines:
                adj = adjust_yolo_bbox(l, orig_w, orig_h, scale, pad_x, pad_y)
                if adj:
                    adjusted_lines.append(adj)
                    
            if not adjusted_lines:
                continue
                
            bname = os.path.splitext(os.path.basename(ipath))[0]
            cv2.imwrite(os.path.join(dst_img_dir, f"{bname}.jpg"), lb_img)
            with open(os.path.join(dst_lbl_dir, f"{bname}.txt"), "w") as f:
                f.write("\n".join(adjusted_lines) + "\n")
                
            proc_cnt += 1
            
        counts[split_name] = proc_cnt
        print(f"Split '{split_name}' completed: {proc_cnt} images.")
        
    yaml_content = f"""# Indian Roads Dataset YOLO Configuration
path: {OUTPUT_BASE}
train: images/train
val: images/val
test: images/test

nc: 3
names: ['road_damage', 'pothole', 'surface_anomaly']
"""
    yaml_path = os.path.join(OUTPUT_BASE, "data.yaml")
    with open(yaml_path, "w") as f:
        f.write(yaml_content)
        
    print(f"Indian Roads Preprocessing Complete: {counts}")
    print(f"YAML config saved to: {yaml_path}")

if __name__ == "__main__":
    process_indian_roads()

