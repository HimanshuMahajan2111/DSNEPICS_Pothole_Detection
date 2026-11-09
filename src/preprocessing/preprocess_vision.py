"""
Vision Preprocessing Module
Converts Pascal VOC XML annotations to normalized YOLO format,
applies CLAHE contrast enhancement for pothole visibility,
resizes images with letterboxing, and creates train/val/test splits.
"""

import os
import glob
import random
import xml.etree.ElementTree as ET
import cv2
import numpy as np

DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "data"))
VISION_INPUT_DIR = os.path.join(DATA_DIR, "dashcam_data", "RDD2022 India Pothole D40", "train")
OUTPUT_BASE = os.path.join(DATA_DIR, "processed", "vision")

def apply_clahe(image):
    """Enhance pothole depth contrast using CLAHE on the luminance channel."""
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    enhanced = cv2.merge((cl, a, b))
    return cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)

def resize_letterbox(image, target_size=(640, 640)):
    """Resize image preserving aspect ratio with zero padding."""
    h, w = image.shape[:2]
    th, tw = target_size
    scale = min(tw / w, th / h)
    nw, nh = int(w * scale), int(h * scale)
    
    resized = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((th, tw, 3), 114, dtype=np.uint8) # Standard YOLO padding
    
    top = (th - nh) // 2
    left = (tw - nw) // 2
    canvas[top:top+nh, left:left+nw] = resized
    
    return canvas, scale, left, top

def convert_voc_to_yolo(xml_path, orig_w, orig_h, scale, pad_x, pad_y, target_size=(640, 640)):
    """
    Parse VOC XML, extract D40 (pothole), and adjust coordinates for letterboxed 640x640 image.
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()
    yolo_boxes = []
    
    tw, th = target_size
    
    for obj in root.findall('object'):
        name = obj.find('name').text
        if name != 'D40': # Focus specifically on potholes
            continue
            
        bndbox = obj.find('bndbox')
        xmin = float(bndbox.find('xmin').text)
        ymin = float(bndbox.find('ymin').text)
        xmax = float(bndbox.find('xmax').text)
        ymax = float(bndbox.find('ymax').text)
        
        # Transform coords into letterboxed frame
        xmin_pad = xmin * scale + pad_x
        xmax_pad = xmax * scale + pad_x
        ymin_pad = ymin * scale + pad_y
        ymax_pad = ymax * scale + pad_y
        
        # Clip to target bounds
        xmin_pad = max(0.0, min(float(tw), xmin_pad))
        xmax_pad = max(0.0, min(float(tw), xmax_pad))
        ymin_pad = max(0.0, min(float(th), ymin_pad))
        ymax_pad = max(0.0, min(float(th), ymax_pad))
        
        box_w = xmax_pad - xmin_pad
        box_h = ymax_pad - ymin_pad
        
        if box_w <= 1 or box_h <= 1:
            continue
            
        x_center = (xmin_pad + box_w / 2.0) / tw
        y_center = (ymin_pad + box_h / 2.0) / th
        norm_w = box_w / tw
        norm_h = box_h / th
        
        # Class 0: pothole
        yolo_boxes.append(f"0 {x_center:.6f} {y_center:.6f} {norm_w:.6f} {norm_h:.6f}")
        
    return yolo_boxes

def process_vision_dataset(target_size=(640, 640), train_split=0.8, val_split=0.1, max_samples=None, seed=42):
    """
    Full pipeline to clean, enhance, format, and split RDD2022 pothole data.
    """
    images_dir = os.path.join(VISION_INPUT_DIR, "images")
    xmls_dir = os.path.join(VISION_INPUT_DIR, "annotations", "xmls")
    
    xml_files = sorted(glob.glob(os.path.join(xmls_dir, "*.xml")))
    print(f"Total XML annotations available: {len(xml_files)}")
    
    # Filter for pairs that actually contain D40 potholes
    valid_pairs = []
    for xf in xml_files:
        base_name = os.path.splitext(os.path.basename(xf))[0]
        img_path = os.path.join(images_dir, f"{base_name}.jpg")
        if not os.path.exists(img_path):
            continue
        try:
            tree = ET.parse(xf)
            has_d40 = any(obj.find('name').text == 'D40' for obj in tree.findall('object'))
            if has_d40:
                valid_pairs.append((img_path, xf, base_name))
        except Exception:
            continue

    print(f"Images verified with confirmed D40 potholes: {len(valid_pairs)}")
    
    random.seed(seed)
    random.shuffle(valid_pairs)
    
    if max_samples:
        valid_pairs = valid_pairs[:max_samples]
        print(f"Processing subset of {len(valid_pairs)} samples...")
        
    n_total = len(valid_pairs)
    n_train = int(n_total * train_split)
    n_val = int(n_total * val_split)
    
    splits = {
        'train': valid_pairs[:n_train],
        'val': valid_pairs[n_train:n_train + n_val],
        'test': valid_pairs[n_train + n_val:]
    }
    
    counts = {'train': 0, 'val': 0, 'test': 0}
    total_boxes = 0
    
    for split_name, items in splits.items():
        img_out_dir = os.path.join(OUTPUT_BASE, "images", split_name)
        lbl_out_dir = os.path.join(OUTPUT_BASE, "labels", split_name)
        os.makedirs(img_out_dir, exist_ok=True)
        os.makedirs(lbl_out_dir, exist_ok=True)
        
        for img_path, xml_path, base_name in items:
            img = cv2.imread(img_path)
            if img is None:
                continue
                
            orig_h, orig_w = img.shape[:2]
            
            # 1. Apply CLAHE enhancement
            enhanced_img = apply_clahe(img)
            
            # 2. Letterbox resize
            lb_img, scale, pad_x, pad_y = resize_letterbox(enhanced_img, target_size=target_size)
            
            # 3. Convert VOC to YOLO coordinates
            yolo_boxes = convert_voc_to_yolo(xml_path, orig_w, orig_h, scale, pad_x, pad_y, target_size=target_size)
            if not yolo_boxes:
                continue
                
            # 4. Save image and label
            out_img_path = os.path.join(img_out_dir, f"{base_name}.jpg")
            out_lbl_path = os.path.join(lbl_out_dir, f"{base_name}.txt")
            
            cv2.imwrite(out_img_path, lb_img)
            with open(out_lbl_path, "w") as f:
                f.write("\n".join(yolo_boxes) + "\n")
                
            counts[split_name] += 1
            total_boxes += len(yolo_boxes)
            
    print(f"Vision Preprocessing Complete!")
    print(f"Processed Splits: Train={counts['train']}, Val={counts['val']}, Test={counts['test']}")
    print(f"Total Pothole Bounding Boxes Formatted: {total_boxes}")
    
    # Create data.yaml
    yaml_content = f"""# YOLOv8 / YOLOv11 Pothole Dataset Configuration
path: {OUTPUT_BASE}
train: images/train
val: images/val
test: images/test

# Classes
nc: 1
names: ['pothole']
"""
    yaml_path = os.path.join(OUTPUT_BASE, "data.yaml")
    with open(yaml_path, "w") as f:
        f.write(yaml_content)
    print(f"Generated YAML config: {yaml_path}")

if __name__ == "__main__":
    process_vision_dataset()

