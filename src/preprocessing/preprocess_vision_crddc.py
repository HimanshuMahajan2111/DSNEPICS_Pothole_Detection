"""
CRDDC2022 Full Road Damage Preprocessing Module
Extracts all road damage classes (D00-D44), applies CLAHE, letterbox 640x640,
and formats into: data/processed/vision/crddc2022/
"""

import os
import glob
import random
import xml.etree.ElementTree as ET
import cv2
import numpy as np

DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "data"))
INPUT_BASE = os.path.join(DATA_DIR, "dashcam_data", "RDD2022_released_through_CRDDC2022", "RDD2022", "India", "train")
OUTPUT_BASE = os.path.join(DATA_DIR, "processed", "vision", "crddc2022")

CLASSES = ['D00', 'D10', 'D20', 'D40', 'D43', 'D44']
CLASS_MAP = {c: i for i, c in enumerate(CLASSES)}

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
    return canvas, scale, left, top

def convert_xml_to_yolo(xml_path, orig_w, orig_h, scale, pad_x, pad_y, target_size=(640, 640)):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    yolo_boxes = []
    tw, th = target_size
    
    for obj in root.findall('object'):
        name = obj.find('name').text
        if name not in CLASS_MAP:
            continue
        cls_idx = CLASS_MAP[name]
        bndbox = obj.find('bndbox')
        xmin = float(bndbox.find('xmin').text)
        ymin = float(bndbox.find('ymin').text)
        xmax = float(bndbox.find('xmax').text)
        ymax = float(bndbox.find('ymax').text)
        
        xmin_pad = max(0.0, min(float(tw), xmin * scale + pad_x))
        xmax_pad = max(0.0, min(float(tw), xmax * scale + pad_x))
        ymin_pad = max(0.0, min(float(th), ymin * scale + pad_y))
        ymax_pad = max(0.0, min(float(th), ymax * scale + pad_y))
        
        bw = xmax_pad - xmin_pad
        bh = ymax_pad - ymin_pad
        if bw <= 1 or bh <= 1:
            continue
            
        xc = (xmin_pad + bw / 2.0) / tw
        yc = (ymin_pad + bh / 2.0) / th
        nw = bw / tw
        nh = bh / th
        yolo_boxes.append(f"{cls_idx} {xc:.6f} {yc:.6f} {nw:.6f} {nh:.6f}")
        
    return yolo_boxes

def process_crddc2022(max_samples=1000, seed=42):
    images_dir = os.path.join(INPUT_BASE, "images")
    xmls_dir = os.path.join(INPUT_BASE, "annotations", "xmls")
    
    xml_files = sorted(glob.glob(os.path.join(xmls_dir, "*.xml")))
    print(f"Found {len(xml_files)} XML annotations in CRDDC2022 India...")
    
    random.seed(seed)
    random.shuffle(xml_files)
    if max_samples:
        xml_files = xml_files[:max_samples]
        
    n_train = int(len(xml_files) * 0.8)
    n_val = int(len(xml_files) * 0.1)
    
    splits = {
        'train': xml_files[:n_train],
        'val': xml_files[n_train:n_train + n_val],
        'test': xml_files[n_train + n_val:]
    }
    
    counts = {}
    for sname, x_list in splits.items():
        dst_img = os.path.join(OUTPUT_BASE, "images", sname)
        dst_lbl = os.path.join(OUTPUT_BASE, "labels", sname)
        os.makedirs(dst_img, exist_ok=True)
        os.makedirs(dst_lbl, exist_ok=True)
        
        proc_cnt = 0
        for xf in x_list:
            bname = os.path.splitext(os.path.basename(xf))[0]
            ipath = os.path.join(images_dir, f"{bname}.jpg")
            if not os.path.exists(ipath):
                continue
            img = cv2.imread(ipath)
            if img is None:
                continue
            orig_h, orig_w = img.shape[:2]
            enh = apply_clahe(img)
            lb, scale, px, py = resize_letterbox(enh)
            yolo_boxes = convert_xml_to_yolo(xf, orig_w, orig_h, scale, px, py)
            if not yolo_boxes:
                continue
            cv2.imwrite(os.path.join(dst_img, f"{bname}.jpg"), lb)
            with open(os.path.join(dst_lbl, f"{bname}.txt"), "w") as f:
                f.write("\n".join(yolo_boxes) + "\n")
            proc_cnt += 1
            
        counts[sname] = proc_cnt
        print(f"CRDDC split '{sname}' completed: {proc_cnt} images.")
        
    yaml_content = f"""# CRDDC2022 All-Damage YOLO Configuration
path: {OUTPUT_BASE}
train: images/train
val: images/val
test: images/test

nc: {len(CLASSES)}
names: {CLASSES}
"""
    with open(os.path.join(OUTPUT_BASE, "data.yaml"), "w") as f:
        f.write(yaml_content)
        
    print(f"CRDDC2022 Preprocessing Complete: {counts}")

if __name__ == "__main__":
    process_crddc2022()

