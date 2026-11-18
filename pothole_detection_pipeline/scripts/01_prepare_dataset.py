"""
=============================================================================
STEP 1: EXPANDED MULTI-DATASET CURATION (5,500+ REAL INDIAN POTHOLES)
=============================================================================
Author: EPICS DSN3099
Focus:
  - Ingests all available datasets:
    1. RDD2022 India Potholes (D40): 3,187 cavity annotations
    2. Indian Roads Dataset: 2,122 cavity annotations
    3. Bharat Pothole Dataset: 276 cavity annotations
  - Balances with 200 clean road/wall/barrier negative samples to prevent
    any false positives on elevated structures or vehicles.
=============================================================================
"""

import os
import glob
import shutil
import random

def prepare_dataset():
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    dashcam_base = os.path.join(project_root, "data", "dashcam_data")
    vision_base = os.path.join(project_root, "data", "processed", "vision")
    dataset_out = os.path.join(project_root, "data", "pothole_detector_dataset")
    
    print("\n" + "="*70)
    print("STEP 1: EXPANDING DATASET WITH ALL INDIAN ROAD POTHOLES")
    print("="*70)
    
    for split in ["train", "val", "test"]:
        os.makedirs(os.path.join(dataset_out, "images", split), exist_ok=True)
        os.makedirs(os.path.join(dataset_out, "labels", split), exist_ok=True)

    # 1. Ingest RDD2022_D40 (Ground truth D40 road cavity potholes)
    d40_base = os.path.join(vision_base, "rdd2022_d40")
    print("\n[1/4] Ingesting RDD2022 D40 pothole frames and labels...")
    copied_d40 = 0
    for split in ["train", "val", "test"]:
        img_src = os.path.join(d40_base, "images", split)
        lbl_src = os.path.join(d40_base, "labels", split)
        
        imgs = glob.glob(os.path.join(img_src, "*.jpg"))
        for img_p in imgs:
            fname = os.path.basename(img_p)
            lbl_p = os.path.join(lbl_src, fname.replace(".jpg", ".txt"))
            
            dest_img = os.path.join(dataset_out, "images", split, fname)
            dest_lbl = os.path.join(dataset_out, "labels", split, fname.replace(".jpg", ".txt"))
            
            if not os.path.exists(dest_img):
                shutil.copy2(img_p, dest_img)
            if os.path.exists(lbl_p) and not os.path.exists(dest_lbl):
                shutil.copy2(lbl_p, dest_lbl)
            copied_d40 += 1
    print(f"      Loaded {copied_d40} D40 image-label pairs.")

    # 2. Ingest Indian Roads Dataset (all 2,122 pothole annotations)
    print("\n[2/4] Ingesting Indian Roads Dataset (Class 1 -> Class 0 potholes)...")
    ir_dir = os.path.join(dashcam_base, "Indian Roads Dataset")
    ir_txts = glob.glob(ir_dir + "/**/*.txt", recursive=True)
    
    ir_pairs = []
    for tp in ir_txts:
        with open(tp, "r", errors="ignore") as f:
            lines = [l.strip() for l in f if l.strip()]
        # Check if contains class 1 (pothole)
        pothole_lines = []
        for l in lines:
            parts = l.split()
            if parts[0] == '1': # Class 1 is pothole
                # Remap to class 0
                pothole_lines.append(f"0 {' '.join(parts[1:])}")
        if pothole_lines:
            img_p = tp.replace(".txt", ".jpg")
            if os.path.exists(img_p):
                ir_pairs.append((img_p, pothole_lines))
                
    random.seed(42)
    random.shuffle(ir_pairs)
    
    # 85% train, 15% val
    n_ir_val = int(len(ir_pairs) * 0.15)
    ir_train = ir_pairs[n_ir_val:]
    ir_val = ir_pairs[:n_ir_val]
    
    for img_p, p_lines in ir_train:
        fname = f"ir_{os.path.basename(img_p)}"
        dest_img = os.path.join(dataset_out, "images", "train", fname)
        dest_lbl = os.path.join(dataset_out, "labels", "train", fname.replace(".jpg", ".txt"))
        if not os.path.exists(dest_img):
            shutil.copy2(img_p, dest_img)
        with open(dest_lbl, "w") as f:
            f.write("\n".join(p_lines) + "\n")
            
    for img_p, p_lines in ir_val:
        fname = f"ir_{os.path.basename(img_p)}"
        dest_img = os.path.join(dataset_out, "images", "val", fname)
        dest_lbl = os.path.join(dataset_out, "labels", "val", fname.replace(".jpg", ".txt"))
        if not os.path.exists(dest_img):
            shutil.copy2(img_p, dest_img)
        with open(dest_lbl, "w") as f:
            f.write("\n".join(p_lines) + "\n")
            
    print(f"      Loaded {len(ir_pairs)} frames ({len(ir_train)} train, {len(ir_val)} val) from Indian Roads Dataset.")

    # 3. Ingest Bharat Pothole Dataset
    print("\n[3/4] Ingesting Bharat Pothole positive samples...")
    bharat_lbls = glob.glob(os.path.join(vision_base, "bharat_pothole", "labels", "**", "*.txt"), recursive=True)
    bp_cnt = 0
    for lp in bharat_lbls:
        with open(lp, "r") as f:
            lines = [x.strip() for x in f if x.strip()]
        if lines:
            img_p = lp.replace("labels", "images").replace(".txt", ".jpg")
            if os.path.exists(img_p):
                fname = f"bp_{os.path.basename(img_p)}"
                dest_img = os.path.join(dataset_out, "images", "train", fname)
                dest_lbl = os.path.join(dataset_out, "labels", "train", fname.replace(".jpg", ".txt"))
                if not os.path.exists(dest_img):
                    shutil.copy2(img_p, dest_img)
                if not os.path.exists(dest_lbl):
                    shutil.copy2(lp, dest_lbl)
                bp_cnt += 1
    print(f"      Loaded {bp_cnt} Bharat Pothole frames into training split.")

    # 4. Inject Negative Background Samples (clean road, walls, elevated borders)
    print("\n[4/4] Adding negative background samples (clean pavement, walls, elevated barriers)...")
    clean_candidates = []
    for lp in bharat_lbls:
        with open(lp, "r") as f:
            if not f.read().strip():
                img_p = lp.replace("labels", "images").replace(".txt", ".jpg")
                if os.path.exists(img_p):
                    clean_candidates.append(img_p)
                    
    random.shuffle(clean_candidates)
    selected_clean = clean_candidates[:200]
    for idx, img_p in enumerate(selected_clean):
        split = "train" if idx < 150 else ("val" if idx < 175 else "test")
        fname = f"bg_{os.path.basename(img_p)}"
        dest_img = os.path.join(dataset_out, "images", split, fname)
        dest_lbl = os.path.join(dataset_out, "labels", split, fname.replace(".jpg", ".txt"))
        if not os.path.exists(dest_img):
            shutil.copy2(img_p, dest_img)
        if not os.path.exists(dest_lbl):
            with open(dest_lbl, "w") as f:
                f.write("")
    print(f"      Added {len(selected_clean)} clean/negative background frames.")

    # Generate YAML configuration
    yaml_path = os.path.join(dataset_out, "pothole_dataset.yaml")
    yaml_content = f"""# Dedicated Pothole Detection Dataset Configuration
path: {dataset_out}
train: images/train
val: images/val
test: images/test

nc: 1
names: ['pothole']
"""
    with open(yaml_path, "w") as f:
        f.write(yaml_content)
    print(f"\nGenerated dataset configuration YAML: {yaml_path}")

    # Dataset Summary Statistics
    print("\n" + "-"*70)
    print("EXPANDED DATASET SUMMARY:")
    print("-"*70)
    for s in ["train", "val", "test"]:
        imgs = glob.glob(os.path.join(dataset_out, "images", s, "*.jpg"))
        lbls = glob.glob(os.path.join(dataset_out, "labels", s, "*.txt"))
        boxes = 0
        multi_count = 0
        for lp in lbls:
            with open(lp, "r") as f:
                b = len([x for x in f if x.strip()])
                boxes += b
                if b > 1:
                    multi_count += 1
        print(f"  Split '{s:5s}': {len(imgs):4d} images | {boxes:4d} potholes | {multi_count:3d} frames with multiple potholes")
    print("="*70 + "\n")
    return yaml_path

if __name__ == "__main__":
    prepare_dataset()
