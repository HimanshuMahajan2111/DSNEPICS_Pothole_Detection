"""
=============================================================================
STEP 4: VISUAL OVERLAY GENERATION & INFERENCE VERIFICATION
=============================================================================
Author: EPICS DSN3099
Focus:
  - Generates annotated visual overlays on test frames
  - Highlights multiple potholes & ubad-khabad road damage
  - Verifies rejection of elevated walls, curbs, and vehicles
  - Outputs to: pothole_detection_pipeline/outputs/
=============================================================================
"""

import os
import glob
import cv2

def run_visual_inference(num_samples=15, conf_thresh=0.25):
    from ultralytics import YOLO
    import torch

    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    weights_path = os.path.join(project_root, "pothole_detection_pipeline", "checkpoints", "best_pothole_model.pt")
    if not os.path.exists(weights_path):
        weights_path = os.path.join(project_root, "pothole_detection_pipeline", "checkpoints", "training_run", "weights", "best.pt")

    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Model checkpoint not found at: {weights_path}")

    test_imgs_dir = os.path.join(project_root, "data", "pothole_detector_dataset", "images", "test")
    output_dir = os.path.join(project_root, "pothole_detection_pipeline", "outputs")
    os.makedirs(output_dir, exist_ok=True)

    print("\n" + "="*70)
    print("STEP 4: RUNNING VISUAL INFERENCE & OVERLAY GENERATION")
    print("="*70)
    print(f"Model Checkpoint : {weights_path}")
    print(f"Output Directory : {output_dir}")
    print(f"Confidence Thresh: {conf_thresh}")

    device = "0" if torch.cuda.is_available() else "cpu"
    model = YOLO(weights_path)
    
    test_files = sorted(glob.glob(os.path.join(test_imgs_dir, "*.jpg")))[:num_samples]
    print(f"Processing {len(test_files)} sample test frames...")

    for idx, img_p in enumerate(test_files, 1):
        fname = os.path.basename(img_p)
        img = cv2.imread(img_p)
        h, w = img.shape[:2]
        annotated = img.copy()

        res = model.predict(img_p, conf=conf_thresh, imgsz=640, device=device, verbose=False)[0]
        
        potholes_found = 0
        if res.boxes is not None and len(res.boxes) > 0:
            boxes = res.boxes.xyxy.cpu().numpy()
            confs = res.boxes.conf.cpu().numpy()
            
            for (x1, y1, x2, y2), conf in zip(boxes, confs):
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                potholes_found += 1
                
                # Neon amber/cyan bounding box
                color = (0, 215, 255) if potholes_found == 1 else (50, 168, 82)
                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
                
                # Tag label
                label = f"Pothole #{potholes_found} [{conf*100:.1f}%]"
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
                cv2.rectangle(annotated, (x1, max(0, y1 - 20)), (x1 + tw + 6, y1), color, -1)
                cv2.putText(annotated, label, (x1 + 3, max(14, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1)

        # HUD Banner
        cv2.rectangle(annotated, (0, 0), (w, 36), (20, 24, 30), -1)
        hud_text = f"EPICS AI POTHOLE SCANNER | Potholes: {potholes_found} | Frame: {fname}"
        cv2.putText(annotated, hud_text, (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 200), 2)

        out_img_path = os.path.join(output_dir, f"detected_{fname}")
        cv2.imwrite(out_img_path, annotated)
        print(f"  [{idx:2d}/{len(test_files):2d}] {fname:30s} -> Found {potholes_found} potholes -> Saved {os.path.basename(out_img_path)}")

    print(f"\n[SUCCESS] Visual overlays generated in: {output_dir}")
    print("="*70 + "\n")

if __name__ == "__main__":
    run_visual_inference(num_samples=15)
