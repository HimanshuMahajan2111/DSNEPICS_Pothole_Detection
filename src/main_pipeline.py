"""
Master End-to-End Inference Pipeline
====================================
EPICS DSN3099: Vision-Inertial 3D Pothole Detection & Validation Engine

Pipeline Stages:
1. Ingest image frame or video stream.
2. Detect pothole bounding boxes (YOLO format / heuristic detection).
3. Reconstruct 3D metric surface geometry (depth, width, length, volume, edge slope).
4. Simulate dynamic vehicle impact response using 2-DOF Quarter-Car model.
5. Cross-validate against empirical sensor shock distributions and compute physical confidence score.
6. Render annotated visualization and output structured JSON audit report.
"""

import os
import sys
import glob
import json
import cv2
import numpy as np

# Ensure project root in python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.geometry.depth_reconstructor import Pothole3DReconstructor
from src.validation.sensor_cross_validator import SensorCrossValidator

class MasterPipeline:
    def __init__(self, h_cam=1.25, fov_deg=70.0, y_horizon=0.45):
        self.reconstructor = Pothole3DReconstructor(h_cam=h_cam, fov_deg=fov_deg, y_horizon=y_horizon)
        self.validator = SensorCrossValidator()
        
    def process_single_frame(self, image_path, label_path=None, simulated_speed_kmh=35.0, output_annotated_path=None):
        """
        Executes end-to-end inference on a single image frame.
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")
            
        img_bgr = cv2.imread(image_path)
        h_img, w_img = img_bgr.shape[:2]
        annotated = img_bgr.copy()
        
        # 1. Parse or Detect Bounding Boxes
        bboxes = []
        if label_path and os.path.exists(label_path):
            with open(label_path, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        bboxes.append([float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])])
        else:
            # Automatic heuristic detection for demonstration if label not supplied:
            # Look for dark low-reflection contours in the lower road region
            gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
            road_roi = gray[int(h_img * 0.55):, :]
            _, thresh = cv2.threshold(road_roi, 70, 255, cv2.THRESH_BINARY_INV)
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if area > 1200:
                    bx, by, bw, bh = cv2.boundingRect(cnt)
                    by += int(h_img * 0.55)
                    cx = (bx + bw / 2.0) / w_img
                    cy = (by + bh / 2.0) / h_img
                    bboxes.append([cx, cy, bw / w_img, bh / h_img])
                    
        detections_report = []
        
        # 2. Process each detected pothole
        for idx, bbox in enumerate(bboxes):
            geom = self.reconstructor.reconstruct_pothole_geometry(img_bgr, bbox)
            if geom is None:
                continue
                
            # 3. Cross-validate with sensor dynamics
            val = self.validator.validate_pothole_reconstruction(
                depth_max_cm=geom["depth_max_cm"],
                length_cm=geom["length_cm"],
                width_cm=geom["width_cm"],
                speed_kmh=simulated_speed_kmh
            )
            
            # Combine record
            record = {
                "pothole_id": idx + 1,
                "geometry": {
                    "distance_z_m": round(geom["distance_z_m"], 2),
                    "width_cm": round(geom["width_cm"], 1),
                    "length_cm": round(geom["length_cm"], 1),
                    "depth_max_cm": round(geom["depth_max_cm"], 1),
                    "volume_cm3": round(geom["volume_cm3"], 1),
                    "edge_slope_deg": round(geom["edge_slope_deg"], 1)
                },
                "dynamic_validation": {
                    "simulated_speed_kmh": round(simulated_speed_kmh, 1),
                    "simulated_peak_g": round(val["simulated_peak_g"], 4),
                    "speed_norm_shock": round(val["speed_normalized_shock"], 4),
                    "confidence_score": round(val["confidence_score"], 4),
                    "severity_tier": val["severity_tier"],
                    "tier_description": val["tier_description"],
                    "validation_status": val["validation_status"]
                }
            }
            detections_report.append(record)
            
            # 4. Render on visualization overlay
            x1, y1, x2, y2 = geom["bbox_pixels"]
            tier = val["severity_tier"]
            # Color code: Tier 1 = Green, Tier 2 = Orange, Tier 3 = Red
            color = (46, 204, 113) if tier == 1 else ((243, 156, 18) if tier == 2 else (231, 76, 60))
            
            # Draw bounding box and filled title tag
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 3)
            label_text = f"Pothole #{idx+1} [T{tier}] d={geom['depth_max_cm']:.1f}cm W={geom['width_cm']:.0f}cm"
            sub_text = f"Shock: {val['simulated_peak_g']:.2f}g | Conf: {val['confidence_score']*100:.1f}%"
            
            cv2.rectangle(annotated, (x1, max(0, y1 - 42)), (x1 + 320, y1), color, -1)
            cv2.putText(annotated, label_text, (x1 + 6, max(16, y1 - 24)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
            cv2.putText(annotated, sub_text, (x1 + 6, max(36, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
            
        # Draw HUD dashboard banner
        cv2.rectangle(annotated, (0, 0), (w_img, 45), (20, 24, 33), -1)
        hud_title = f"EPICS DSN3099 AI ROAD SCANNER | Potholes: {len(detections_report)} | Speed: {simulated_speed_kmh:.0f} km/h"
        cv2.putText(annotated, hud_title, (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 220, 255), 2)
        
        if output_annotated_path:
            os.makedirs(os.path.dirname(output_annotated_path), exist_ok=True)
            cv2.imwrite(output_annotated_path, annotated)
            
        return detections_report, annotated


def run_batch_evaluation(max_images=10):
    """
    Runs batch evaluation across processed test dataset and produces a summary JSON report.
    """
    print(f"\n[Master Pipeline] Running batch evaluation across {max_images} test frames...")
    pipeline = MasterPipeline()
    
    img_dir = "d:/DSN3099-EPICS/Implementation/data/processed/vision/rdd2022_d40/images/test"
    lbl_dir = "d:/DSN3099-EPICS/Implementation/data/processed/vision/rdd2022_d40/labels/test"
    out_dir = "d:/DSN3099-EPICS/Implementation/eda_report/pipeline_outputs"
    os.makedirs(out_dir, exist_ok=True)
    
    test_files = sorted(glob.glob(os.path.join(img_dir, "*.jpg")))[:max_images]
    
    all_results = []
    for i, img_path in enumerate(test_files, 1):
        fname = os.path.basename(img_path)
        lbl_path = os.path.join(lbl_dir, fname.replace(".jpg", ".txt"))
        out_img = os.path.join(out_dir, f"detected_{fname}")
        
        # Vary vehicle speed between 25 and 45 km/h to simulate realistic driving
        speed = 25.0 + (i % 5) * 5.0
        
        report, _ = pipeline.process_single_frame(
            img_path,
            label_path=lbl_path,
            simulated_speed_kmh=speed,
            output_annotated_path=out_img
        )
        
        all_results.append({
            "image": fname,
            "simulated_speed_kmh": speed,
            "potholes_found": len(report),
            "detections": report
        })
        print(f"  Frame {i}/{len(test_files)}: {fname} -> Found {len(report)} potholes (Output: {os.path.basename(out_img)})")
        
    summary_path = os.path.join(out_dir, "pipeline_batch_report.json")
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=4)
        
    print(f"\n[SUCCESS] Batch Pipeline completed! Report saved to: {summary_path}")
    return all_results


if __name__ == "__main__":
    run_batch_evaluation(max_images=5)

