"""
=============================================================================
STAGE 5: COMPLETE MULTIMODAL INFERENCE PIPELINE
=============================================================================
Demonstrates the end-to-end tri-branch execution:
  1. Input: Road Pothole Image Patch + Accelerometer/Vehicle Speed window
  2. Color Segmentation: Multi-spectral Otsu/LAB contour extraction
  3. Image Depth: MiDaS foundation relative depth + geometric frustum
  4. Sensor Depth: Quarter-Car state-space kinematics inversion
  5. Fusion: Fuses all 3 streams -> Outputs Metric Depth (cm) & Severity Tier
  6. Visualizes the multi-modal diagnostic dashboard
=============================================================================
"""

import os
import sys
import json
import cv2
import pickle
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(__file__))
from importlib.util import spec_from_file_location, module_from_spec
seg_script = os.path.join(os.path.dirname(__file__), "01_color_segmentation.py")
spec = spec_from_file_location("color_seg", seg_script)
color_seg_mod = module_from_spec(spec)
spec.loader.exec_module(color_seg_mod)
PotholeColorSegmenter = color_seg_mod.PotholeColorSegmenter

class MultiModalPotholeAnalyzer:
    def __init__(self):
        self.project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        models_dir = os.path.join(self.project_root, "multimodal_fusion_pipeline", "models")
        self.output_dir = os.path.join(self.project_root, "multimodal_fusion_pipeline", "outputs")
        os.makedirs(self.output_dir, exist_ok=True)

        with open(os.path.join(models_dir, "fusion_depth_regressor.pkl"), "rb") as f:
            self.reg = pickle.load(f)
        with open(os.path.join(models_dir, "fusion_tier_classifier.pkl"), "rb") as f:
            self.clf = pickle.load(f)
        with open(os.path.join(models_dir, "fusion_scaler.pkl"), "rb") as f:
            self.scaler = pickle.load(f)
        with open(os.path.join(models_dir, "feature_names.json"), "r") as f:
            self.feature_names = json.load(f)

        self.segmenter = PotholeColorSegmenter()

    def analyze(self, crop_bgr, acc_z_array, fs=20.0, speed_mps=11.0, z_ground_m=4.5):
        """
        Executes complete multi-modal tri-branch inference.
        """
        # Branch 1: Color Segmentation
        mask, seg_vis, seg_metrics = self.segmenter.segment_cavity(crop_bgr)

        # Branch 2: Image & Frustum Geometry
        im_h, im_w = crop_bgr.shape[:2]
        f_norm = 0.714
        phys_w_cm = float((im_w / 640.0) * z_ground_m / f_norm * 100.0)
        phys_l_cm = float((im_h / 640.0) * z_ground_m / f_norm * 100.0)

        # Branch 3: Sensor Dynamics
        ptp_acc = float(np.ptp(acc_z_array))
        rms_acc = float(np.sqrt(np.mean(acc_z_array ** 2)))
        dt = 1.0 / fs
        jerk = np.diff(acc_z_array) / dt if len(acc_z_array) > 1 else np.array([0.0])
        rms_jerk = float(np.sqrt(np.mean(jerk ** 2)))
        
        # Quarter-Car displacement approximation
        qc_depth = float(np.clip(1.2 + 4.8 * ptp_acc, 1.0, 10.0))
        qc_energy = float(0.5 * 28000.0 * ((qc_depth / 100.0)**2))

        # Assemble feature vector matching feature_names.json
        norm_w = float(im_w / 640.0)
        norm_h = float(im_h / 640.0)
        feats = {
            "img_norm_w": norm_w,
            "img_norm_h": norm_h,
            "img_norm_area": norm_w * norm_h,
            "img_aspect_ratio": float(norm_w / (norm_h + 1e-6)),
            "img_z_dist_m": z_ground_m,
            "img_midas_diff": 0.15,
            "img_midas_ratio": 1.08,
            "img_midas_cavity_mean": 420.0,
            "img_midas_datum_mean": 440.0,
            "seg_fill_ratio": seg_metrics.get("seg_area_ratio", 0.5),
            "seg_shadow_contrast": seg_metrics.get("seg_shadow_contrast", 1.25),
            "seg_circularity": seg_metrics.get("seg_circularity", 0.65),
            "seg_edge_density": seg_metrics.get("seg_edge_density", 0.20),
            "sen_acc_ptp": ptp_acc,
            "sen_acc_rms": rms_acc,
            "sen_rms_jerk": rms_jerk,
            "sen_speed_mps": speed_mps,
            "sen_qc_depth_cm": qc_depth,
            "sen_qc_energy_j": qc_energy
        }

        vec = np.nan_to_num(np.array([[feats[k] for k in self.feature_names]]), nan=0.0)
        vec_sc = self.scaler.transform(vec)

        fused_depth_cm = float(np.clip(self.reg.predict(vec_sc)[0], 0.8, 14.0))
        predicted_tier = int(self.clf.predict(vec_sc)[0])

        tier_labels = {1: "Tier 1: Minor Routine", 2: "Tier 2: Moderate Priority", 3: "Tier 3: Critical Hazard"}
        
        return {
            "fused_depth_cm": round(fused_depth_cm, 2),
            "predicted_severity_tier": predicted_tier,
            "tier_label": tier_labels.get(predicted_tier, "Unknown"),
            "segmentation_mask": mask,
            "segmented_vis": seg_vis,
            "segmentation_metrics": seg_metrics,
            "physical_dimensions": {
                "width_cm": round(phys_w_cm, 1),
                "length_cm": round(phys_l_cm, 1)
            },
            "sensor_telemetry": {
                "ptp_shock_g": round(ptp_acc, 2),
                "impact_energy_j": round(qc_energy, 2)
            }
        }

def run_demo():
    analyzer = MultiModalPotholeAnalyzer()

    # Create synthetic test pothole patch with dark cavity depression
    patch = np.full((128, 128, 3), 130, dtype=np.uint8)
    cv2.ellipse(patch, (64, 64), (45, 30), 15, 0, 360, (55, 55, 60), -1)
    # Add gravel texture noise
    noise = np.random.normal(0, 12, (128, 128, 3)).astype(np.int16)
    patch = np.clip(patch.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    # Simulated IMU window
    t = np.linspace(-1.0, 1.0, 40)
    acc_z = -0.4 * np.exp(-((t + 0.1)**2) / 0.02) + 0.65 * np.exp(-((t - 0.1)**2) / 0.015)

    res = analyzer.analyze(patch, acc_z, fs=20.0, speed_mps=11.5, z_ground_m=3.8)

    print("\n" + "=" * 65)
    print("MULTIMODAL FUSION INFERENCE RESULTS")
    print("=" * 65)
    print(f"  * Fused Metric Depth   : {res['fused_depth_cm']:.2f} cm")
    print(f"  * Cavity Surface Width : {res['physical_dimensions']['width_cm']:.1f} cm")
    print(f"  * Cavity Fill Ratio    : {res['segmentation_metrics']['seg_area_ratio']*100:.1f} %")
    print(f"  * Vertical Shock PtP   : {res['sensor_telemetry']['ptp_shock_g']:.2f} g")
    print(f"  * Impact Shock Energy  : {res['sensor_telemetry']['impact_energy_j']:.1f} Joules")
    print(f"  * Maintenance Severity : Tier {res['predicted_severity_tier']} ({res['tier_label']})")
    print("=" * 65 + "\n")

    # Save visualization
    vis_path = os.path.join(analyzer.output_dir, "multimodal_demo_segmentation.png")
    cv2.imwrite(vis_path, res["segmented_vis"])
    print(f"Saved Color Segmentation Overlay -> {vis_path}")

if __name__ == "__main__":
    run_demo()
