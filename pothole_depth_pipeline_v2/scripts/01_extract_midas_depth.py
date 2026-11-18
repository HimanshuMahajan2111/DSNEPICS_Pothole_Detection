"""
=============================================================================
STEP 1: EXTRACT REAL DEPTH MAPS USING MiDaS FOUNDATION MODEL
=============================================================================
Uses Intel MiDaS DPT-Hybrid (trained on 12 real-world RGB-D datasets) to
produce per-pixel relative depth maps.  For each pothole bounding box:
  1. Extract the MiDaS depth inside the cavity.
  2. Extract the MiDaS depth on the surrounding road datum ring.
  3. Compute depth_differential = datum_depth - cavity_depth.
  4. Apply metric calibration via camera geometry.

This replaces the v1 approach of using hand-crafted luminance formulas.
=============================================================================
"""

import os
import sys
import glob
import json
import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image

# ── Camera calibration constants ──
H_CAM = 1.25         # Camera height above pavement (m)
Y_HORIZON = 0.45     # Calibrated horizon line in normalised image coords
FOV_H_RAD = 1.2217   # ~70° horizontal FOV

def load_midas_model(device):
    """Load MiDaS DPT-Hybrid via torch.hub."""
    print("[MiDaS] Downloading / loading DPT-Hybrid model...")
    model = torch.hub.load("intel-isl/MiDaS", "DPT_Hybrid", trust_repo=True)
    model.to(device).eval()

    midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms", trust_repo=True)
    transform = midas_transforms.dpt_transform
    print(f"[MiDaS] Model loaded on {device}.")
    return model, transform


def compute_depth_map(model, transform, img_bgr, device):
    """Run MiDaS on a BGR image → returns relative inverse-depth map (H×W float32)."""
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    input_batch = transform(img_rgb).to(device)

    with torch.no_grad():
        prediction = model(input_batch)
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=img_bgr.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()

    depth_map = prediction.cpu().numpy()
    return depth_map  # Higher value = closer to camera


def extract_pothole_depth_features(depth_map, cx, cy, w, h, img_h, img_w):
    """
    Given MiDaS inverse-depth map and normalised bbox, compute:
      - cavity_depth_mean: mean inverse-depth inside the bbox
      - datum_depth_mean:  mean inverse-depth in a surrounding ring
      - depth_differential: datum - cavity (positive if cavity is deeper)
      - depth_ratio: cavity / datum
      - depth_std: std of inverse-depth inside cavity (texture/roughness proxy)
    """
    x1 = max(0, int((cx - w / 2.0) * img_w))
    y1 = max(0, int((cy - h / 2.0) * img_h))
    x2 = min(img_w, int((cx + w / 2.0) * img_w))
    y2 = min(img_h, int((cy + h / 2.0) * img_h))

    if x2 - x1 < 4 or y2 - y1 < 4:
        return None

    cavity_region = depth_map[y1:y2, x1:x2]
    cavity_mean = float(np.mean(cavity_region))
    cavity_std = float(np.std(cavity_region))
    cavity_min = float(np.min(cavity_region))
    cavity_max = float(np.max(cavity_region))

    # Surrounding datum ring (expand bbox by 30%)
    pad_x = max(4, int((x2 - x1) * 0.30))
    pad_y = max(4, int((y2 - y1) * 0.30))
    rx1 = max(0, x1 - pad_x)
    ry1 = max(0, y1 - pad_y)
    rx2 = min(img_w, x2 + pad_x)
    ry2 = min(img_h, y2 + pad_y)

    # Create mask: outer ring only (exclude inner cavity)
    outer_mask = np.zeros_like(depth_map, dtype=bool)
    outer_mask[ry1:ry2, rx1:rx2] = True
    outer_mask[y1:y2, x1:x2] = False

    if np.sum(outer_mask) < 10:
        datum_mean = cavity_mean
    else:
        datum_mean = float(np.mean(depth_map[outer_mask]))

    depth_diff = datum_mean - cavity_mean  # positive = cavity is farther from camera = deeper
    depth_ratio = cavity_mean / (datum_mean + 1e-8)

    return {
        "midas_cavity_mean": cavity_mean,
        "midas_cavity_std": cavity_std,
        "midas_cavity_min": cavity_min,
        "midas_cavity_max": cavity_max,
        "midas_datum_mean": datum_mean,
        "midas_depth_diff": depth_diff,
        "midas_depth_ratio": depth_ratio,
    }


def geometric_metric_calibration(cy, norm_w, norm_h, midas_depth_diff,
                                  midas_datum_mean, img_w_px, img_h_px):
    """
    Convert MiDaS relative depth differential to metric estimates.
    Uses frustum unprojection from known camera height.
    """
    # Ground-plane distance Z from camera
    if cy <= Y_HORIZON + 0.02:
        z_dist = 10.0  # far away fallback
    else:
        z_dist = H_CAM / (cy - Y_HORIZON)
    z_dist = np.clip(z_dist, 1.0, 15.0)

    # Physical width & length from bbox + Z
    f_norm = 0.5 / np.tan(FOV_H_RAD / 2.0)
    phys_width_m = norm_w * z_dist / f_norm
    phys_length_m = norm_h * z_dist / f_norm

    # Metric depth from MiDaS differential:
    # MiDaS outputs inverse depth. The differential tells us how much deeper
    # the cavity is relative to the road surface.
    # We calibrate: depth_cm ≈ k * |diff| / datum_mean * Z * 100
    # where k is a learned calibration constant (we'll learn it in training)
    # For now, store the raw features and let the model learn the mapping.

    return {
        "z_dist_m": float(z_dist),
        "phys_width_cm": float(phys_width_m * 100.0),
        "phys_length_cm": float(phys_length_m * 100.0),
        "bbox_area_m2": float(phys_width_m * phys_length_m),
    }


def main():
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    geom_csv = os.path.join(project_root, "data", "processed", "features",
                            "vision_engineered_geometry.csv")
    out_dir = os.path.join(project_root, "pothole_depth_pipeline_v2", "data")
    os.makedirs(out_dir, exist_ok=True)

    print("\n" + "=" * 70)
    print("STEP 1: MiDaS FOUNDATION DEPTH EXTRACTION")
    print("=" * 70)

    if not os.path.exists(geom_csv):
        print(f"ERROR: Geometry CSV not found: {geom_csv}")
        sys.exit(1)

    df = pd.read_csv(geom_csv)
    print(f"Loaded {len(df)} pothole geometry records.")

    # Build image index
    img_map = {}
    search_dirs = [
        os.path.join(project_root, "data", "processed", "vision"),
        os.path.join(project_root, "data", "dashcam_data"),
        os.path.join(project_root, "data", "pothole_detector_dataset", "images", "train"),
        os.path.join(project_root, "data", "pothole_detector_dataset", "images", "val"),
        os.path.join(project_root, "data", "pothole_detector_dataset", "images", "test"),
    ]
    for sdir in search_dirs:
        if not os.path.isdir(sdir):
            continue
        for ext in ["*.jpg", "*.png", "*.jpeg"]:
            for p in glob.glob(os.path.join(sdir, "**", ext), recursive=True):
                fname = os.path.basename(p)
                if fname not in img_map:
                    img_map[fname] = p
    print(f"Indexed {len(img_map)} images on disk.")

    # Load MiDaS
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, transform = load_midas_model(device)

    # Process each pothole record
    records = []
    processed_images = {}  # Cache depth maps per image
    skipped = 0

    for idx, row in df.iterrows():
        src_name = str(row["source_file"]).replace(".txt", ".jpg")
        img_path = img_map.get(src_name)

        cx = float(row["norm_cx"])
        cy = float(row["norm_cy"])
        w = float(row["norm_w"])
        h = float(row["norm_h"])

        # Skip potholes too close to horizon
        if cy <= Y_HORIZON + 0.02:
            skipped += 1
            continue

        z_dist = H_CAM / (cy - Y_HORIZON)
        if z_dist > 15.0 or z_dist < 1.0:
            skipped += 1
            continue

        # Ground-truth from geometry CSV (formula-derived, used as reference)
        gt_depth_cm = float(row["est_depth_cm"])
        gt_width_cm = float(row["phys_width_m"]) * 100.0
        gt_length_cm = float(row["phys_length_m"]) * 100.0
        gt_volume = float(row["volume_cm3"])
        gt_tier = int(row["severity_tier"])

        # Run MiDaS depth
        midas_feats = None
        if img_path and os.path.exists(img_path):
            if img_path not in processed_images:
                img_bgr = cv2.imread(img_path)
                if img_bgr is not None:
                    depth_map = compute_depth_map(model, transform, img_bgr, device)
                    processed_images[img_path] = (depth_map, img_bgr.shape[:2])
                else:
                    processed_images[img_path] = None

            cached = processed_images.get(img_path)
            if cached is not None:
                depth_map, (img_h, img_w) = cached
                midas_feats = extract_pothole_depth_features(
                    depth_map, cx, cy, w, h, img_h, img_w
                )

        if midas_feats is None:
            skipped += 1
            continue

        # Geometric calibration
        geo = geometric_metric_calibration(
            cy, w, h, midas_feats["midas_depth_diff"],
            midas_feats["midas_datum_mean"], 640, 640
        )

        # Optical cues from original image
        if img_path and img_path in processed_images and processed_images[img_path] is not None:
            _, (img_h, img_w) = processed_images[img_path]
            img_bgr = cv2.imread(img_path)
            x1 = max(0, int((cx - w / 2.0) * img_w))
            y1 = max(0, int((cy - h / 2.0) * img_h))
            x2 = min(img_w, int((cx + w / 2.0) * img_w))
            y2 = min(img_h, int((cy + h / 2.0) * img_h))

            if x2 - x1 >= 4 and y2 - y1 >= 4:
                crop = img_bgr[y1:y2, x1:x2]
                gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
                texture_var = float(np.var(gray))
                edge_density = float(np.mean(cv2.Canny(gray, 50, 150) > 0))
                mean_intensity = float(np.mean(gray))
            else:
                texture_var = 0.0
                edge_density = 0.0
                mean_intensity = 128.0
        else:
            texture_var = 0.0
            edge_density = 0.0
            mean_intensity = 128.0

        record = {
            "sample_id": idx,
            "source_file": src_name,
            # Bounding box geometry
            "norm_cx": cx,
            "norm_cy": cy,
            "norm_w": w,
            "norm_h": h,
            "aspect_ratio": w / (h + 1e-6),
            "norm_area": w * h,
            # Geometric calibration
            "z_dist_m": geo["z_dist_m"],
            "phys_width_cm_geo": geo["phys_width_cm"],
            "phys_length_cm_geo": geo["phys_length_cm"],
            "bbox_area_m2": geo["bbox_area_m2"],
            # MiDaS depth features (REAL foundation model output)
            "midas_cavity_mean": midas_feats["midas_cavity_mean"],
            "midas_cavity_std": midas_feats["midas_cavity_std"],
            "midas_cavity_min": midas_feats["midas_cavity_min"],
            "midas_cavity_max": midas_feats["midas_cavity_max"],
            "midas_datum_mean": midas_feats["midas_datum_mean"],
            "midas_depth_diff": midas_feats["midas_depth_diff"],
            "midas_depth_ratio": midas_feats["midas_depth_ratio"],
            # Visual texture cues
            "texture_variance": texture_var,
            "edge_density": edge_density,
            "mean_intensity": mean_intensity,
            # Reference ground truth (formula-derived, for calibration)
            "ref_depth_cm": gt_depth_cm,
            "ref_width_cm": gt_width_cm,
            "ref_length_cm": gt_length_cm,
            "ref_volume_cm3": gt_volume,
            "severity_tier": gt_tier,
        }
        records.append(record)

        if (idx + 1) % 200 == 0:
            print(f"  Processed {idx + 1}/{len(df)} records "
                  f"({len(records)} valid, {skipped} skipped)")

    print(f"\nDone. {len(records)} valid samples, {skipped} skipped.")

    # Save full dataset
    df_out = pd.DataFrame(records)
    out_csv = os.path.join(out_dir, "midas_depth_features.csv")
    df_out.to_csv(out_csv, index=False)
    print(f"Saved → {out_csv}")

    # Print MiDaS feature statistics
    print("\n" + "-" * 50)
    print("MiDaS DEPTH FEATURE STATISTICS:")
    print("-" * 50)
    for col in ["midas_depth_diff", "midas_depth_ratio", "midas_cavity_mean",
                "midas_datum_mean", "ref_depth_cm"]:
        print(f"  {col}: mean={df_out[col].mean():.4f}  std={df_out[col].std():.4f}  "
              f"min={df_out[col].min():.4f}  max={df_out[col].max():.4f}")

    # Check correlation between MiDaS features and reference depth
    from scipy import stats
    corr_diff, p_diff = stats.pearsonr(df_out["midas_depth_diff"], df_out["ref_depth_cm"])
    corr_ratio, p_ratio = stats.pearsonr(df_out["midas_depth_ratio"], df_out["ref_depth_cm"])
    print(f"\n  Correlation(midas_depth_diff, ref_depth): r={corr_diff:.4f}  p={p_diff:.2e}")
    print(f"  Correlation(midas_depth_ratio, ref_depth): r={corr_ratio:.4f}  p={p_ratio:.2e}")
    print("=" * 70)


if __name__ == "__main__":
    main()
