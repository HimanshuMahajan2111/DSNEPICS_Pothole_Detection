"""
=============================================================================
STEP 3: CALIBRATED MULTI-METRIC, ACCURACY & CV SPATIAL ERROR EVALUATION
=============================================================================
Author: EPICS DSN3099
Focus:
  - Calibrated for Road Surface Cavities (Non-rigid amorphous geometry):
    * Confidence Threshold: 0.15 (calibrated for real-world road texture contrast)
    * IoU Matching Threshold: 0.35 (standard for amorphous non-rigid objects)
  - Evaluates:
    1. Scene-Level Road Surface Classification Accuracy (Target: >= 85%)
    2. Multi-Pothole Cluster Density & Count Accuracy (Tolerance +/-1 and +/-2)
    3. Detection Precision, Recall (Detection Rate), and F1-Score
    4. Computer Vision Spatial Errors: mIoU, Center Distance, Aspect Ratio Distortion
    5. Clean Road, Barrier, and Wall False Alarm Rate (Must be 0.00%)
=============================================================================
"""

import os
import glob
import json
import numpy as np

def compute_box_iou(box1, box2):
    """Computes IoU between two boxes in normalized [cx, cy, w, h] format."""
    cx1, cy1, w1, h1 = box1
    cx2, cy2, w2, h2 = box2
    
    xmin1, ymin1 = cx1 - w1 / 2.0, cy1 - h1 / 2.0
    xmax1, ymax1 = cx1 + w1 / 2.0, cy1 + h1 / 2.0
    
    xmin2, ymin2 = cx2 - w2 / 2.0, cy2 - h2 / 2.0
    xmax2, ymax2 = cx2 + w2 / 2.0, cy2 + h2 / 2.0
    
    inter_xmin = max(xmin1, xmin2)
    inter_ymin = max(ymin1, ymin2)
    inter_xmax = min(xmax1, xmax2)
    inter_ymax = min(ymax1, ymax2)
    
    inter_w = max(0.0, inter_xmax - inter_xmin)
    inter_h = max(0.0, inter_ymax - inter_ymin)
    inter_area = inter_w * inter_h
    
    area1 = w1 * h1
    area2 = w2 * h2
    union_area = area1 + area2 - inter_area
    
    if union_area <= 0:
        return 0.0
    return inter_area / union_area

def run_evaluation(weights_path=None, conf_thresh=0.15, iou_thresh=0.35):
    from ultralytics import YOLO
    import torch

    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    if weights_path is None:
        weights_path = os.path.join(project_root, "pothole_detection_pipeline", "checkpoints", "best_pothole_model.pt")

    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Model checkpoint not found at: {weights_path}")

    test_imgs_dir = os.path.join(project_root, "data", "pothole_detector_dataset", "images", "test")
    test_lbls_dir = os.path.join(project_root, "data", "pothole_detector_dataset", "labels", "test")
    report_out_dir = os.path.join(project_root, "pothole_detection_pipeline", "reports")
    os.makedirs(report_out_dir, exist_ok=True)

    print("\n" + "="*70)
    print("STEP 3: CALIBRATED ROAD CAVITY ACCURACY & SPATIAL ERROR EVALUATION")
    print("="*70)
    print(f"Model Weights      : {weights_path}")
    print(f"Test Images Dir    : {test_imgs_dir}")
    print(f"Calibrated Thresh  : Confidence = {conf_thresh} | Match IoU = {iou_thresh}")

    device = "0" if torch.cuda.is_available() else "cpu"
    model = YOLO(weights_path)

    test_images = sorted(glob.glob(os.path.join(test_imgs_dir, "*.jpg")))
    print(f"Loaded {len(test_images)} test frames for evaluation.")

    ground_truths = {}
    predictions = {}

    for img_p in test_images:
        fname = os.path.basename(img_p)
        lbl_p = os.path.join(test_lbls_dir, fname.replace(".jpg", ".txt"))
        
        gt_boxes = []
        if os.path.exists(lbl_p):
            with open(lbl_p, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        gt_boxes.append([float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])])
        ground_truths[fname] = gt_boxes

        res = model.predict(img_p, conf=conf_thresh, imgsz=640, device=device, verbose=False)[0]
        preds_for_img = []
        if res.boxes is not None and len(res.boxes) > 0:
            boxes_xywhn = res.boxes.xywhn.cpu().numpy()
            confs = res.boxes.conf.cpu().numpy()
            for b, c in zip(boxes_xywhn, confs):
                preds_for_img.append({"box": b.tolist(), "conf": float(c)})
        predictions[fname] = preds_for_img

    total_gt = sum(len(b) for b in ground_truths.values())
    total_pred = sum(len(b) for b in predictions.values())
    
    tp = 0
    fp = 0
    fn = 0
    
    ious = []
    center_dist_norm = []
    center_dist_px = []
    aspect_ratio_errs = []
    area_scale_errs = []
    
    gt_counts = []
    pred_counts = []
    
    img_tp = 0
    img_fn = 0
    img_tn = 0
    img_fp = 0

    img_w, img_h = 640.0, 640.0

    for fname, gt_boxes in ground_truths.items():
        preds = predictions.get(fname, [])
        n_gt = len(gt_boxes)
        n_pred = len(preds)
        
        gt_counts.append(n_gt)
        pred_counts.append(n_pred)

        # Scene Classification (Pothole Road vs Clean Normal Road)
        if n_gt > 0:
            if n_pred > 0:
                img_tp += 1
            else:
                img_fn += 1
        else:
            if n_pred == 0:
                img_tn += 1
            else:
                img_fp += 1

        if n_gt == 0:
            fp += n_pred
            continue

        matched_gt = set()
        sorted_preds = sorted(enumerate(preds), key=lambda x: x[1]['conf'], reverse=True)

        for p_idx, p_dict in sorted_preds:
            p_box = p_dict['box']
            best_iou = 0.0
            best_g_idx = -1

            for g_idx, g_box in enumerate(gt_boxes):
                if g_idx in matched_gt:
                    continue
                iou = compute_box_iou(p_box, g_box)
                if iou > best_iou:
                    best_iou = iou
                    best_g_idx = g_idx

            if best_iou >= iou_thresh:
                matched_gt.add(best_g_idx)
                tp += 1
                g_box = gt_boxes[best_g_idx]
                
                ious.append(best_iou)
                dx = p_box[0] - g_box[0]
                dy = p_box[1] - g_box[1]
                dist_norm = float(np.sqrt(dx**2 + dy**2))
                dist_px = float(np.sqrt((dx * img_w)**2 + (dy * img_h)**2))
                center_dist_norm.append(dist_norm)
                center_dist_px.append(dist_px)

                ar_pred = p_box[2] / (p_box[3] + 1e-6)
                ar_gt = g_box[2] / (g_box[3] + 1e-6)
                aspect_ratio_errs.append(abs(ar_pred - ar_gt))

                area_pred = p_box[2] * p_box[3]
                area_gt = g_box[2] * g_box[3]
                area_scale_errs.append(abs(area_pred - area_gt) / (area_gt + 1e-6))
            else:
                fp += 1

        fn += (len(gt_boxes) - len(matched_gt))

    # ACCURACY CALCULATIONS
    total_frames = len(ground_truths)
    scene_classification_accuracy = (img_tp + img_tn) / (total_frames + 1e-9)

    precision = tp / (tp + fp + 1e-9)
    recall = tp / (tp + fn + 1e-9)
    f1 = 2.0 * (precision * recall) / (precision + recall + 1e-9)
    
    miss_rate = fn / (total_gt + 1e-9)
    fdr = fp / (total_pred + 1e-9) if total_pred > 0 else 0.0
    fppi = fp / total_frames
    clean_frames_count = img_tn + img_fp
    far_clean = img_fp / clean_frames_count if clean_frames_count > 0 else 0.0

    gt_counts = np.array(gt_counts)
    pred_counts = np.array(pred_counts)
    count_err = pred_counts - gt_counts
    mae_count = float(np.mean(np.abs(count_err)))
    rmse_count = float(np.sqrt(np.mean(count_err**2)))
    mse_count = float(np.mean(count_err**2))
    exact_count_acc = float(np.mean(count_err == 0) * 100.0)
    within_one_acc = float(np.mean(np.abs(count_err) <= 1) * 100.0)
    within_two_acc = float(np.mean(np.abs(count_err) <= 2) * 100.0)

    mean_iou = float(np.mean(ious)) if ious else 0.0
    iou_above_50_acc = float(np.mean(np.array(ious) >= 0.50) * 100.0) if ious else 0.0
    center_dist_px_mean = float(np.mean(center_dist_px)) if center_dist_px else 0.0
    center_dist_norm_mean = float(np.mean(center_dist_norm)) if center_dist_norm else 0.0
    jitter_std_px = float(np.std(center_dist_px)) if center_dist_px else 0.0
    ar_err_mean = float(np.mean(aspect_ratio_errs)) if aspect_ratio_errs else 0.0
    area_scale_err_pct = float(np.mean(area_scale_errs) * 100.0) if area_scale_errs else 0.0

    metrics = {
        "dataset_statistics": {
            "total_test_frames": total_frames,
            "negative_clean_wall_frames": clean_frames_count,
            "frames_with_potholes": img_tp + img_fn,
            "total_ground_truth_potholes": total_gt,
            "total_predicted_potholes": total_pred,
            "confusion_matrix_boxes": {"TP": tp, "FP": fp, "FN": fn},
            "confusion_matrix_frames": {"TP": img_tp, "TN": img_tn, "FP": img_fp, "FN": img_fn}
        },
        "accuracy_measurements": {
            "scene_level_classification_accuracy": round(scene_classification_accuracy * 100.0, 2),
            "multi_pothole_cluster_density_accuracy_plus_minus_2": round(within_two_acc, 2),
            "multi_pothole_cluster_density_accuracy_plus_minus_1": round(within_one_acc, 2),
            "multi_pothole_exact_count_accuracy": round(exact_count_acc, 2),
            "bounding_box_localization_precision_iou_ge_50": round(iou_above_50_acc, 2),
            "clean_road_and_wall_false_alarm_rejection_accuracy": round((1.0 - far_clean) * 100.0, 2)
        },
        "standard_detection_metrics": {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1_score": round(f1, 4)
        },
        "detection_error_rates": {
            "miss_rate_fnr": round(miss_rate, 4),
            "false_discovery_rate_fdr": round(fdr, 4),
            "false_positives_per_image_fppi": round(fppi, 4),
            "false_alarm_rate_on_clean_roads": round(far_clean, 4)
        },
        "multi_pothole_counting_metrics": {
            "mae_count": round(mae_count, 4),
            "rmse_count": round(rmse_count, 4),
            "mse_count": round(mse_count, 4)
        },
        "cv_spatial_geometry_errors": {
            "mean_iou_on_true_potholes": round(mean_iou, 4),
            "center_distance_error_pixels": round(center_dist_px_mean, 2),
            "center_distance_error_normalized": round(center_dist_norm_mean, 4),
            "center_localization_jitter_std_pixels": round(jitter_std_px, 2),
            "aspect_ratio_distortion_error": round(ar_err_mean, 4),
            "area_scale_relative_error_pct": round(area_scale_err_pct, 2)
        }
    }

    report_file = os.path.join(report_out_dir, "pothole_evaluation_report.json")
    with open(report_file, "w") as f:
        json.dump(metrics, f, indent=4)

    print("\n" + "="*70)
    print("POTHOLE DETECTOR CALIBRATED ACCURACY & ERROR REPORT")
    print("="*70)
    print(f"\n1. ACCURACY MEASUREMENTS (TARGET: >= 85%):")
    print(f"   Scene-Level Road Classification Accuracy       : {metrics['accuracy_measurements']['scene_level_classification_accuracy']:.2f}%")
    print(f"   Multi-Pothole Cluster Accuracy (+/-2 potholes) : {metrics['accuracy_measurements']['multi_pothole_cluster_density_accuracy_plus_minus_2']:.2f}%")
    print(f"   Multi-Pothole Cluster Accuracy (+/-1 pothole)  : {metrics['accuracy_measurements']['multi_pothole_cluster_density_accuracy_plus_minus_1']:.2f}%")
    print(f"   Wall / Clean Road False Alarm Rejection Acc    : {metrics['accuracy_measurements']['clean_road_and_wall_false_alarm_rejection_accuracy']:.2f}%")
    print(f"   Detected Cavity Localization Accuracy (IoU>=50): {metrics['accuracy_measurements']['bounding_box_localization_precision_iou_ge_50']:.2f}%")
    
    print(f"\n2. DETECTION METRICS:")
    print(f"   Precision                                      : {metrics['standard_detection_metrics']['precision']*100:.2f}%")
    print(f"   Recall (Detection Rate)                        : {metrics['standard_detection_metrics']['recall']*100:.2f}%")
    print(f"   F1-Score                                       : {metrics['standard_detection_metrics']['f1_score']:.4f}")
    
    print(f"\n3. ERROR RATES:")
    print(f"   Miss Rate (FNR = 1 - Recall)                   : {metrics['detection_error_rates']['miss_rate_fnr']*100:.2f}%")
    print(f"   False Discovery Rate (FDR = 1 - Precision)     : {metrics['detection_error_rates']['false_discovery_rate_fdr']*100:.2f}%")
    print(f"   False Positives / Image (FPPI)                 : {metrics['detection_error_rates']['false_positives_per_image_fppi']:.3f}")
    print(f"   False Alarm Rate on Clean Roads/Walls          : {metrics['detection_error_rates']['false_alarm_rate_on_clean_roads']*100:.2f}%")
    
    print(f"\n4. MULTI-POTHOLE COUNT ESTIMATION ERRORS:")
    print(f"   Mean Absolute Error (MAE)                      : {metrics['multi_pothole_counting_metrics']['mae_count']:.3f} potholes/frame")
    print(f"   Root Mean Squared Error (RMSE)                 : {metrics['multi_pothole_counting_metrics']['rmse_count']:.3f}")
    
    print(f"\n5. COMPUTER VISION SPATIAL GEOMETRY ERRORS:")
    print(f"   Mean IoU on Detected Cavities                  : {metrics['cv_spatial_geometry_errors']['mean_iou_on_true_potholes']*100:.2f}%")
    print(f"   Center Euclidean Distance Error                : {metrics['cv_spatial_geometry_errors']['center_distance_error_pixels']:.2f} px ({metrics['cv_spatial_geometry_errors']['center_distance_error_normalized']*100:.2f}% normalized)")
    print(f"   Center Localization Jitter (Std)               : {metrics['cv_spatial_geometry_errors']['center_localization_jitter_std_pixels']:.2f} px")
    print(f"   Aspect Ratio Distortion Error                  : {metrics['cv_spatial_geometry_errors']['aspect_ratio_distortion_error']:.3f}")
    print(f"   Area Scale Relative Error                      : {metrics['cv_spatial_geometry_errors']['area_scale_relative_error_pct']:.2f}%")
    print("="*70)
    print(f"Full JSON report saved to: {report_file}\n")
    return metrics

if __name__ == "__main__":
    run_evaluation()
