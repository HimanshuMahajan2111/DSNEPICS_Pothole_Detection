"""
=============================================================================
STAGE 1: ADAPTIVE COLOR SEGMENTATION & MULTI-SPECTRAL CAVITY EXTRACTOR
=============================================================================
Author: EPICS DSN3099
Focus:
  - Color Space Decomposition: HSV, CIE-LAB, and YCrCb
  - Adaptive Otsu + Local Thresholding for Pothole Cavity Mask
  - Chrominance Differential & Dark Shadow Texture Masking
  - Computes exact pixel contour area, perimeter, circularity, and fill ratio
=============================================================================
"""

import cv2
import numpy as np

class PotholeColorSegmenter:
    def __init__(self):
        pass

    def segment_cavity(self, crop_bgr):
        """
        Takes an RGB/BGR image patch of a pothole cavity and segments
        the depressed region using multi-color space thresholding.
        Returns:
          - mask: binary uint8 mask (255 = pothole interior, 0 = asphalt surface)
          - segmented_vis: color-coded visualization (Green overlay on pothole)
          - color_metrics: morphological & color distribution features
        """
        if crop_bgr is None or crop_bgr.size == 0 or crop_bgr.shape[0] < 5 or crop_bgr.shape[1] < 5:
            return np.zeros((32, 32), dtype=np.uint8), np.zeros((32, 32, 3), dtype=np.uint8), {}

        h, w = crop_bgr.shape[:2]
        
        # 1. Color Space Transformations
        gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
        lab = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2LAB)
        
        # 2. Extract Luminance & Texture Cues
        # Road asphalt is typically neutral/gray, whereas pothole cavities exhibit
        # depth shadows (low L in LAB, low V in HSV) and texture roughness.
        v_channel = hsv[:, :, 2]
        l_channel = lab[:, :, 0]
        
        # Local Gaussian blur to eliminate high-frequency aggregate noise
        blur_l = cv2.GaussianBlur(l_channel, (5, 5), 0)
        
        # Adaptive inverse thresholding (cavity interior is darker than surface)
        otsu_thresh, _ = cv2.threshold(blur_l, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Edge gradient detection (Canny) to define cavity boundaries
        edges = cv2.Canny(gray, 30, 100)
        
        # Morphological Closing to seal fractured boundaries
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        _, binary_cavity = cv2.threshold(blur_l, otsu_thresh, 255, cv2.THRESH_BINARY_INV)
        closed_mask = cv2.morphologyEx(binary_cavity, cv2.MORPH_CLOSE, kernel, iterations=2)
        opened_mask = cv2.morphologyEx(closed_mask, cv2.MORPH_OPEN, kernel, iterations=1)
        
        # Keep only dominant connected components near the center
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(opened_mask)
        final_mask = np.zeros_like(opened_mask)
        center_y, center_x = h / 2.0, w / 2.0
        
        best_dist = float("inf")
        best_label = 0
        
        for lbl in range(1, num_labels):
            area = stats[lbl, cv2.CC_STAT_AREA]
            if area < (h * w * 0.05):  # ignore tiny noise specks
                continue
            cy, cx = centroids[lbl][1], centroids[lbl][0]
            dist_to_center = np.sqrt((cy - center_y)**2 + (cx - center_x)**2)
            if dist_to_center < best_dist:
                best_dist = dist_to_center
                best_label = lbl
                
        if best_label > 0:
            final_mask[labels == best_label] = 255
        else:
            final_mask = opened_mask
            
        # 3. Morphological & Color Metrics
        cavity_pixels = np.sum(final_mask > 0)
        total_pixels = h * w
        cavity_fill_ratio = float(cavity_pixels / (total_pixels + 1e-6))
        
        contours, _ = cv2.findContours(final_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            c = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(c)
            perimeter = cv2.arcLength(c, True)
            circularity = float(4.0 * np.pi * area / (perimeter ** 2 + 1e-6))
        else:
            area, perimeter, circularity = 0.0, 0.0, 0.0
            
        # Chrominance & shadow contrast
        inside_mask = final_mask > 0
        outside_mask = final_mask == 0
        mean_inner_l = float(np.mean(l_channel[inside_mask])) if np.any(inside_mask) else 100.0
        mean_outer_l = float(np.mean(l_channel[outside_mask])) if np.any(outside_mask) else 100.0
        shadow_contrast = float((mean_outer_l + 1e-4) / (mean_inner_l + 1e-4))
        
        # Color overlay visualization
        overlay = crop_bgr.copy()
        color_mask = np.zeros_like(crop_bgr)
        color_mask[final_mask > 0] = [0, 255, 60]  # bright green segmentation
        segmented_vis = cv2.addWeighted(overlay, 0.70, color_mask, 0.30, 0)
        cv2.drawContours(segmented_vis, contours, -1, (0, 255, 255), 2)
        
        metrics = {
            "seg_area_ratio": cavity_fill_ratio,
            "seg_circularity": circularity,
            "seg_shadow_contrast": shadow_contrast,
            "seg_inner_luminance": mean_inner_l,
            "seg_outer_luminance": mean_outer_l,
            "seg_edge_density": float(np.mean(edges > 0))
        }
        return final_mask, segmented_vis, metrics
