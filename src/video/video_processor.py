"""
Video Stream Processor
======================
Processes raw/sampled consumer dashcam video streams:
1. Samples frames at target decimation frequency (e.g. 5 FPS).
2. Performs pothole bounding box detection per frame.
3. Performs spatio-temporal tracking across sequential frames (Kalman filter / IoU tracker)
   to ensure each physical pothole is logged once rather than duplicated across adjacent frames.
4. Feeds unique detected pothole entities into 3D reconstruction and dynamic validation.
"""

import os
import sys
import glob
import cv2
import numpy as np
import pandas as pd

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.geometry.depth_reconstructor import Pothole3DReconstructor
from src.validation.sensor_cross_validator import SensorCrossValidator

class PotholeTrack:
    def __init__(self, track_id, bbox, frame_idx):
        self.track_id = track_id
        self.bbox = bbox  # [cx, cy, w, h]
        self.history = [bbox]
        self.last_frame = frame_idx
        self.hits = 1
        self.reconstruction = None
        self.validation = None

class VideoStreamProcessor:
    def __init__(self, h_cam=1.25, fov_deg=70.0, y_horizon=0.45):
        self.reconstructor = Pothole3DReconstructor(h_cam=h_cam, fov_deg=fov_deg, y_horizon=y_horizon)
        self.validator = SensorCrossValidator()
        self.tracks = []
        self.next_track_id = 1

    def _compute_iou(self, b1, b2):
        x1_a, y1_a, x2_a, y2_a = b1[0] - b1[2]/2, b1[1] - b1[3]/2, b1[0] + b1[2]/2, b1[1] + b1[3]/2
        x1_b, y1_b, x2_b, y2_b = b2[0] - b2[2]/2, b2[1] - b2[3]/2, b2[0] + b2[2]/2, b2[1] + b2[3]/2
        
        inter_x1 = max(x1_a, x1_b)
        inter_y1 = max(y1_a, y1_b)
        inter_x2 = min(x2_a, x2_b)
        inter_y2 = min(y2_a, y2_b)
        
        if inter_x2 <= inter_x1 or inter_y2 <= inter_y1:
            return 0.0
            
        inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
        area_a = (x2_a - x1_a) * (y2_a - y1_a)
        area_b = (x2_b - x1_b) * (y2_b - y1_b)
        
        return inter_area / (area_a + area_b - inter_area + 1e-6)

    def detect_potholes_in_frame(self, frame_bgr):
        h, w = frame_bgr.shape[:2]
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        
        # Focus strictly on the lower road surface below horizon (cy > 0.50)
        road_y_start = int(h * 0.52)
        road_roi = gray[road_y_start:, :]
        
        # Adaptive thresholding and morphological closure
        blur = cv2.GaussianBlur(road_roi, (7, 7), 0)
        thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 25, 8)
        
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 5))
        closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        
        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        bboxes = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            # Filter noise and large road borders
            if 1500 < area < 45000:
                bx, by, bw, bh = cv2.boundingRect(cnt)
                aspect = float(bw) / float(bh + 1e-6)
                if 0.8 <= aspect <= 4.0:  # Realistic pothole aspect ratio
                    cy_global = (by + road_y_start + bh / 2.0) / h
                    cx_global = (bx + bw / 2.0) / w
                    # Potholes must be within camera lane corridor
                    if 0.15 <= cx_global <= 0.85:
                        bboxes.append([cx_global, cy_global, bw / w, bh / h])
                        
        return bboxes

    def update_tracker(self, detected_bboxes, frame_idx):
        matched_tracks = set()
        matched_dets = set()
        
        for det_idx, det_box in enumerate(detected_bboxes):
            best_iou = 0.25  # Minimum IoU threshold
            best_track = None
            
            for trk in self.tracks:
                if trk.track_id in matched_tracks:
                    continue
                iou = self._compute_iou(trk.bbox, det_box)
                if iou > best_iou:
                    best_iou = iou
                    best_track = trk
                    
            if best_track is not None:
                best_track.bbox = det_box
                best_track.history.append(det_box)
                best_track.last_frame = frame_idx
                best_track.hits += 1
                matched_tracks.add(best_track.track_id)
                matched_dets.add(det_idx)
            else:
                new_track = PotholeTrack(self.next_track_id, det_box, frame_idx)
                self.next_track_id += 1
                self.tracks.append(new_track)
                matched_tracks.add(new_track.track_id)
                matched_dets.add(det_idx)

    def process_video_directory(self, frames_dir, simulated_speed_kmh=35.0, max_frames=30):
        frame_files = sorted(glob.glob(os.path.join(frames_dir, "*.jpg")))[:max_frames]
        print(f"[Video Processor] Processing {len(frame_files)} frames from: {frames_dir}")
        
        unique_potholes = []
        
        for frame_idx, fpath in enumerate(frame_files):
            img = cv2.imread(fpath)
            if img is None:
                continue
                
            dets = self.detect_potholes_in_frame(img)
            self.update_tracker(dets, frame_idx)
            
            # For tracks with >= 2 hits in sweet spot distance zone (cy in [0.65, 0.85]), reconstruct once
            for trk in self.tracks:
                if trk.hits >= 2 and trk.reconstruction is None:
                    cx, cy, w, h = trk.bbox
                    if 0.60 <= cy <= 0.88:
                        geom = self.reconstructor.reconstruct_pothole_geometry(img, trk.bbox)
                        if geom:
                            val = self.validator.validate_pothole_reconstruction(
                                depth_max_cm=geom["depth_max_cm"],
                                length_cm=geom["length_cm"],
                                width_cm=geom["width_cm"],
                                speed_kmh=simulated_speed_kmh
                            )
                            trk.reconstruction = geom
                            trk.validation = val
                            unique_potholes.append({
                                "track_id": trk.track_id,
                                "first_frame": os.path.basename(fpath),
                                "geometry": geom,
                                "validation": val
                            })
                            
        print(f"[Video Processor] Tracking complete! Identified {len(unique_potholes)} unique physically validated potholes.")
        return unique_potholes


if __name__ == "__main__":
    test_video_dir = "d:/DSN3099-EPICS/Implementation/data/processed/video/road_surface_videos/Video_(1)"
    processor = VideoStreamProcessor()
    potholes = processor.process_video_directory(test_video_dir, max_frames=20)
    for p in potholes:
        print(f"Pothole #{p['track_id']} | Depth: {p['geometry']['depth_max_cm']:.1f}cm | Conf: {p['validation']['confidence_score']:.2f} | Status: {p['validation']['validation_status']}")

