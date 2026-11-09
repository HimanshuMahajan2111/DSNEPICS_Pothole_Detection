"""
Video Frame Extraction & Preprocessing Module
Extracts temporally downsampled frames (5 FPS) from dashcam MP4 videos,
applies CLAHE enhancement, and builds a video frame index for tracking.
"""

import os
import glob
import cv2
import pandas as pd

DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "data"))
VIDEO_INPUT_DIR = os.path.join(DATA_DIR, "combined", "ROAD SURFACE", "Road Surface Video")
OUTPUT_BASE = os.path.join(DATA_DIR, "processed", "video_frames")

def apply_clahe(image):
    """Enhance road contrast using CLAHE on the luminance channel."""
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    enhanced = cv2.merge((cl, a, b))
    return cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)

def extract_video_frames(target_fps=5, max_videos=None):
    """
    Extract frames at target_fps from dashcam videos.
    """
    os.makedirs(OUTPUT_BASE, exist_ok=True)
    video_files = sorted(glob.glob(os.path.join(VIDEO_INPUT_DIR, "*.mp4")))
    
    if max_videos:
        video_files = video_files[:max_videos]
        
    print(f"Found {len(video_files)} video files in: {VIDEO_INPUT_DIR}")
    
    index_records = []
    
    for v_idx, v_path in enumerate(video_files):
        v_name = os.path.splitext(os.path.basename(v_path))[0].replace(" ", "_")
        v_out_dir = os.path.join(OUTPUT_BASE, v_name)
        os.makedirs(v_out_dir, exist_ok=True)
        
        cap = cv2.VideoCapture(v_path)
        if not cap.isOpened():
            print(f"Warning: Could not open {v_path}")
            continue
            
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration = total_frames / fps if fps > 0 else 0
        
        step = max(1, int(round(fps / target_fps)))
        frame_count = 0
        extracted_count = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            if frame_count % step == 0:
                enhanced = apply_clahe(frame)
                frame_filename = f"frame_{extracted_count:05d}.jpg"
                save_path = os.path.join(v_out_dir, frame_filename)
                cv2.imwrite(save_path, enhanced)
                
                timestamp = frame_count / fps
                index_records.append({
                    'video_name': v_name,
                    'frame_idx': extracted_count,
                    'source_frame_num': frame_count,
                    'timestamp_sec': round(timestamp, 3),
                    'frame_path': os.path.relpath(save_path, DATA_DIR),
                    'width': w,
                    'height': h
                })
                extracted_count += 1
                
            frame_count += 1
            
        cap.release()
        print(f"Extracted {extracted_count} frames from {v_name} (orig: {total_frames} frames, {duration:.1f}s)")
        
    df_index = pd.DataFrame(index_records)
    index_path = os.path.join(OUTPUT_BASE, "video_index.csv")
    df_index.to_csv(index_path, index=False)
    print(f"Saved video frame index with {len(df_index)} total frames to: {index_path}")

if __name__ == "__main__":
    extract_video_frames()

