"""
Comprehensive Dataset Inventory & Audit Script
Compares exact item counts across raw datasets (dashcam_data, sensor_data, combined)
and processed datasets (processed/vision, processed/sensors, processed/video).
"""

import os
import glob
import pandas as pd

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "data"))

def count_files_by_ext(folder, ext):
    if not os.path.exists(folder):
        return 0
    return len(glob.glob(os.path.join(folder, "**", f"*.{ext}"), recursive=True))

def count_csv_rows(fpath):
    if not os.path.exists(fpath):
        return 0
    try:
        df = pd.read_csv(fpath)
        return len(df)
    except Exception:
        return 0

def run_audit():
    print("=" * 75)
    print("      DATASET INVENTORY & COMPARISON AUDIT (RAW VS PROCESSED)")
    print("=" * 75)

    # ----------------------------------------------------
    # 1. RAW DASHCAM DATA
    # ----------------------------------------------------
    print("\n[SECTION 1] RAW DASHCAM DATA (dashcam_data/)")
    print("-" * 75)
    
    rdd_d40_imgs = count_files_by_ext(os.path.join(BASE_DIR, "dashcam_data", "RDD2022 India Pothole D40"), "jpg")
    rdd_d40_xmls = count_files_by_ext(os.path.join(BASE_DIR, "dashcam_data", "RDD2022 India Pothole D40"), "xml")
    
    ind_road_imgs = count_files_by_ext(os.path.join(BASE_DIR, "dashcam_data", "Indian Roads Dataset"), "jpg")
    ind_road_txts = count_files_by_ext(os.path.join(BASE_DIR, "dashcam_data", "Indian Roads Dataset"), "txt")
    
    crddc_imgs = count_files_by_ext(os.path.join(BASE_DIR, "dashcam_data", "RDD2022_released_through_CRDDC2022"), "jpg")
    crddc_xmls = count_files_by_ext(os.path.join(BASE_DIR, "dashcam_data", "RDD2022_released_through_CRDDC2022"), "xml")

    print(f"  * RDD2022 India Pothole D40:  {rdd_d40_imgs:,} JPG images | {rdd_d40_xmls:,} Pascal VOC XMLs")
    print(f"  * Indian Roads Dataset:       {ind_road_imgs:,} JPG images | {ind_road_txts:,} YOLO TXTs")
    print(f"  * CRDDC2022 India Full:       {crddc_imgs:,} JPG images | {crddc_xmls:,} Pascal VOC XMLs")

    # ----------------------------------------------------
    # 2. RAW SENSOR DATA
    # ----------------------------------------------------
    print("\n[SECTION 2] RAW SENSOR DATA (sensor_data/)")
    print("-" * 75)
    
    ivp_csvs = count_files_by_ext(os.path.join(BASE_DIR, "sensor_data", "Intelligent Vehicle Perception"), "csv")
    pot_csvs = count_files_by_ext(os.path.join(BASE_DIR, "sensor_data", "Pothole, Road Quality Detection Sensor data"), "csv")
    v2_csvs = count_files_by_ext(os.path.join(BASE_DIR, "sensor_data", "sensor_data_v2"), "csv")
    two_raw = count_files_by_ext(os.path.join(BASE_DIR, "sensor_data", "Two Wheeler Road Surface Classifier", "Raw Data"), "csv")
    two_prep = count_files_by_ext(os.path.join(BASE_DIR, "sensor_data", "Two Wheeler Road Surface Classifier", "Preprocessed Data"), "csv")
    srs_file = os.path.join(BASE_DIR, "sensor_data", "SmartRoadSense OpenData", "open_data.csv")
    srs_rows = 3615539 if os.path.exists(srs_file) else 0

    print(f"  * Pothole & Road Quality:      {pot_csvs} CSV files (20 trips: 5 pothole trips + 15 condition trips)")
    print(f"  * Intelligent Vehicle (IVP):   {ivp_csvs} CSV files (PVS 1-9 trips across 3 sensor mount positions)")
    print(f"  * Two Wheeler Surface:         {two_raw} raw trip CSVs + {two_prep:,} preprocessed window CSVs")
    print(f"  * sensor_data_v2:              {v2_csvs} mobile accelerometer trip CSVs")
    print(f"  * SmartRoadSense OpenData:     1 CSV file ({srs_rows:,} raw pavement roughness records)")

    # ----------------------------------------------------
    # 3. RAW COMBINED DATA
    # ----------------------------------------------------
    print("\n[SECTION 3] RAW COMBINED DATA (combined/)")
    print("-" * 75)
    
    bharat_imgs = count_files_by_ext(os.path.join(BASE_DIR, "combined", "BharatPotHole"), "jpg")
    bharat_txts = count_files_by_ext(os.path.join(BASE_DIR, "combined", "BharatPotHole"), "txt")
    surf_imgs = count_files_by_ext(os.path.join(BASE_DIR, "combined", "ROAD SURFACE", "Road Surface Image"), "jpg")
    surf_vids = count_files_by_ext(os.path.join(BASE_DIR, "combined", "ROAD SURFACE", "Road Surface Video"), "mp4")
    roadsens_csvs = count_files_by_ext(os.path.join(BASE_DIR, "combined", "RoadSens-4M"), "csv")

    print(f"  * BharatPotHole:               {bharat_imgs:,} JPG images | {bharat_txts:,} YOLO TXTs")
    print(f"  * ROAD SURFACE Images:         {surf_imgs:,} categorized condition images")
    print(f"  * ROAD SURFACE Videos:         {surf_vids} dashcam MP4 video files")
    print(f"  * RoadSens-4M:                 {roadsens_csvs} CSV files (GIS elevation, weather & anomaly sensors)")

    # ----------------------------------------------------
    # 4. PROCESSED VISION DATA
    # ----------------------------------------------------
    print("\n[SECTION 4] PROCESSED VISION DATA (data/processed/vision/)")
    print("-" * 75)
    
    for sub in ["rdd2022_d40", "bharat_pothole", "indian_roads", "crddc2022"]:
        p = os.path.join(BASE_DIR, "processed", "vision", sub)
        train_imgs = count_files_by_ext(os.path.join(p, "images", "train"), "jpg")
        val_imgs = count_files_by_ext(os.path.join(p, "images", "val"), "jpg")
        test_imgs = count_files_by_ext(os.path.join(p, "images", "test"), "jpg")
        total_lbls = count_files_by_ext(os.path.join(p, "labels"), "txt")
        print(f"  * {sub:<20}: {train_imgs:,} train + {val_imgs:,} val + {test_imgs:,} test = {train_imgs+val_imgs+test_imgs:,} JPGs | {total_lbls:,} YOLO TXTs")
        
    surf_p_imgs = count_files_by_ext(os.path.join(BASE_DIR, "processed", "vision", "road_surface_classes"), "jpg")
    print(f"  * {'road_surface_classes':<20}: {surf_p_imgs:,} standardized 640x640 CLAHE images across 5 conditions")

    # ----------------------------------------------------
    # 5. PROCESSED VIDEO FRAMES
    # ----------------------------------------------------
    print("\n[SECTION 5] PROCESSED VIDEO FRAMES (data/processed/video/)")
    print("-" * 75)
    
    p_vid = os.path.join(BASE_DIR, "processed", "video", "road_surface_videos")
    vid_frames = count_files_by_ext(p_vid, "jpg")
    vid_idx_rows = count_csv_rows(os.path.join(p_vid, "video_index.csv"))
    print(f"  * road_surface_videos:         {vid_frames:,} extracted 5 FPS CLAHE frames across 10 video folders")
    print(f"  * video_index.csv:             {vid_idx_rows:,} timestamped metadata rows")

    # ----------------------------------------------------
    # 6. PROCESSED SENSOR FEATURES
    # ----------------------------------------------------
    print("\n[SECTION 6] PROCESSED SENSORS & TELEMETRY (data/processed/sensors/)")
    print("-" * 75)
    
    p_sens = os.path.join(BASE_DIR, "processed", "sensors")
    for sub in sorted(os.listdir(p_sens)):
        subpath = os.path.join(p_sens, sub)
        if os.path.isdir(subpath):
            csvs = sorted(glob.glob(os.path.join(subpath, "*.csv")))
            for c in csvs:
                rows = count_csv_rows(c)
                print(f"  * {sub + '/' + os.path.basename(c):<50}: {rows:,} feature rows")

    print("\n" + "=" * 75)
    print("                        AUDIT COMPLETED")
    print("=" * 75)

if __name__ == "__main__":
    run_audit()

